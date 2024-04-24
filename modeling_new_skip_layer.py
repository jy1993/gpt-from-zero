import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import random
from utils import build_chat_input, logits_processor
is_flash_attn_available = True
try:
	from flash_attn import flash_attn_func
except:
	is_flash_attn_available = False

class RMSNorm(nn.Module):
	"""docstring for RMSNorm"""
	def __init__(self, hidden_size, eps=1e-6):
		super(RMSNorm, self).__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.eps = eps

	def forward(self, x):
		variance = x.float().pow(2).mean(dim=-1, keepdim=True)
		x = x * torch.rsqrt(variance + self.eps)
		if self.weight.dtype in [torch.float16, torch.bfloat16]:
			x = x.to(self.weight.dtype)
		return self.weight * x

def rotate_half(x):
	x0, x1 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
	return torch.concat((-x1, x0), dim=-1)

class RotaryEmbedding(nn.Module):
	"""docstring for RotaryEmbedding"""
	def __init__(self, dim, seq_len=2048, base_freq=10000):
		super(RotaryEmbedding, self).__init__()
		self.inv_freqs = 1 / (base_freq ** (torch.arange(0, dim, 2) / dim))
		t = torch.outer(torch.arange(seq_len), self.inv_freqs)
		cos, sin = torch.cos(t), torch.sin(t)
		self.cos = torch.cat([cos, cos], dim=-1)
		self.sin = torch.cat([sin, sin], dim=-1)

	def forward(self, x, position_ids):
		self.cos = self.cos.to(x.device)
		self.sin = self.sin.to(x.device)
		cos = self.cos[position_ids].unsqueeze(2).to(x.dtype)
		sin = self.sin[position_ids].unsqueeze(2).to(x.dtype)
		return x * cos + rotate_half(x) * sin

class SelfAttn(nn.Module):
	def __init__(self, num_heads, num_hid_dim, hidden_size, dropout):
		super(SelfAttn, self).__init__()
		self.num_heads = num_heads
		self.num_hid_dim = num_hid_dim
		self.hidden_size = hidden_size
		assert self.num_heads * self.num_hid_dim == self.hidden_size
		self.query_linear = nn.Linear(hidden_size, hidden_size, bias=False)
		self.key_linear = nn.Linear(hidden_size, hidden_size, bias=False)
		self.value_linear = nn.Linear(hidden_size, hidden_size, bias=False)
		self.out_linear = nn.Linear(hidden_size, hidden_size, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.rotary_embedding = RotaryEmbedding(self.num_hid_dim)

	def forward(self, query, key, value, mask=None, layer_past=None, use_cache=False):
		'''
		Parameters:
			query: bsz, seq_len, hidden_size
			key: bsz, seq_len, hidden_size
			value: bsz, seq_len, hidden_size
			mask: bsz, seq_len
		Returns:
			output: bsz, seq_len, hidden_size
		'''
		bsz, seq_len, hidden_size = query.shape

		# linear
		mixed_query = self.query_linear(query)
		mixed_key = self.key_linear(key)
		mixed_value = self.value_linear(value)

		query_t = mixed_query.view(bsz, -1, self.num_heads, self.num_hid_dim)
		key_t = mixed_key.view(bsz, -1, self.num_heads, self.num_hid_dim)
		value_t = mixed_value.view(bsz, -1, self.num_heads, self.num_hid_dim)
		query_t = self.rotary_embedding(query_t, position_ids)
		key_t = self.rotary_embedding(key_t, position_ids)

		if layer_past is not None:
			past_keys, past_values = layer_past
			key_t = torch.cat([past_keys, key_t], dim=1)
			value_t = torch.cat([past_values, value_t], dim=1)
		if use_cache:
			layer_past = (key_t, value_t)
		if is_flash_attn_available:
			attn_out = flash_attn_func(query_t, key_t, value_t, dropout_p=0, causal=True)
		else:
			# transpose for attn
			# bsz, seq_len, num_heads, num_hid_dim --> bsz, num_heads, seq_len, num_hid_dim
			query_t = _query.view(bsz, -1, self.num_heads, self.num_hid_dim).contiguous().permute(0, 2, 1, 3)
			# bsz, seq_len, num_heads, num_hid_dim --> bsz, num_heads, num_hid_dim, seq_len
			key_t = _key.view(bsz, -1, self.num_heads, self.num_hid_dim).contiguous().permute(0, 2, 3, 1)
			# bsz, seq_len, num_heads, num_hid_dim --> bsz, num_heads, seq_len, num_hid_dim
			value_t = _value.view(bsz, -1, self.num_heads, self.num_hid_dim).contiguous().permute(0, 2, 1, 3)

			# bsz, num_heads, seq_len, seq_len
			score = torch.matmul(query_t, key_t) / math.sqrt(self.num_hid_dim)
			# bsz, seq_len --> bsz, 1, seq_len, 1
			if mask is not None:
				if mask.dim() == 2:
					mask = mask.unsqueeze(1).unsqueeze(1)
				elif mask.dim() == 3:
					mask = mask.unsqueeze(1)
				mask = (1 - mask) * (-10000)
				mask = mask.to(score.dtype)
				# attention
				attn = F.softmax(score + mask, dim=-1)
			else:
				attn = F.softmax(score, dim=-1)
			attn = self.dropout(attn)

			attn_out = torch.matmul(attn, value_t)
			# bsz, seq_len, num_heads, num_hid_dim --> bsz, seq_len, hidden_size
		attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(bsz, -1, self.hidden_size)
		attn_out = self.out_linear(attn_out)
		return attn_out, layer_past

class FeedForward(nn.Module):
	"""docstring for FeedForward"""
	def __init__(self, hidden_size, mid_size, dropout):
		super(FeedForward, self).__init__()
		self.gate_proj = nn.Linear(hidden_size, mid_size, bias=False)
		self.down_proj = nn.Linear(mid_size, hidden_size, bias=False)
		self.up_proj = nn.Linear(hidden_size, mid_size, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.act = nn.SiLU()

	def forward(self, x):
		out = self.down_proj(self.up_proj(x) * self.act(self.gate_proj(x)))
		return out

class DecoderLayer(nn.Module):
	"""docstring for DecoderLayer"""
	def __init__(self, num_heads, num_hid_dim, hidden_size, dropout, mid_size):
		super(DecoderLayer, self).__init__()
		self.attn = SelfAttn(num_heads, num_hid_dim, hidden_size, dropout)
		self.fc = FeedForward(hidden_size, mid_size, dropout)
		self.input_norm = RMSNorm(hidden_size)
		self.post_attention_norm = RMSNorm(hidden_size)

	def forward(self, hidden_states, attention_mask=None, position_ids=None, multi=None, layer_past=None, use_cache=False):
		bsz, seq_len, _ = hidden_states.shape
		all_ones = hidden_states.new_ones(bsz, seq_len, seq_len)
		mask = torch.tril(all_ones)
		if attention_mask is None:
			attention_mask = mask
		else:
			attention_mask = attention_mask.unsqueeze(1) * mask

		residual = hidden_states
		hidden_states = self.input_norm(hidden_states)
		hidden_states, layer_past = self.attn(hidden_states, hidden_states, hidden_states, attention_mask, layer_past, use_cache)
		hidden_states = hidden_states + residual

		residual = hidden_states
		hidden_states = self.post_attention_norm(hidden_states)
		hidden_states = self.fc(hidden_states)
		if multi is not None:
			hidden_states = multi * hidden_states + residual
		else:
			hidden_states = hidden_states + residual
		return hidden_states, layer_past

class Embedding(nn.Module):
	"""docstring for Embedding"""
	def __init__(self, vocab_size, hidden_size, dropout, alpha=0.1):
		super(Embedding, self).__init__()
		self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
		# self.position_embeddings = nn.Embedding(1024, hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.alpha = alpha

	def forward(self, input_ids, position_ids=None):
		token_embed = self.token_embeddings(input_ids)
		token_embed = self.alpha * token_embed + (1 - self.alpha) * token_embed.detach()
		# if position_ids is None:
		# 	position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
		# pos_embed = self.position_embeddings(position_ids)
		# outputs = self.dropout(token_embed + pos_embed)
		return token_embed

class Selector(nn.Module):
	"""docstring for Selector"""
	def __init__(self, hidden_size):
		super(Selector, self).__init__()
		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, x):
		return torch.sigmoid(self.fc(x).squeeze(-1))

class GPT(nn.Module):
	def __init__(self, config, generation_config=None):
		super(GPT, self).__init__()
		self.config = config
		self.generation_config = generation_config
		self.embeddings = Embedding(config['vocab_size'], config['hidden_size'], config['dropout'])
		self.layers = nn.ModuleList([DecoderLayer(config['num_attention_heads'], config['num_hid_dim'], config['hidden_size'], config['dropout'], config['mid_size']) for _ in range(config['num_layers'])])
		self.selectors = nn.ModuleList([Selector(config['hidden_size']) for _ in range(config['num_layers'])])
		self.final_norm = RMSNorm(config['hidden_size'])
		self.classifier = nn.Linear(config['hidden_size'], config['vocab_size'])
		if config['tie_weight']:
			self.classifier.weight = self.embeddings.token_embeddings.weight
		self.gradient_checkpointing = False
		self.apply(self.init_weights)

	def get_mask(self, mask, hidden_states):
		mask = mask.nonzero().view(-1)
		mask = mask.unsqueeze(0).unsqueeze(-1).repeat(hidden_states.shape[0], 1, hidden_states.shape[-1])
		return mask.to(hidden_states.device)

	def get_keep_mask(self, scores, n):
		sorted_scores, indices = scores.sort(dim=-1, descending=True)
		return sorted_scores[:, :n], indices[:, :n]

	def forward(self, input_ids, attention_mask=None, labels=None):
		position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
		hidden_states = self.embeddings(input_ids)
		bsz, _, hidden_size = hidden_states.shape
		# for idx, layer in enumerate(self.layers):
		# 	mask = torch.rand(hidden_states.shape[1]) > self.config['skip_layer_ratio_list'][idx]
		# 	keep_mask = self.get_mask(mask, hidden_states)
		# 	keep_hidden_states = torch.gather(hidden_states, 1, keep_mask)
		# 	if self.gradient_checkpointing and self.training:
		# 		keep_hidden_states, _ = torch.utils.checkpoint.checkpoint(layer, keep_hidden_states, attention_mask)
		# 		hidden_states = hidden_states.scatter(1, keep_mask, keep_hidden_states)
		# 	else:
		# 		keep_hidden_states, _ = layer(keep_hidden_states, attention_mask)
		# 		hidden_states = hidden_states.scatter(1, keep_mask, keep_hidden_states)

		for idx, layer in enumerate(self.layers):
			if idx % 2 == 0:
				scores = self.selectors[idx](hidden_states)
				keep_prob, keep_mask = self.get_keep_mask(scores, int(self.config['avg_keep_layer_ratio'] * self.config['max_length']))
				keep_mask = torch.sort(keep_mask, dim=-1)[0]
				keep_mask = keep_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
				keep_hidden_states = torch.gather(hidden_states, 1, keep_mask)
				keep_position_ids = torch.gather(position_ids, 1, keep_mask[:, :, 0])
				# keep_hidden_states = keep_hidden_states * keep_prob.unsqueeze(-1)
				if self.gradient_checkpointing and self.training:
					keep_hidden_states, _ = torch.utils.checkpoint.checkpoint(layer, keep_hidden_states, attention_mask, keep_position_ids, keep_prob.unsqueeze(-1))
					hidden_states = hidden_states.scatter(1, keep_mask, keep_hidden_states)
				else:
					keep_hidden_states, _ = layer(keep_hidden_states, attention_mask, keep_position_ids, keep_prob.unsqueeze(-1))
					hidden_states = hidden_states.scatter(1, keep_mask, keep_hidden_states)
			else:
				if self.gradient_checkpointing and self.training:
					hidden_states, _ = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask)
				else:
					hidden_states, _ = layer(hidden_states, attention_mask)

		hidden_states = self.final_norm(hidden_states)
		logits = self.classifier(hidden_states)
		loss_fun = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])
		shift_logits = logits[:, :-1].contiguous()
		if labels is not None:
			shift_labels = labels[:, 1:].contiguous()
		else:	
			shift_labels = input_ids.clone()[:, 1:].contiguous()
		loss = loss_fun(shift_logits.view(-1, self.config['vocab_size']), shift_labels.view(-1))
		return loss

	def prepare_inputs_for_generation(self, input_ids, position_ids, past_key_values):
		model_inputs = {}
		if past_key_values[0] is not None:
			model_inputs['input_ids'] = input_ids[:, -1:]
			model_inputs['position_ids'] = position_ids[:, -1:]
		else:
			model_inputs['input_ids'] = input_ids
			model_inputs['position_ids'] = position_ids
		return model_inputs

	def update_model_inputs(self, position_ids, attention_mask):
		max_pos = position_ids[:, -1:]
		position_ids = torch.cat([position_ids, max_pos+1], dim=1)
		attention_mask = torch.cat([attention_mask, attention_mask.new_ones(position_ids.shape[0], 1)], dim=1)
		return position_ids, attention_mask

	@torch.inference_mode()
	def generate(self, input_ids, position_ids=None, attention_mask=None):
		# generation mode only support use_cache now
		if self.generation_config['use_cache']:
			past_key_values = [None for _ in range(self.config['num_layers'])]
		if position_ids is None:
			position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
		if attention_mask is None:
			attention_mask = torch.ones(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
		not_finished = input_ids.new_ones(input_ids.shape[0])
		while True:
			model_inputs = self.prepare_inputs_for_generation(input_ids, position_ids, past_key_values)
			hidden_states = self.embeddings(model_inputs['input_ids'], model_inputs['position_ids'])
			for idx, layer in enumerate(self.layers):
				hidden_states, layer_past = layer(hidden_states, attention_mask=attention_mask, layer_past=past_key_values[idx], use_cache=self.generation_config['use_cache'])
				if self.generation_config['use_cache']:
					past_key_values[idx] = layer_past
			hidden_states = self.final_norm(hidden_states)
			logits = self.classifier(hidden_states)
			next_tokens = logits.argmax(dim=-1)[:, -1]
			input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
			position_ids, attention_mask = self.update_model_inputs(position_ids, attention_mask)

			not_finished = not_finished * next_tokens.ne(self.generation_config['eos_token_id'])

			if not_finished.sum() == 0 or input_ids.shape[1] == self.config['max_length']:
				break
		return input_ids

	@torch.inference_mode()
	def chat(self, tokenizer, query, device, only_query):
		input_ids = tokenizer.build_chat_input(query, only_query)
		input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
		outputs = self.generate(input_ids)
		response = tokenizer.decode(outputs[0].tolist())
		return response

	def enable_gradient_checkpointing(self):
		self.gradient_checkpointing = True

	def init_weights(self, module):
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(0, self.config["weight_std_range"])
			# module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(0, self.config["weight_std_range"])
		elif isinstance(module, nn.LayerNorm):
			module.weight.data.fill_(1)
			module.bias.data.zero_()

	@torch.inference_mode()
	def stream_generate(self, input_ids, position_ids=None, attention_mask=None):
		# generation mode only support use_cache now
		if self.generation_config['use_cache']:
			past_key_values = [None for _ in range(self.config['num_layers'])]
		if position_ids is None:
			position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
		if attention_mask is None:
			attention_mask = torch.ones(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
		not_finished = input_ids.new_ones(input_ids.shape[0])
		while True:
			model_inputs = self.prepare_inputs_for_generation(input_ids, position_ids, past_key_values)
			hidden_states = self.embeddings(model_inputs['input_ids'], model_inputs['position_ids'])
			for idx, layer in enumerate(self.layers):
				hidden_states, layer_past = layer(hidden_states, attention_mask=attention_mask, layer_past=past_key_values[idx], use_cache=self.generation_config['use_cache'])
				if self.generation_config['use_cache']:
					past_key_values[idx] = layer_past
			hidden_states = self.final_norm(hidden_states)
			logits = self.classifier(hidden_states)
			if self.generation_config['do_sampling']:
				probs = F.softmax(logits_processor(input_ids, logits[:, -1], self.generation_config['temperature'], self.generation_config['top_p'], self.generation_config['repetition_penalty']), dim=-1)
				next_tokens = torch.multinomial(probs, 1).squeeze(-1)
			else:
				next_tokens = logits.argmax(dim=-1)[:, -1]
			input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
			position_ids, attention_mask = self.update_model_inputs(position_ids, attention_mask)

			not_finished = not_finished * next_tokens.ne(self.generation_config['eos_token_id'])

			yield input_ids
			if not_finished.sum() == 0 or input_ids.shape[1] == self.config['max_length']:
				break

	@torch.inference_mode()
	def stream_chat(self, tokenizer, query, device, only_query):
		_, input_ids = build_chat_input(tokenizer, query, only_query)
		input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
		for outputs in self.stream_generate(input_ids):
			response = tokenizer.decode(outputs[0].tolist())
			yield response