import torch
import json
# import joblib
import torch.nn.functional as F

def read_jsonl(filename, num_samples=None):
	data = []
	with open(filename, 'r', encoding='utf8') as f:
		for line in f.readlines():
			data.append(json.loads(line.rstrip()))
			if num_samples is not None and len(data) > num_samples:
				break
	return data

def read_json(filename):
	with open(filename, 'r', encoding='utf8') as f:
		data = json.load(f)
	return data

def to_json(data, filename):
	with open(filename, 'w', encoding='utf8') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

def fast_dump(data, filename):
	with open(filename, 'wb') as f:
		joblib.dump(data, f)

# def fast_load(filename):
# 	with open(filename, 'rb') as f:
# 		data = joblib.load(f)
# 	return data

# pretrain_ds
def collate_for_lm(batch):
	return torch.LongTensor([x['input_ids'] for x in batch]), 

def collate_for_sft(batch):
	return torch.LongTensor([x['input_ids'] for x in batch]), torch.LongTensor([x['attention_mask'] for x in batch]), torch.LongTensor([x['labels'] for x in batch])

def collate_for_dpo(batch):
	return torch.LongTensor([x['chosen_ids'] for x in batch]), torch.LongTensor([x['chosen_attention_mask'] for x in batch]), torch.LongTensor([x['chosen_labels'] for x in batch]), torch.LongTensor([x['rejected_ids'] for x in batch]), torch.LongTensor([x['rejected_attention_mask'] for x in batch]), torch.LongTensor([x['rejected_labels'] for x in batch])

def safe_load(model, model_path):
	sd = torch.load(model_path, map_location='cpu')
	model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
	return model

def print_rank_0(dataset, tokenizer, task):
	item = dataset[0]
	if task == 'pretrain':
		pass
	elif task == 'sft':
		input_ids = item['input_ids']
		labels = item['labels']
		decoded = tokenizer._convert_id_to_token(input_ids)
		for i, l, d in zip(input_ids, labels, decoded):
			if d == '<unk>':
				continue
			print(i, l, d if d != '\n' else '_n')
	elif task == 'dpo' or task == 'orpo':
		input_ids = item['chosen_ids']
		labels = item['chosen_labels']
		decoded = tokenizer._convert_id_to_token(input_ids)
		for i, l, d in zip(input_ids, labels, decoded):
			if d == '<unk>':
				continue
			print(i, l, d if d != '\n' else '_n')

		print()
		input_ids = item['rejected_ids']
		labels = item['rejected_labels']
		decoded = tokenizer._convert_id_to_token(input_ids)
		for i, l, d in zip(input_ids, labels, decoded):
			if d == '<unk>':
				continue
			print(i, l, d if d != '\n' else '_n')

def prepare_model_inputs(batch, task):
	if task == 'dpo' or task == 'orpo':
		inputs = {
			'input_ids': torch.cat([batch[0], batch[3]], dim=0),
			'attention_mask': torch.cat([batch[1], batch[4]], dim=0)
		}
	else:
		inputs = {'input_ids': batch[0]}
		if task == 'sft':
			inputs['attention_mask'] = batch[1]
			inputs['labels'] = batch[2]
	return inputs

def get_train_ds_config(offload,
						stage,
						global_batch_size,
						micro_batch_size,
						grad_acc,
						bf16=False,
						enable_hybrid_engine=False,
						inference_tp_size=1,
						release_inference_cache=False,
						pin_parameters=True,
						tp_gather_partition_size=8,
						max_out_tokens=512):
	device = "cpu" if offload else "none"
	zero_opt_dict = {
		"stage": stage,
		"offload_param": {
			"device": device
		},
		"offload_optimizer": {
			"device": device
		},
		"stage3_param_persistence_threshold": 1e4,
		"stage3_max_live_parameters": 3e7,
		"stage3_prefetch_bucket_size": 3e7,
		"memory_efficient_linear": False
	}
	return {
		"train_batch_size": global_batch_size,
		"train_micro_batch_size_per_gpu": micro_batch_size,
		"steps_per_print": 500,
		"zero_optimization": zero_opt_dict,
		"fp16": {
			"enabled": True if not bf16 else False,
			"auto_cast": False,
			"loss_scale": 0,
			"initial_scale_power": 16,
			"loss_scale_window": 1000,
			"hysteresis": 2,
			"consecutive_hysteresis": False,
			"min_loss_scale": 1
		},
		"bf16":{
			"enabled": True if bf16 else False
		},
		"gradient_clipping": 1.0,
		"prescale_gradients": False,
		"wall_clock_breakdown": False,
		"gradient_acculation_steps": grad_acc,
	}

def get_batch_logps(logits, labels, average=False, label_pad_token_id=-100, eps=1e-5):
	labels[labels == label_pad_token_id] = 0
	loss_mask = labels != label_pad_token_id
	per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
	if average:
		return (per_token_logps * loss_mask).sum(-1) / (loss_mask.sum(-1) + eps)
	else:
		return (per_token_logps * loss_mask).sum(-1)

def get_dpo_loss(policy_logits, reference_logits, labels, beta=0.1):
	bsz = policy_logits.shape[0] // 2
	policy_logps = get_batch_logps(policy_logits, labels)
	reference_logps = get_batch_logps(reference_logits, labels)

	policy_chosen_logps, policy_rejected_logps = policy_lops.split(bsz, dim=0)
	reference_chosen_logps, reference_rejected_logps = reference_logps.split(bsz, dim=0)

	logits = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
	loss = - F.logsigmoid(beta * logits)
	return loss

def get_orpo_loss(logits, labels, beta=0.1):
	bsz = logits.shape[0] // 2
	logps = get_batch_logps(logits, labels, average=True)
	chosen_logps, rejected_logps = logps.split(bsz, dim=0)
	log_odds = chosen_logps - rejected_logps - (torch.log1p(-torch.exp(chosen_logps) - torch.log1p(-torch.exp(rejected_logps))))
	odds_ratio_loss = - F.logsigmoid(log_odds)
	sft_loss = -chosen_logps
	loss = (sft_loss + beta * odds_ratio_loss).mean()
	metrics = {}
	metrics['chosen_rewards'] = beta * chosen_logps.detach().cpu().mean()
	metrics['rejected_rewards'] = beta * rejected_logps.detach().cpu().mean()
	metrics['acc'] = (chosen_logps > rejected_logps).float().detach().cpu().mean()
	metrics['sft_loss'] = sft_loss.detach().cpu().mean()
	metrics['orpo_loss'] = beta * odds_ratio_loss.detach().cpu().mean()
	return loss, metrics

# pretrain_ds_moe
def load_dense_weight_for_moe(model, model_path, config, add_noise=False):
	sd = torch.load(model_path, map_location='cpu')
	layer_numbers = ['layers.%s.' % x for x in range(config['num_layers']) if x % 2 == 1]
	to_replace_dict = {}
	for k, v in sd.items():
		for layer_number in layer_numbers:
			if layer_number in k and 'fc' in k:
				to_replace_dict[k] = [k.replace('fc', 'experts.%s' % n) for n in range(config['num_of_experts'])]
	
	new_sd = {}
	for k, v in sd.items():
		if k in to_replace_dict:
			for e in to_replace_dict[k]:
				new_sd[e] = v
		else:
			new_sd[k] = v
	
	model.load_state_dict(new_sd, strict=False)
	return model

# cli_demo
def build_chat_input(tokenizer, query, only_query=False, history=[]):
	if only_query:
		# for pretrained-llm continue writing
		return [], tokenizer(query, add_special_tokens=False)['input_ids']
	# same as qwen
	system_prompt = '<|startoftext|>system\n你是一个由George Play AI创建的智能体，请按照指令回答用户的问题。<|endoftext|>'
	system_prompt_ids = tokenizer(system_prompt, add_special_tokens=False)['input_ids']
	pairs = []
	for i in range(len(history)):
		source_ids = tokenizer('\n<|startoftext|>user\n' + history[i][0] + '<|endoftext|>\n<|startoftext|>assistant\n', add_special_tokens=False)['input_ids']
		if i == 0:
			source_ids = system_prompt_ids + source_ids
		tgt_ids = tokenizer(history[i][1] + '<|endoftext|>', add_special_tokens=False)['input_ids']
		pairs.append((source_ids, tgt_ids))
	query_ids = tokenizer('\n<|startoftext|>user\n' + query + '<|endoftext|>\n<|startoftext|>assistant\n', add_special_tokens=False)['input_ids']
	if len(history) == 0:
		input_ids = system_prompt_ids + query_ids
	else:
		input_ids = query_ids
	return pairs, input_ids

def logits_processor(input_ids, logits, temperature, top_p, repetition_penalty):
	assert temperature > 0
	assert 0 < top_p < 1
	# repetition_penalty 
	score = torch.gather(logits, 1, input_ids)
	score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
	logits.scatter_(1, input_ids, score)
	# top-p
	scores = logits / temperature
	sorted_scores, sorted_indices = scores.sort(dim=-1, descending=False)
	cum_probs = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
	sorted_indices_to_remove = cum_probs <= (1 - top_p)
	indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
	scores = scores.masked_fill(indices_to_remove, -float("Inf"))
	return scores