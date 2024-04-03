import torch
import json
# import joblib

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

class InputFeature(object):
	"""docstring for InputFeature"""
	def __init__(self, input_ids, attention_mask=None):
		self.input_ids = input_ids 
		# self.attention_mask = attention_mask

def convert_to_features(examples, tokenizer, max_length):
	features = []
	examples = [ex['text'] + ' [SEP]' for ex in examples]
	tokenized_examples = tokenizer(examples, add_special_tokens=False)
	flat_input_ids, flat_attention_mask = [], []
	for input_ids in tokenized_examples['input_ids']:
		flat_input_ids += input_ids
	for attention_mask in tokenized_examples['attention_mask']:
		flat_attention_mask += attention_mask
	assert len(flat_input_ids) == len(flat_attention_mask)
	num_chunks = len(flat_input_ids) // max_length

	for i in range(num_chunks):
		# if len(features) < 3:
		# 	print('*' * 10)
		# 	print(input_ids)
		# 	print(attention_mask)
		features.append(InputFeature(flat_input_ids[i*max_length:i*max_length+max_length]))
	return features

def get_dataset(features):
	all_input_ids = torch.LongTensor([f.input_ids for f in features])
	# all_attention_mask = torch.LongTensor([f.attention_mask for f in features])
	return torch.utils.data.TensorDataset(all_input_ids)

def preprocess_pretrain_dataset(examples, tokenizer, max_length):
	all_input_ids = []
	examples = [text + tokenizer.eos_token for text in examples['text']]
	tokenized_examples = tokenizer(examples, add_special_tokens=False)
	flat_input_ids = []
	for input_ids in tokenized_examples['input_ids']:
		flat_input_ids += input_ids
	num_chunks = len(flat_input_ids) // max_length

	for i in range(num_chunks):
		all_input_ids.append(flat_input_ids[i*max_length:i*max_length+max_length])
	return {'input_ids': all_input_ids}

def preprocess_pretrain_code_dataset(examples, tokenizer, max_length):
	all_input_ids = []
	examples = [text + tokenizer.eos_token for text in examples['code']]
	tokenized_examples = tokenizer(examples, add_special_tokens=False)
	flat_input_ids = []
	for input_ids in tokenized_examples['input_ids']:
		flat_input_ids += input_ids
	num_chunks = len(flat_input_ids) // max_length

	for i in range(num_chunks):
		all_input_ids.append(flat_input_ids[i*max_length:i*max_length+max_length])
	return {'input_ids': all_input_ids}

def preprocess_pretrain_wikipedia_dataset(examples, tokenizer, max_length):
	all_input_ids = []
	examples = [title + '\n' + text + tokenizer.eos_token for title, text in zip(examples['title'], examples['text'])]
	tokenized_examples = tokenizer(examples, add_special_tokens=False)
	flat_input_ids = []
	for input_ids in tokenized_examples['input_ids']:
		flat_input_ids += input_ids
	num_chunks = len(flat_input_ids) // max_length

	for i in range(num_chunks):
		all_input_ids.append(flat_input_ids[i*max_length:i*max_length+max_length])
	return {'input_ids': all_input_ids}

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

def preprocess_sft_dataset_alpaca(examples, tokenizer, max_length, eos_token_id, pad_token_id):
	all_input_ids = []
	instructions = examples['instruction']
	inputs = examples['input']
	historys = examples['history']
	instructions = [inst + '\n' + input_ if input_ != '' else inst for inst, input_ in zip(instructions, inputs)]
	outputs = examples['output']
	# tokenized_instructions = tokenizer.tokenizer(instructions, add_special_tokens=False)
	tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
	all_input_ids, all_labels, all_attention_masks = [], [], []
	# for ti, to in zip(tokenized_instructions['input_ids'], tokenized_outputs['input_ids']):
	for inst, history, res_ids in zip(instructions, historys, tokenized_outputs['input_ids']):
		pairs, inst_ids = build_chat_input(tokenizer, inst, history=history)
		input_ids, labels = [], []
		for source_ids, tgt_ids in pairs:
			input_ids += source_ids + tgt_ids
			labels += [-100] * len(source_ids) + tgt_ids
		input_ids += inst_ids + res_ids + [eos_token_id]
		labels += [-100] * len(inst_ids) + res_ids + [eos_token_id]
		attention_mask = [1] * len(input_ids)
		if len(input_ids) < max_length:
			padding_length = max_length - len(input_ids)
		else:
			print('the length of example is %s which exceeds max_length' % len(input_ids))
			continue
		all_input_ids.append(input_ids + [pad_token_id] * padding_length)
		all_attention_masks.append(attention_mask + [0] * padding_length)
		all_labels.append(labels + [-100]*padding_length)
	return {'input_ids': all_input_ids, 'attention_mask': all_attention_masks, 'labels': all_labels}

def collate_for_lm(batch):
	return torch.LongTensor([x['input_ids'] for x in batch]), 

def collate_for_sft(batch):
	return torch.LongTensor([x['input_ids'] for x in batch]), torch.LongTensor([x['attention_mask'] for x in batch]), torch.LongTensor([x['labels'] for x in batch])

def safe_load(model, model_path):
	sd = torch.load(model_path, map_location='cpu')
	model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
	return model

def print_rank_0(dataset, tokenizer):
	item = dataset[0]
	input_ids = item['input_ids']
	labels = item['labels']
	decoded = tokenizer._convert_id_to_token(input_ids)
	for i, l, d in zip(input_ids, labels, decoded):
		if d == '<unk>':
			continue
		print(i, l, d if d != '\n' else '_n')

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
