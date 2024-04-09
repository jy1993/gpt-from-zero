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

# pre_tokenization
def preprocess_pretrain_wanjuan_dataset(examples, tokenizer, max_length):
	all_input_ids = []
	examples = [text + tokenizer.eos_token for text in examples['content']]
	tokenized_examples = tokenizer(examples, add_special_tokens=False)
	flat_input_ids = []
	for input_ids in tokenized_examples['input_ids']:
		flat_input_ids += input_ids
	num_chunks = len(flat_input_ids) // max_length

	for i in range(num_chunks):
		all_input_ids.append(flat_input_ids[i*max_length:i*max_length+max_length])
	return {'input_ids': all_input_ids}

def preprocess_pretrain_pile_dataset(examples, tokenizer, max_length):
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

def preprocess_dpo_dataset_alpaca(examples, tokenizer, max_length, eos_token_id, pad_token_id):
	all_input_ids = []
	instructions = examples['instruction']
	inputs = examples['input']
	instructions = [inst + '\n' + input_ if input_ != '' else inst for inst, input_ in zip(instructions, inputs)]
	outputs = examples['output']
	# tokenized_instructions = tokenizer.tokenizer(instructions, add_special_tokens=False)
	for output in outputs:
		assert len(output) == 2
	tokenized_chosen_ids = tokenizer([output[0] for output in outputs], add_special_tokens=False)
	tokenized_rejected_ids = tokenizer([output[1] for output in outputs], add_special_tokens=False)
	all_prompt_chosen_ids, all_prompt_rejected_ids = [], []
	all_chosen_labels, all_rejected_labels = [], []
	all_chosen_attention_masks, all_rejected_attention_masks = [], []
	# for ti, to in zip(tokenized_instructions['input_ids'], tokenized_outputs['input_ids']):
	for inst, chosen_ids, rejected_ids in zip(instructions, tokenized_chosen_ids, tokenized_rejected_ids):
		pairs, inst_ids = build_chat_input(tokenizer, inst, history=[])
		input_ids, labels = [], []
		prompt_chosen_ids = inst_ids + chosen_ids + [eos_token_id]
		chosen_labels = [-100] * len(inst_ids) + chosen_ids + [eos_token_id]
		chosen_attention_mask = [1] * len(prompt_chosen_ids)

		prompt_rejected_ids = inst_ids + rejected_ids + [eos_token_id]
		rejected_labels = [-100] * len(inst_ids) + rejected_ids + [eos_token_id]
		rejected_attention_mask = [1] * len(prompt_rejected_ids)
		if len(prompt_chosen_ids) < max_length:
			chosen_padding_length = max_length - len(prompt_chosen_ids)
		else:
			print('the length of example is %s which exceeds max_length' % len(prompt_chosen_ids))
			continue
		all_prompt_chosen_ids.append(prompt_chosen_ids + [pad_token_id] * chosen_padding_length)
		all_chosen_attention_masks.append(chosen_attention_mask + [0] * chosen_padding_length)
		all_chosen_labels.append(chosen_labels + [-100]*chosen_padding_length)

		if len(prompt_rejected_ids) < max_length:
			rejected_padding_length = max_length - len(prompt_rejected_ids)
		else:
			print('the length of example is %s which exceeds max_length' % len(prompt_rejected_ids))
			continue
		all_prompt_rejected_ids.append(prompt_rejected_ids + [pad_token_id] * prompt_rejected_ids)
		all_rejected_attention_masks.append(rejected_attention_mask + [0] * prompt_rejected_ids)
		all_rejected_labels.append(rejected_labels + [-100]*chosen_padding_length)
	return {'chosen_ids': all_prompt_chosen_ids, 
		'chosen_attention_mask': all_chosen_attention_masks, 
		'chosen_labels': all_chosen_labels,
		'rejected_ids': all_prompt_rejected_ids, 
		'rejected_attention_mask': all_rejected_attention_masks, 
		'rejected_labels': all_rejected_labels}

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

def print_rank_0(dataset, tokenizer):
	item = dataset[0]
	input_ids = item['input_ids']
	labels = item['labels']
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

def get_batch_logps(logits, labels, average=False, label_pad_token_id=-100):
	loss_mask = labels != label_pad_token_id
	per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
	if average:
		return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
	else:
		return (per_token_logps * loss_mask).sum(-1)

def get_dpo_loss(policy_logits, reference_logits, labels, beta=0.1):
	bsz = policy_logits.shape[0] // 2
	policy_logps = get_batch_logps(policy_logits, labels)
	reference_logps = get_batch_logps(reference_logits, labels)

	policy_chosen_logps, policy_rejected_logps = policy_lops.split(bsz, dim=0)
	reference_chosen_logps, reference_rejected_logps = reference_logps.split(bsz, dim=0)

	logits = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
	loss = - F.log_sigmoid(beta * logits)
	return loss

def get_orpo_loss(logits, labels, beta=0.1):
	bsz = logits.shape[0] // 2
	logps = get_batch_logps(logits, labels, average=True)
	chosen_logps, rejected_logps = lops.split(bsz, dim=0)
	log_odds = chosen_logps - rejected_logps - (torch.log1p(-torch.exp(chosen_logps) - torch.log1p(-torch.exp(rejected_logps))))
	odds_ratio_loss = - F.log_sigmoid(log_odds)
	sft_loss = -chosen_logps
	loss = (sft_loss + beta * odds_ratio_loss).mean()
	return loss

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