import re

def clean_text(text):
	text = re.sub('\n\s+', '\n', text)
	text = re.sub('\s+\n', '\n', text)
	text = re.sub('\n\s+\n', '\n\n', text)
	text = re.sub('^\s+', '', text)
	text = re.sub('\s+$', '', text)
	return text

def concat_all(title, summary, sections):
	title = clean_text(title)
	if summary is not None:
		if summary == title or len(summary) < 16:
			text = ''
		else:
			text = title + ':' + summary + '<|endoftext|>'
	else:
		text = ''
	for idx, section in enumerate(sections):
		if section['title'] in ['全集在线观看', '截图']:
			continue
		content = section['content']
		content = clean_text(content)
		content = content.replace('\xa0\xa0' + title, '')
		if section['title'].endswith('\n') or content.startswith('\n'):
			text += title + '-' + section['title'] + content
		else:
			text += title + '-' + section['title'] + '\n' + content
		text += '<|endoftext|>'
	text = text.replace('。。<|endoftext|>', '。<|endoftext|>').replace('\n\n<|endoftext|>', '<|endoftext|>').replace('\n<|endoftext|>', '<|endoftext|>').replace('<|endoftext|><|endoftext|>', '<|endoftext|>')
	text = text.replace('\xa0', ' ').replace('\u3000', ' ')
	text = text.replace('\x05', '').replace('�', '').replace('\x10', '').replace('\x08', '').replace('\x18', '')
	# table
	if text.endswith('<|endoftext|>'):
		text = text[:-13]
	# todo no dup
	return text

def get_length(x):
	length = 0
	if x['summary'] is not None:
		length += len(x['summary'])
	for section in x['sections']:
		length += len(section['title'] + section['content'])
	return length

def preprocess_pretrain_baidubaike_dataset(examples, tokenizer, max_length):
	all_input_ids = []
	examples = [concat_all(title, summary, sections) + tokenizer.eos_token for title, summary, sections in zip(examples['title'], examples['summary'], examples['sections'])]
	tokenized_examples = tokenizer(examples, add_special_tokens=False)
	flat_input_ids = []
	for input_ids in tokenized_examples['input_ids']:
		flat_input_ids += input_ids
	num_chunks = len(flat_input_ids) // max_length

	for i in range(num_chunks):
		all_input_ids.append(flat_input_ids[i*max_length:i*max_length+max_length])
	return {'input_ids': all_input_ids}

def cut_tail(text):
	index = len(text)
	if 'References\n\n' in text:
		index = text.index('References\n\n')
	elif 'References\n' in text:
		index = min(index, text.index('References\n'))
	if 'External links\n\n' in text:
		index = min(index, text.index('External links\n\n'))
	elif 'External links\n' in text:
		index = min(index, text.index('External links\n'))
	if 'Charts\n' in text:
		index = min(index, text.index('Charts\n'))
	elif 'Charts\n\n' in text:
		index = min(index, text.index('Charts\n\n'))
	return text[:index]

def preprocess_pretrain_wikipedia_dataset(examples, tokenizer, max_length):
	all_input_ids = []
	examples = [title + '\n' + cut_tail(text) + tokenizer.eos_token for title, text in zip(examples['title'], examples['text'])]
	tokenized_examples = tokenizer(examples, add_special_tokens=False)
	flat_input_ids = []
	for input_ids in tokenized_examples['input_ids']:
		flat_input_ids += input_ids
	num_chunks = len(flat_input_ids) // max_length

	for i in range(num_chunks):
		all_input_ids.append(flat_input_ids[i*max_length:i*max_length+max_length])
	return {'input_ids': all_input_ids}

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
	for inst, chosen_ids, rejected_ids in zip(instructions, tokenized_chosen_ids['input_ids'], tokenized_rejected_ids['input_ids']):
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
		if len(prompt_rejected_ids) < max_length:
			rejected_padding_length = max_length - len(prompt_rejected_ids)
		else:
			print('the length of example is %s which exceeds max_length' % len(prompt_rejected_ids))
			continue
		all_prompt_chosen_ids.append(prompt_chosen_ids + [pad_token_id] * chosen_padding_length)
		all_chosen_attention_masks.append(chosen_attention_mask + [0] * chosen_padding_length)
		all_chosen_labels.append(chosen_labels + [-100]*chosen_padding_length)
		all_prompt_rejected_ids.append(prompt_rejected_ids + [pad_token_id] * rejected_padding_length)
		all_rejected_attention_masks.append(rejected_attention_mask + [0] * rejected_padding_length)
		all_rejected_labels.append(rejected_labels + [-100]*rejected_padding_length)
	return {'chosen_ids': all_prompt_chosen_ids, 
		'chosen_attention_mask': all_chosen_attention_masks, 
		'chosen_labels': all_chosen_labels,
		'rejected_ids': all_prompt_rejected_ids, 
		'rejected_attention_mask': all_rejected_attention_masks, 
		'rejected_labels': all_rejected_labels}