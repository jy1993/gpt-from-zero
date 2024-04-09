import re

def concat_all(title, summary, sections):
	if summary is not None:
		if summary == title or len(summary) < 16:
			text = ''
		else:
			text = title + ':' + summary + '\n\n'
	else:
		text = ''
	for idx, section in enumerate(sections):
		if section['title'] in ['全集在线观看', '截图']:
			continue
		content = section['content']
		content = re.sub('\n\s+', '\n', content)
		content = re.sub('\s+\n', '\n', content)
		content = re.sub('\n\s+\n', '\n\n', content)
		content = re.sub('^\s+', '', content)
		content = re.sub('\s+$', '', content)
		content = content.replace('\xa0\xa0' + title, '')
		if section['title'].endswith('\n') or content.startswith('\n'):
			text += title + '-' + section['title'] + content
		else:
			text += title + '-' + section['title'] + '\n' + content
		if not content.endswith('\n\n'):
			if content.endswith('\n'):
				text += '\n'
			else:
				text += '\n\n'
	if text.endswith('\n\n'):
		text = text[:-2]
	text = text.replace('。。', '。').replace('\xa0', ' ').replace('\u3000', ' ')
	text = text.replace('\x05', '').replace('�', '').replace('\x10', '').replace('\x08', '').replace('\x18', '')
	# table
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