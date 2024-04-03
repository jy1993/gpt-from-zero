from transformers import AutoTokenizer
from modeling_new import GPT
from utils import read_json, safe_load
import torch

tokenizer = AutoTokenizer.from_pretrained('yi-tokenizer', trust_remote_code=True)
config = read_json('configs/0.9B.json')
generation_config = read_json('configs/generation_config.json')
config['max_length'] = 200
device = 'cuda:0'
model = GPT(config, generation_config)
model = safe_load(model, 'gpt_0.9B_sft/sft_steps_3600')
model = model.half().to(device)

while True:
	query = input('please enter a query, enter quit to stop the program:\n')
	if query == 'quit':
		break
	start = 0
	for response in model.stream_chat(tokenizer, query, device=device, only_query=False):
		print(''.join(response[start:]), end='', flush=True)
		start = len(response)
	print('')