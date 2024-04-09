from transformers import AutoTokenizer
from modeling_new import GPT
from utils import read_json, safe_load
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='configs/2B.json')
parser.add_argument('--generation_config_path', type=str, default='configs/generation_config.json')
parser.add_argument('--tokenizer_path', type=str, default='yi-tokenizer')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--cut_off', type=int, default=300)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
config = read_json(args.config_path)
generation_config = read_json(args.generation_config_path)
device = 'cuda:0'
model = GPT(config, generation_config)
model = safe_load(model, args.model_path)
model = model.half().to(device)

while True:
	query = input('please enter a query, enter quit to stop the program:\n')
	if query == 'quit':
		break
	start = 0
	for response in model.stream_chat(tokenizer, query, device=device, only_query=True if 'pretrain_steps' in args.model_path else False):
		print(''.join(response[start:]), end='', flush=True)
		start = len(response)
		if start > args.cut_off:
			break
	print('')