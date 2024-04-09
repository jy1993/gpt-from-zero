import copy
import sys
import os
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import get_linear_schedule_with_warmup, AdamW, AutoTokenizer
from modeling_new import GPT
from tqdm import tqdm
from utils import *
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from datasets import interleave_datasets, load_dataset
from functools import partial

parser = ArgumentParser()
parser.add_argument('--task', type=str, default='pretrain')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--per_device_train_batch_size', type=int, default=16)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--zh_cache_path', type=str, default='cached/baidubaike')
parser.add_argument('--en_cache_path', type=str, default='cached/wikipedia-en')
parser.add_argument('--code_cache_path', type=str, default='cached/code')
parser.add_argument('--wanjuan_cache_path', type=str, default='cached/wanjuan')
parser.add_argument('--pile_cache_path', type=str, default='cached/the-pile')
parser.add_argument('--alpaca_cache_path', type=str, default='cached/sft-alpaca')
parser.add_argument('--tokenizer_path', type=str, default='yi-tokenizer')
parser.add_argument('--config_path', type=str, default=None)
parser.add_argument('--save_steps', type=int, default=10000)
parser.add_argument('--eval_steps', type=int, default=50000)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--clip_grad_norm', type=float, default=1.0)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--offload', action='store_true')
parser.add_argument('--zero_stage', type=int, default=0)
parser.add_argument('--gradient_checkpointing', action='store_true')
parser.add_argument('--local_rank', type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

def train(model, train_loader, valid_loader, optimizer, scheduler, scaler, writer): # training locally
	global_step = 0
	for _ in trange(args.epochs):
		for i, batch in enumerate(tqdm(train_loader)):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			inputs = prepare_model_inputs(batch, args.task)
			if args.task == 'dpo' or args.task == 'orpo':
				_, logits = model(**inputs)
				labels = torch.cat([batch[2], batch[5]], dim=0)
				if args.task == 'orpo':
					loss = get_orpo_loss(logits, labels)
				else:
					raise NotImplementedError("DPO not implemented")
			else:
				loss, _ = model(**inputs)
			if args.n_gpus > 1:
				loss = loss.mean()
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps
			if args.global_rank <= 0:
				writer.add_scalar('Train/total_loss', loss.item(), global_step)
			model.backward(loss)
			model.step()
			global_step += 1

			if global_step % args.eval_steps == 0:
				ppl = valid(model, valid_loader)
				if args.global_rank <= 0:
					writer.add_scalar('Val/ppl', ppl, global_step)

			if global_step % args.save_steps == 0:
				sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
				torch.save(sd, args.output_dir + '/%s_steps_%s' % (args.task, global_step))

def valid(model, data_loader):
	val_loss, batch_num = 0, 0
	with torch.no_grad():
		model.eval()
		for batch in tqdm(data_loader):
			batch = tuple(t.to(args.device) for t in batch)
			inputs = prepare_model_inputs(batch, args.task)
			if args.task == 'dpo' or args.task == 'orpo':
				_, logits = model(**inputs)
				labels = torch.cat([batch[2], batch[5]], dim=0)
				if args.task == 'orpo':
					loss = get_orpo_loss(logits, labels)
				else:
					raise NotImplementedError("DPO not implemented")
			else:
				loss, _ = model(**inputs)
			if args.n_gpus > 1:
				loss = loss.mean()
			val_loss += loss.item()
			batch_num += 1
	ppl = math.exp(val_loss / batch_num)
	return ppl

def main():
	if args.local_rank == -1:
		args.device = torch.device('cuda')
	else:
		torch.cuda.set_device(args.local_rank)
		args.device = torch.device('cuda', args.local_rank)
		deepspeed.init_distributed()
	args.global_rank = torch.distributed.get_rank()
	args.n_gpus = torch.distributed.get_world_size()
	torch.distributed.barrier()

	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
	config = read_json(args.config_path)
	model = GPT(config)
	print(model)
	if args.task in ['sft', 'dpo', 'orpo']:
		model = safe_load(model, args.model_path)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay, 
			"lr": args.lr
		},     
		{   
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
			"weight_decay": 0.0, 
			"lr": args.lr
		}           
	]
	AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
	optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.95))

	if args.task == 'pretrain':
		dataset_zh_dict = load_from_disk(args.zh_cache_path)
		dataset_en_dict = load_from_disk(args.en_cache_path)
		dataset_code_dict = load_from_disk(args.code_cache_path)
		dataset_wanjuan_dict = load_from_disk(args.wanjuan_cache_path)
		dataset_pile_dict = load_from_disk(args.pile_cache_path)
		train_dataset = interleave_datasets([dataset_zh_dict['train'], dataset_en_dict['train'], dataset_code_dict['train'], dataset_wanjuan_dict['train'], dataset_pile_dict['train']])
		val_dataset = interleave_datasets([dataset_zh_dict['validation'], dataset_en_dict['validation'], dataset_code_dict['validation'], dataset_wanjuan_dict['validation'], dataset_pile_dict['validation']])
		dataset_dict = {'train': train_dataset, 'validation': val_dataset}
		print(dataset_dict)
	elif args.task == 'sft':
		dataset_dict = load_from_disk(args.alpaca_cache_path)
	print('done loading data')
	if args.global_rank <= 0 and args.task in ['sft', 'dpo', 'orpo']:
		print_rank_0(dataset_dict['train'], tokenizer)
	if args.local_rank == -1:
		train_sampler = torch.utils.data.RandomSampler(dataset_dict['train'])
		valid_sampler = torch.utils.data.SequentialSampler(dataset_dict['validation'])
	else:
		train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_dict['train'])
		valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_dict['validation'])
	if args.task == 'pretrain':
		collate_fn = collate_for_lm
	elif args.task == 'sft':
		collate_fn = collate_for_sft
	elif args.task == 'dpo' or args.task == 'orpo':
		collate_fn = collate_for_dpo
	train_loader = torch.utils.data.DataLoader(dataset_dict['train'], batch_size=args.per_device_train_batch_size, sampler=train_sampler, collate_fn=collate_fn)
	valid_loader = torch.utils.data.DataLoader(dataset_dict['validation'], batch_size=args.per_device_train_batch_size, sampler=valid_sampler, collate_fn=collate_fn)
	
	t_total = len(train_loader) * args.epochs 
	warmup_steps = 0.06 * t_total
	lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
	os.makedirs(args.output_dir, exist_ok=True)
	scaler = torch.cuda.amp.GradScaler()
	if args.global_rank <= 0:
		writer = SummaryWriter('logs/%s' % args.exp_name)
	else:
		writer = None

	ds_config = get_train_ds_config(offload=args.offload, 
		stage=args.zero_stage, 
		global_batch_size=args.per_device_train_batch_size*args.gradient_accumulation_steps*args.n_gpus,
		micro_batch_size=args.per_device_train_batch_size,
		grad_acc=args.gradient_accumulation_steps,
		bf16=args.bf16)
	model, optimizer, _, lr_scheduler = deepspeed.initialize(
		model=model,
		optimizer=optimizer,
		args=args,
		config=ds_config,
		lr_scheduler=lr_scheduler,
		dist_init_required=True)
	if args.gradient_checkpointing:
		model.enable_gradient_checkpointing()
	train(model, train_loader, valid_loader, optimizer, lr_scheduler, scaler, writer)
	if args.global_rank <= 0:
		writer.close()

if __name__ == '__main__':
	main()