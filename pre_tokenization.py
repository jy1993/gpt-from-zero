from utils import *
import os
import argparse
from transformers import AutoTokenizer
# from joblib import Parallel, delayed
# import joblib
import multiprocessing as mp
from functools import partial
from datasets import load_dataset
from baidubaike_preprocess import preprocess_pretrain_baidubaike_dataset, get_length

def get_batches(train_examples, batch_size):
    n = len(train_examples) // batch_size
    if n * batch_size < len(train_examples):
        n += 1
    batches = []
    for i in range(n):
        batches.append(train_examples[i*batch_size:i*batch_size+batch_size])
    return batches

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='pretrain')
parser.add_argument('--file_type', type=str, default='json')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--input_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--n_jobs', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--valid_dir', type=str, default=None)
parser.add_argument('--num_files', type=int, default=1)
parser.add_argument('--config_path', type=str, default='configs/0.9B.json')
parser.add_argument('--tokenizer_path', type=str, default='yi-tokenizer')
parser.add_argument('--train_filename', type=str, default='')
parser.add_argument('--valid_filename', type=str, default='')
parser.add_argument('--eos_token_id', type=int, default=None)
parser.add_argument('--pad_token_id', type=int, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    config = read_json(args.config_path)
    
    # train_examples = read_jsonl(os.path.join(args.input_path), args.num_samples)
    # features = convert_to_features(tokenizer, train_examples, config['max_length'])
    # torch.save(features, args.output_path)
    # features = Parallel(n_jobs=args.n_jobs)(delayed(convert_to_features)(tokenizer, examples, config['max_length']) for examples in get_examples(train_examples, args.batch_size))

    # func = partial(convert_to_features, tokenizer=tokenizer, max_length=config['max_length'])
    # with mp.Pool(processes=args.n_jobs) as pool:
    #     features = pool.map(func, get_batches(train_examples, args.batch_size))
    # print(sum(len(f) for f in features) * config['max_length'] / (10**9))
    # fast_dump(features, args.output_path)
    
    if args.task == 'pretrain':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        train_files = [os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir)]
        valid_files = [os.path.join(args.valid_dir, f) for f in os.listdir(args.valid_dir)]
        dataset = load_dataset(args.file_type, data_files={'train':train_files[:args.num_files], 'validation': valid_files[:args.num_files]})
        # dataset = dataset.select(range(1000))
        print(dataset)
        if args.dataset == 'code':
            dataset = dataset.filter(lambda x: x['language'] == 'Python')
            preprocess_fn = preprocess_pretrain_code_dataset
            remove_columns = ['code', 'repo_name', 'path', 'language', 'license', 'size']
        elif args.dataset == 'wikipedia':
            preprocess_fn = preprocess_pretrain_wikipedia_dataset
            remove_columns = ['id', 'url', 'title', 'text']
        elif args.dataset == 'baidubaike':
            dataset = dataset.filter(lambda x: get_length(x) > 100)
            preprocess_fn = preprocess_pretrain_baidubaike_dataset
            remove_columns = ['title', 'summary', 'sections', 'tags', 'url']
        func = partial(preprocess_fn, tokenizer=tokenizer, max_length=config['max_length'])
        dataset = dataset.map(func, batched=True, remove_columns=remove_columns, num_proc=args.n_jobs)
    elif args.task == 'sft':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        dataset = load_dataset(args.file_type, data_files={'train': args.train_filename, 'validation': args.valid_filename})
        print(dataset)
        func = partial(preprocess_sft_dataset_alpaca, tokenizer=tokenizer, max_length=config['max_length'], eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        dataset = dataset.map(func, batched=True, remove_columns=['instruction', 'input', 'output', 'history'], num_proc=args.n_jobs)
    print(dataset)
    dataset.save_to_disk(args.output_path)