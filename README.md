# gpt-from-zero
this repo records how i train a gpt-style model from scratch

## phase1: pretrain
### step1: data preparation
python pre_tokenization.py --train_dir train/zh --valid_dir val/zh --output_path cached/zh --num_files 100 --n_jobs 12 --dataset

### step2: train the pt model
deepspeed --include localhost:0,1 pretrain_ds.py --exp_name gpt_1B --output_dir gpt_1B --zero_stage 2 --config_path configs/1B.json --gradient_checkpointing --per_device_train_batch_size 72 --bf16 --offload

## phase2: supervised-finetuning
### step1: data preparation
python pre_tokenization.py --task sft --train_filename sft-data/alpaca_chinese_train.json --valid_filename sft-data/alpaca_chinese_train.json --output_path cached/sft

### step2: train the sft model
deepspeed --include localhost:0 pretrain_ds.py --task sft --exp_name gpt_1B_sft --output_dir gpt_1B_sft --zero_stage 0 --config_path configs/1B.json --gradient_checkpointing --per_device_train_batch_size 24 --model_path gpt_1B/pretrain_steps_3000 --eval_steps 300 --save_steps 300 --epochs 3 --lr 3e-5 --bf16

## phase3: DPO/ORPO
### step1: data preparation
python pre_tokenization.py --task dpo --train_filename dpo-data/alpaca_chinese_train.json --valid_filename dpo-data/alpaca_chinese_train.json --output_path cached/dpo

### step2: train the dpo/ocpo model
deepspeed --include localhost:5,6 --master_port 29600 pretrain_ds.py --task orpo --exp_name gpt_1B_orpo --output_dir gpt_1B_orpo --zero_stage 2 --config_path configs/1B.json --gradient_checkpointing --per_device_train_batch_size 18 --model_path gpt_1B/pretrain_steps_20000 --eval_steps 1000 --save_steps 1000 --epochs 3 --lr 1e-5 --bf16