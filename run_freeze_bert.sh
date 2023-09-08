# !/bin/bash
#预训练加对比学习加linear
DATA_DIR='./data/'
MODEL_NAME_OR_PATH='/disc1/yu/SCL/checkpoint-10000'
OUTPUT='./deep_output_freeze'
CACHE_DIR='./data_freeze_cache'

CUDA_VISIBLE_DEVICES=2 python run.py \
    --data_dir $DATA_DIR \
    --model_type bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT \
    --data_cache_dir $CACHE_DIR \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 116 \
    --per_gpu_eval_batch_size 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --seed 32 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --do_freeze
