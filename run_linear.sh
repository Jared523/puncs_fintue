# !/bin/bash
#只有预训练
DATA_DIR='./data/'
MODEL_NAME_OR_PATH='/disc1/yu/puncs/chinese-roberta-wwm-ext'
OUTPUT='./output_linear'
CACHE_DIR='./data_linear_cache'

CUDA_VISIBLE_DEVICES=0 python run.py \
    --data_dir $DATA_DIR \
    --model_type bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT \
    --data_cache_dir $CACHE_DIR \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --seed 42 \
    --do_train \
    --do_eval \
    --overwrite_output_dir
