# !/bin/bash
#预训练加对比学习加linear
DATA_DIR='./data/'
MODEL_NAME_OR_PATH='/disc1/yu/puncs/chinese-roberta-wwm-ext'
OUTPUT='./output_pcf'
CACHE_DIR='./data_pcf'

CUDA_VISIBLE_DEVICES=1 python run.py \
    --data_dir $DATA_DIR \
    --model_type bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT \
    --data_cache_dir $CACHE_DIR \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 116 \
    --per_gpu_eval_batch_size 64 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --seed 42 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --do_freeze
