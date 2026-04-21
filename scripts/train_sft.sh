#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running SFT training (Multi-GPU) with eval + early stopping..."
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 -m amr.training.sft \
    --dataset1_path "data/train.amr" \
    --eval_dataset_path "data/dev.amr" \
    --output_dir "outputs/Qwen-1.5B-SFT" \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --deepspeed_path "config/ds_zero2.json" \
    --learning_rate 2e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_input_length 2048 \
    --num_train_epochs 5 \
    --eval_steps 200 \
    --metric_for_best_model eval_loss \
    --greater_is_better 0 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.0 \
    --resume_from_checkpoint auto \
    --use_lora 0 \
    2>&1 | tee Qwen-SFT.log
