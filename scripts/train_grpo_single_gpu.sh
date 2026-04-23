#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running GRPO training (Single GPU)..."

WANDB_MODE=disabled python -m viamr.training.grpo \
    --dataset1_path "data/train.amr" \
    --output_dir "outputs/Qwen-1.5B-GRPO" \
    --model_name "outputs/Qwen-1.5B-SFT" \
    --learning_rate 5e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 50 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 4 \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --max_steps 500 \
    --temperature 0.8 \
    --num_train_epochs 1 \
    --save_steps 100 \
    --save_total_limit 5 \
    --use_lora 1 \
    --lora_r 32 \
    --lora_alpha 64 \
    --wandb_project "viamr" \
    --wandb_run_name "grpo-qwen1.5b-single" \
    2>&1 | tee Qwen-GRPO-single.log
