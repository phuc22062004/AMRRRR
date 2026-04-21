#!/bin/bash
set -e

# GRPO training (Group Relative Policy Optimization)
# Run AFTER SFT to optimize for Smatch F1 via reinforcement learning

cd "$(dirname "$0")/.."

echo "Running GRPO training..."
export CUDA_VISIBLE_DEVICES=0,1

# Set WANDB_MODE=disabled if you don't want to log to Weights & Biases
# Or login first: wandb login

torchrun --nproc_per_node=2 -m amr.training.grpo \
    --dataset1_path "data/train.amr" \
    --output_dir "outputs/Qwen-1.5B-GRPO" \
    --model_name "outputs/Qwen-1.5B-SFT" \
    --deepspeed_path "config/ds_zero2.json" \
    --learning_rate 5e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_generations 4 \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --save_steps 200 \
    --log_on_each_node \
    --use_lora 0 \
    --wandb_project "viamr" \
    --wandb_run_name "grpo-qwen1.5b" \
    2>&1 | tee Qwen-GRPO.log
