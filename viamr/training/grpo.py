"""GRPO (Group Relative Policy Optimization) training entrypoint."""
import argparse
import glob
import os

import torch
from trl import GRPOConfig, GRPOTrainer

from ..dataset import get_data
from ..rewards import combined_reward
from ._common import add_common_args, build_lora_config, build_model_and_tokenizer


def _find_last_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    return ckpts[-1]


def main(args: argparse.Namespace) -> None:
    if os.environ.get("WANDB_MODE") != "disabled":
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        report_to = "wandb"
    else:
        report_to = "none"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_data(args.dataset1_path, args.dataset2_path, type="grpo")
    model, tokenizer = build_model_and_tokenizer(args.model_name, device)
    peft_config = build_lora_config(args)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        log_on_each_node=args.log_on_each_node,
        temperature=args.temperature,
        use_vllm=False,
        vllm_gpu_memory_utilization=0.6,
        report_to=report_to,
        deepspeed=args.deepspeed_path,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[combined_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    resume_from: str | bool | None = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            last = _find_last_checkpoint(args.output_dir)
            if last:
                print(f"[resume] Found checkpoint: {last}")
                resume_from = last
            else:
                print("[resume] No checkpoint found, starting fresh.")
        else:
            resume_from = args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for ViAMR")
    add_common_args(parser)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--log_on_each_node", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="viamr")
    parser.add_argument("--wandb_run_name", type=str, default="grpo-run")
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint, or 'auto' to pick the latest inside --output_dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
