"""Supervised fine-tuning entrypoint.

Saves ONLY the best checkpoint (by eval loss) and overwrites previous ones.
Stops training early if the eval loss stops improving.
"""
import argparse
import glob
import os
import shutil

import torch
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from ..dataset import get_data
from ._common import add_common_args, build_lora_config, build_model_and_tokenizer


def _find_last_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    return ckpts[-1]


def _cleanup_non_best_checkpoints(output_dir: str, best_ckpt: str | None) -> None:
    if not os.path.isdir(output_dir):
        return
    for p in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        if best_ckpt and os.path.abspath(p) == os.path.abspath(best_ckpt):
            continue
        print(f"[cleanup] Removing {p}")
        shutil.rmtree(p, ignore_errors=True)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_data(args.dataset1_path, args.dataset2_path, type="sft")
    eval_dataset = get_data(args.eval_dataset_path, type="sft") if args.eval_dataset_path else None
    if eval_dataset is None:
        raise ValueError(
        )

    model, tokenizer = build_model_and_tokenizer(args.model_name, device)
    peft_config = build_lora_config(args)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        completion_only_loss=True,
        deepspeed=args.deepspeed_path,
        max_length=args.max_input_length,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
    ]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config,
        callbacks=callbacks,
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
            print(f"[resume] Using checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    best_ckpt = trainer.state.best_model_checkpoint
    print(f"[train] Best checkpoint: {best_ckpt}")
    print(f"[train] Best metric ({args.metric_for_best_model}): {trainer.state.best_metric}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _cleanup_non_best_checkpoints(args.output_dir, best_ckpt=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for ViAMR")
    add_common_args(parser)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--eval_dataset_path", type=str, default="data/dev.amr")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_loss",
        help="Metric to track for best-model selection.",
    )
    parser.add_argument(
        "--greater_is_better",
        type=lambda v: str(v).lower() in ("1", "true", "yes"),
        default=False,
        help="True if higher metric is better (e.g. accuracy). False for loss.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop if eval metric doesn't improve for this many evals in a row.",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum improvement to count as progress.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint, or 'auto' to resume the latest in --output_dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
