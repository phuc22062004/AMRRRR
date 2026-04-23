"""Inference pipeline: QwenReasoner and the batch AMR extraction CLI."""
import argparse
import os
import re

import penman
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_processing import read_amr_direct
from .postprocessing import has_duplicate_nodes, join_concepts_underscores, penman_safe_minimal
from .prompts import SYSTEM_PROMPT


class QwenReasoner:
    def __init__(self, model_name: str, lora_path: str | None = None, device: str = "cuda:0"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

    def inference(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        is_extract_amr: bool = False,
        is_thinking: bool = False,
    ):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sentence: {prompt}"},
        ]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=is_thinking, **template_kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        if is_thinking:
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            if is_extract_amr:
                return thinking_content, self._extract_answer(content)
            return thinking_content, content

        trimmed = [
            out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        decoded = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        if is_extract_amr:
            return None, self._extract_answer(decoded)
        return None, decoded

    @staticmethod
    def _extract_answer(text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return match.group(1).strip() if match else ""


def _iter_input_lines(input_file: str, my_test: bool) -> list[str]:
    if my_test:
        return read_amr_direct(input_file)["query"].tolist()
    with open(input_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    lines = _iter_input_lines(args.input_file, bool(args.my_test))
    model = QwenReasoner(model_name=args.model_name)

    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for idx, line in enumerate(lines):
            amr_str = "fail"
            retry_count = 0
            while retry_count < args.max_retries:
                _, predict = model.inference(
                    line,
                    is_extract_amr=True,
                    is_thinking=True,
                )
                try:
                    predict = join_concepts_underscores(predict)
                    graph = penman.decode(predict)
                    amr_str = penman.encode(graph)
                    break
                except Exception:
                    print(f"[Error] Cannot decode AMR (try {retry_count + 1})")
                    retry_count += 1

            try:
                amr_str = penman_safe_minimal(amr_str)
                graph = penman.decode(amr_str)
                amr_str = penman.encode(graph)
                print("[Success] Processed AMR")
            except Exception as e:
                print(f"[Error] Failed to process AMR after retries: {e}")

            if has_duplicate_nodes(amr_str):
                print(f"[Warning] AMR has duplicate nodes: {amr_str}")

            out_f.write(f"#::snt {idx} {line}\n")
            out_f.write(f"{amr_str}\n\n")
            out_f.flush()

            print(f"Processed {idx}: {line} (Retries: {retry_count}): amr: {amr_str}")

    print(f"Save completed. Results saved to {args.output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMR inference pipeline.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen-7B")
    parser.add_argument("--my_test", type=int, default=0, help="Input file is in AMR format (use `query` column)")
    parser.add_argument("--max_retries", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
