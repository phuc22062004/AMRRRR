"""Reward and scoring functions for GRPO training."""
import re

import smatch

from .postprocessing import (
    balance_parens,
    fix_amr_vars,
    penman_safe_minimal,
)


def get_amr_match(amr1: str, amr2: str):
    vals = smatch.get_amr_match(amr1, amr2)
    smatch.match_triple_dict.clear()
    return vals


def compute_smatch_f1(gold_str: str, pred_str: str) -> tuple[float, float, float]:
    try:
        M, T, G = get_amr_match(gold_str, pred_str)
        precision = M / T if T else 0
        recall = M / G if G else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    except Exception as e:
        print(e)
        f1, precision, recall = 0.0, 0.0, 0.0
    return f1, precision, recall


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text.strip(), flags=re.DOTALL)
    return match.group(1).strip() if match else None


def check_valid_format(text: str) -> bool:
    return bool(text and re.search(r"<answer>.*?</answer>", text.strip(), flags=re.DOTALL))


def check_balanced_parens(s: str) -> bool:
    stack = 0
    for ch in s:
        if ch == '(':
            stack += 1
        elif ch == ')':
            stack -= 1
            if stack < 0:
                return False
    return stack == 0


def check_unique_vars(amr_str: str) -> bool:
    vars_found = re.findall(r"\((\w+)\s*/", amr_str)
    return len(vars_found) == len(set(vars_found))


def check_var_word_conflict(amr_str: str) -> bool:
    matches = re.findall(r"\((\w+)\s*/\s*([^\s)]+)", amr_str)
    if not matches:
        return False
    for var, word in matches:
        if not word or var[0].lower() != word[0].lower():
            return False
    return True


def combined_reward(prompts, completions, answers, **kwargs) -> list[float]:
    scores = []
    for completion, gold in zip(completions, answers):
        response_text = completion[0]['content'].strip()

        try:
            pred_answer = extract_answer(response_text) or ""
            pred_answer = penman_safe_minimal(pred_answer)
        except Exception:
            pred_answer = ''

        gold_answer = extract_answer(gold) or gold.strip()

        format_score = 0.1 if check_valid_format(response_text) else 0.0
        paren_score = 0.1 if pred_answer and check_balanced_parens(pred_answer) else 0.0
        unique_var_score = 0.1 if pred_answer and check_unique_vars(pred_answer) else 0.0
        var_word_conflict_score = 0.1 if pred_answer and check_var_word_conflict(pred_answer) else 0.0

        smatch_f1 = 0.0
        if pred_answer:
            smatch_f1, _, _ = compute_smatch_f1(gold_answer, pred_answer)
        smatch_score = 0.6 * smatch_f1

        total = min(format_score + paren_score + unique_var_score + var_word_conflict_score + smatch_score, 1.0)
        scores.append(total)

        print(
            f"Format: {format_score:.2f}, Parens: {paren_score:.2f}, "
            f"Unique vars: {unique_var_score:.2f}, No var-word conflict: {var_word_conflict_score:.2f}, "
            f"Smatch: {smatch_f1:.4f} ({smatch_score:.2f}), Total: {total:.4f}"
        )

    return scores
