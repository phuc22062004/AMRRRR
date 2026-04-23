import pandas as pd
from datasets import Dataset

from .data_processing import read_amr_direct
from .prompts import SYSTEM_PROMPT


def _build_user_prompt(sentence: str) -> str:
    return f"Sentence: {sentence}"


def get_data(train_path1: str, train_path2: str | None = None, type: str = "grpo") -> Dataset:
    df = read_amr_direct(train_path1)
    if train_path2:
        df2 = read_amr_direct(train_path2)
        df = pd.concat([df, df2], ignore_index=True)

    records = []
    max_in, max_out = 0, 0
    for _, row in df.iterrows():
        user_prompt = _build_user_prompt(row["query"])
        max_in = max(max_in, len(user_prompt.split()))
        max_out = max(max_out, len(row["amr"].split()))
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if type == "grpo":
            records.append({"prompt": prompt, "answers": row["amr"]})
        else:
            records.append({
                "prompt": prompt,
                "completion": [
                    {"role": "assistant", "content": f"<answer>{row['amr']}</answer>"}
                ],
            })

    print(f"Max input length: {max_in}, Max output length: {max_out}")
    return Dataset.from_list(records)
