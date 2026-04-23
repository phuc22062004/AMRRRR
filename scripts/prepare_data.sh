#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Validating ViAMR dataset splits in ./data ..."
python - <<'PY'
from viamr.data_processing import read_amr_direct

for split in ("train", "dev", "test"):
    path = f"data/{split}.amr"
    df = read_amr_direct(path)
    n = len(df)
    if n == 0:
        raise SystemExit(f"[FAIL] {path} parsed 0 graphs")
    in_lens = df["query"].str.split().map(len)
    out_lens = df["amr"].str.split().map(len)
    print(
        f"[OK] {split:5s}: {n} graphs | "
        f"query tokens max/avg = {in_lens.max()}/{in_lens.mean():.1f} | "
        f"amr tokens max/avg = {out_lens.max()}/{out_lens.mean():.1f}"
    )
print("All splits parsed successfully.")
PY
