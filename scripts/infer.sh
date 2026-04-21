#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running inference..."
python3 -m amr.inference \
    --input_file "data/test.amr" \
    --output_file "results.txt" \
    --model_name "outputs/Qwen-1.5B-SFT" \
    --my_test 1 \
    --max_retries 5
