#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running scoring..."
python3 -m amr.scoring \
    --predict_file "results.txt" \
    --gold_file "data/test.amr"
