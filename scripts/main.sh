#!/bin/bash
set -e

cd "$(dirname "$0")/.."

#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "===== ViAMR Full Pipeline ====="
echo ""
echo "Step 1/4: Validating data..."
bash scripts/prepare_data.sh
echo ""

echo "Step 2/4: Training SFT (multi-GPU)..."
echo "  (Use scripts/train_sft_single_gpu.sh for single GPU)"
bash scripts/train_sft.sh
echo ""

echo "Step 3/4: Running inference on test set..."
bash scripts/infer.sh
echo ""

echo "Step 4/4: Computing Smatch score..."
bash scripts/get_score.sh
echo ""

echo "===== Pipeline Complete ====="
echo ""
echo "Optional: Run GRPO training to boost Smatch further"
echo "  bash scripts/train_grpo.sh  # multi-GPU"
echo "  bash scripts/train_grpo_single_gpu.sh  # single GPU"
