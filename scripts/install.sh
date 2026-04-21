#!/bin/bash
set -e

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip wheel setuptools

# Install torch first so later packages (e.g. deepspeed) can import it
# during their build step. Adjust the CUDA index URL for your machine.
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing torch (CUDA 12.1 wheels)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

OS="$(uname -s 2>/dev/null || echo Windows)"
case "$OS" in
    Linux*)
        echo "Detected Linux -> installing full stack (DeepSpeed, vLLM, mpi4py)."
        # Disable op pre-compilation during install; ops are JIT-compiled on first use.
        DS_BUILD_OPS=0 pip install --no-build-isolation -r requirements-linux.txt
        ;;
    *)
        echo "Detected $OS -> installing Windows-compatible core only."
        echo "For DeepSpeed / vLLM / mpi4py, use WSL2 + scripts/install.sh."
        pip install -r requirements.txt
        ;;
esac

echo "Done. Verifying core imports..."
python - <<'PY'
import importlib
for m in ("torch", "transformers", "trl", "peft", "penman", "smatch", "pandas"):
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
PY
