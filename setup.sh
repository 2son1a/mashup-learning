#!/usr/bin/env bash
# setup.sh — one-command setup for Mashup Learning
# Usage: bash setup.sh
set -euo pipefail

echo "=== Mashup Learning Setup ==="

# 1. Check that uv is available (it handles Python version automatically via pyproject.toml)
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
echo "[1/6] uv found (Python version managed automatically via pyproject.toml)"

# 2. Install dependencies (including analysis extras for plots/tables)
echo "[2/6] Installing dependencies (uv sync)..."
uv sync --extra analysis

# 3. Install axolotl with flash-attn and deepspeed
echo "[3/6] Installing axolotl (with flash-attn & deepspeed)..."
uv pip install --no-build-isolation "axolotl[flash-attn,deepspeed]"

# 4. Patch axolotl/trl/transformers for transformers 5.x compatibility
echo "[4/6] Patching packages for transformers 5.x compatibility..."
uv run python scripts/postinstall.py

# 5. Create .env if it doesn't exist
if [[ ! -f .env ]]; then
    echo "[5/6] Creating .env from .env.example..."
    sed \
        -e 's/^WANDB_API_KEY=.*/WANDB_API_KEY=/' \
        -e 's/^HF_TOKEN=.*/HF_TOKEN=/' \
        .env.example > .env
    echo "     (Edit .env to add your WANDB_API_KEY and HF_TOKEN if needed)"
else
    echo "[5/6] .env already exists, skipping"
fi

# 6. Preprocess datasets
echo "[6/6] Preprocessing datasets..."
uv run python scripts/preprocess_datasets.py --all

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  # (Optional) Edit .env with your WandB API key and HuggingFace token"
echo "  # Run the test pipeline:"
echo "  uv run python scripts/pipeline.py configs/pipeline/test_lora.yaml"
