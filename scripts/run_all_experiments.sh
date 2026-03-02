#!/usr/bin/env bash
# Run all experiments: 3 models × 2 regimes × 3 seeds = 18 pipeline runs.
set -euo pipefail

MODELS=(
  "google/gemma-3-1b-it:gemma3_1b"
  "google/gemma-2-2b-it:gemma2_2b"
  "google/gemma-3-4b-it:gemma3_4b"
)
SEEDS=(42 43 44)

for entry in "${MODELS[@]}"; do
  base_model="${entry%%:*}"
  exp_base="${entry##*:}"

  for seed in "${SEEDS[@]}"; do
    # LoRA
    uv run python scripts/pipeline.py \
      configs/pipeline/leave_one_out_lora.yaml \
      --base-model "${base_model}" --experiment-name "${exp_base}" --seed "${seed}"

    # Full fine-tuning
    uv run python scripts/pipeline.py \
      configs/pipeline/leave_one_out_fullft.yaml \
      --base-model "${base_model}" --experiment-name "${exp_base}_fullft" --seed "${seed}"
  done
done
