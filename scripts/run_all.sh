#!/usr/bin/env bash
# Run every experimental condition x 5 seeds (40 runs), then aggregate.
# Each run writes its own JSON to results/runs/, so the script can be
# re-run after an interruption: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1 2 3 4)

for config in configs/*.yaml; do
  dataset_condition="$(basename "$config" .yaml)"
  for seed in "${SEEDS[@]}"; do
    out="results/runs/${dataset_condition}_seed${seed}.json"
    if [[ -f "$out" ]]; then
      echo "skip ${dataset_condition} seed ${seed} (already done)"
      continue
    fi
    echo "=== ${dataset_condition} seed ${seed} ==="
    python -m src.train --config "$config" --seed "$seed"
  done
done

python scripts/aggregate.py
