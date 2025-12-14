#!/bin/bash
# Part I: Single run per task/method combination (best LRs)

set -e

TRACKER="${TRACKER:-wandb}"
TASKS="sst2 mrpc rte"
METHODS="full_ft bitfit bitfit_subset lora prompt_tuning"

for task in $TASKS; do
    for method in $METHODS; do
        echo "============================================================"
        echo "Running: task=$task method=$method"
        echo "============================================================"
        uv run python scripts/train.py task=$task method=$method tracker=$TRACKER
    done
done

echo "Part I complete!"
