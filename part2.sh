#!/bin/bash
# Part II: SQuAD data-size experiments (BitFit vs Full-FT)

set -euo pipefail

# Default to Weights & Biases tracking for SQuAD runs; override with TRACKER env if needed.
TRACKER="${TRACKER:-wandb}"

# More train-set sizes (log-ish spaced); override with TRAIN_SIZES env if desired.
# Note: values larger than the true SQuAD train split size are treated as "full" by the loader.
TRAIN_SIZES="${TRAIN_SIZES:-500 1000 2000 3000 5000 7000 10000 15000 20000}"

# Step budget policy:
# By default, pick a per-dataset max_steps that scales with dataset size, but never goes below MIN_STEPS.
# This avoids small datasets getting too few optimizer updates when training with a fixed epoch count.
#
# You can override behavior via env:
# - MAX_STEPS: if set (and not -1), forces the same max_steps for all sizes.
# - MIN_STEPS: minimum optimizer updates for small datasets (default 2000).
# - TARGET_EPOCHS: epoch-equivalent updates for larger datasets (default 2).
# - BATCH_SIZE / GRAD_ACCUM: used only for step estimation (defaults match configs/squad_config.yaml).
MIN_STEPS="${MIN_STEPS:-2000}"
TARGET_EPOCHS="${TARGET_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
FULL_MAX_STEPS="${FULL_MAX_STEPS:-15000}"

METHODS="full_ft bitfit"

steps_for_size() {
    local size="$1"

    if [[ "$size" == "full" ]]; then
        echo "$FULL_MAX_STEPS"
        return
    fi

    if [[ -n "${MAX_STEPS:-}" && "${MAX_STEPS:-}" != "-1" ]]; then
        echo "$MAX_STEPS"
        return
    fi

    local eff_bs=$((BATCH_SIZE * GRAD_ACCUM))
    if (( eff_bs <= 0 )); then
        echo "Error: effective batch size must be > 0 (BATCH_SIZE=$BATCH_SIZE, GRAD_ACCUM=$GRAD_ACCUM)" >&2
        exit 1
    fi

    local steps_per_epoch=$(( (size + eff_bs - 1) / eff_bs ))
    local target_steps=$(( steps_per_epoch * TARGET_EPOCHS ))

    if (( target_steps < MIN_STEPS )); then
        target_steps="$MIN_STEPS"
    fi

    echo "$target_steps"
}

for method in $METHODS; do
    for size in $TRAIN_SIZES; do
        steps="$(steps_for_size "$size")"
        echo "============================================================"
        echo "Running: part2 method=$method train_size=$size max_steps=$steps"
        echo "============================================================"
        uv run python scripts/train_squad.py \
            method="$method" \
            train_size="$size" \
            tracker="$TRACKER" \
            training.max_steps="$steps" \
            +task.dataset="squad" \
            +task.name="part2_ds${size}"
    done

    # full dataset
    steps="$(steps_for_size "full")"
    echo "============================================================"
    echo "Running: part2 method=$method train_size=full max_steps=$steps"
    echo "============================================================"
    uv run python scripts/train_squad.py \
        method="$method" \
        tracker="$TRACKER" \
        training.max_steps="$steps" \
        +task.dataset="squad" \
        +task.name="part2_dsfull"
done

echo "Part II complete!"
