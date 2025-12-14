#!/bin/bash
# Part II: SQuAD data-size experiments (BitFit vs Full-FT)

set -e

TRACKER="${TRACKER:-none}"
TRAIN_SIZES="1000 5000 10000 25000 50000 100000"
METHODS="full_ft bitfit"

for method in $METHODS; do
    for size in $TRAIN_SIZES; do
        echo "============================================================"
        echo "Running: method=$method train_size=$size"
        echo "============================================================"
        uv run python scripts/train_squad.py method=$method train_size=$size tracker=$TRACKER
    done

    # full dataset
    echo "============================================================"
    echo "Running: method=$method train_size=full"
    echo "============================================================"
    uv run python scripts/train_squad.py method=$method tracker=$TRACKER
done

echo "Part II complete!"
