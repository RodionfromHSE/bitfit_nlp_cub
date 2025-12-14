#!/bin/bash
# Part I: Full grid search with multiple seeds

set -e

TRACKER="${TRACKER:-none}"

uv run python scripts/sweep.py --mode full --tracker $TRACKER

echo "Part I grid search complete!"
