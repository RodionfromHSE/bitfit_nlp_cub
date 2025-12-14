# BitFit and PEFT Experiments

> Me and my kent Nikita are doing bitfit stuff

Comparing parameter-efficient fine-tuning methods on GLUE tasks and SQuAD.

## Setup

```bash
pip install uv # if you don't have it
uv sync
```

## Project Structure

- `src/` - core modules (data loading, models, metrics, training)
- `scripts/` - training scripts
- `configs/` - hydra configs for tasks, methods, and training

## Part I: GLUE Experiments

Train on SST-2, MRPC, or RTE with different PEFT methods.

```bash
# single run
uv run python scripts/train.py task=sst2 method=full_ft

# available tasks: sst2, mrpc, rte
# available methods: full_ft, bitfit, bitfit_subset, lora, prompt_tuning
```

Override any config parameter:
```bash
uv run python scripts/train.py task=mrpc method=lora method.lr=1e-4 training.epochs=5
```

Run a sweep across tasks/methods/seeds:
```bash
uv run python scripts/sweep.py
```

## Part II: SQuAD Data-Size Experiments

Compare Full-FT vs BitFit on SQuAD with varying training set sizes.

```bash
# run with subset of training data
uv run python scripts/train_squad.py method=bitfit train_size=1000

# run with full dataset
uv run python scripts/train_squad.py method=full_ft

# available methods: full_ft, bitfit
# train_size options: 1000, 5000, 10000, 25000, 50000, 100000, or null (full)
```

## Config System

Uses Hydra. Configs are in `configs/`:
- `config.yaml` - main config for GLUE (defaults + training params)
- `squad_config.yaml` - main config for SQuAD
- `task/` - task-specific configs (sst2, mrpc, rte, squad)
- `method/` - method configs with lr grids (full_ft, bitfit, lora, etc.)
- `tracker/` - experiment tracking (wandb, mlflow, none)

Enable wandb tracking:
```bash
uv run python scripts/train.py tracker=wandb tracker.project=my-project
```

## Running All Experiments

Shell scripts to run full experiments:

```bash
./part1.sh        # Part I: all GLUE tasks × all methods (single run each)
./part1_grid.sh   # Part I: full grid search (3 tasks × 5 methods × 3 seeds)
./part2.sh        # Part II: SQuAD data-size experiments (2 methods × 7 sizes)
```

Enable tracking by setting TRACKER env var:
```bash
TRACKER=wandb ./part1.sh
```

## Quick Test

Run a short test with limited steps:
```bash
uv run python scripts/train.py training.max_steps=50 tracker=none
uv run python scripts/train_squad.py training.max_steps=50 train_size=100
```
