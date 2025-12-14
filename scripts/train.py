import os
import random

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    from scripts.setup_gpu import setup_cuda_device

    gpu_idx = setup_cuda_device()
    print(f"Auto-selected GPU {gpu_idx}")

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.analysis import count_parameters, print_model_summary
from src.data import get_data_collator, get_tokenizer, prepare_glue_dataset
from src.metrics import get_compute_metrics
from src.models import create_model
from src.tracking import create_tracker
from src.trainer import InstrumentedTrainer, TrackingCallback, create_training_args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.training.seed)

    tracker = create_tracker(cfg.tracker, cfg)

    tokenizer = get_tokenizer()
    train_dataset, eval_dataset = prepare_glue_dataset(
        task_name=cfg.task.name,
        sentence1_key=cfg.task.sentence1_key,
        sentence2_key=cfg.task.sentence2_key,
        tokenizer=tokenizer,
    )

    method_cfg = OmegaConf.to_container(cfg.method, resolve=True)
    model = create_model(
        method=cfg.method.name,
        num_labels=cfg.task.num_labels,
        method_cfg=method_cfg,
    )

    print_model_summary(model, model_name=f"{cfg.method.name} on {cfg.task.name}")

    stats = count_parameters(model)
    tracker.log_params(
        {
            "total_params": stats.total,
            "trainable_params": stats.trainable,
            "trainable_percentage": stats.percentage,
        }
    )

    training_args = create_training_args(cfg, cfg.output_dir)

    data_collator = get_data_collator(tokenizer)
    compute_metrics = get_compute_metrics(cfg.task.name)

    trainer = InstrumentedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        experiment_tracker=tracker,
        callbacks=[TrackingCallback(tracker)],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"\nEvaluation results: {eval_results}")

    tracker.log_metrics(
        {f"final_{k}": v for k, v in eval_results.items() if isinstance(v, (int, float))}
    )

    tracker.finish()

    metric_key = f"eval_{cfg.task.metric}"
    return eval_results.get(metric_key, 0.0)


if __name__ == "__main__":
    main()

