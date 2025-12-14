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
from transformers import Trainer, TrainingArguments

from src.analysis import bytes_to_gb, count_parameters, get_peak_memory, print_model_summary, reset_memory_stats
from src.data import get_squad_data_collator, get_tokenizer, prepare_squad_dataset
from src.metrics import compute_squad_metrics, postprocess_qa_predictions
from src.models import create_qa_model
from src.tracking import create_tracker
from src.trainer import TrackingCallback


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="squad_config")
def main(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.training.seed)

    tracker = create_tracker(cfg.tracker, cfg)

    tokenizer = get_tokenizer()
    train_dataset, eval_features, eval_raw = prepare_squad_dataset(
        tokenizer=tokenizer,
        train_size=cfg.get("train_size"),
        seed=cfg.training.seed,
    )

    model = create_qa_model(method=cfg.method.name)
    print_model_summary(model, model_name=f"{cfg.method.name} on SQuAD")

    stats = count_parameters(model)
    tracker.log_params({
        "total_params": stats.total,
        "trainable_params": stats.trainable,
        "trainable_percentage": stats.percentage,
        "train_size": len(train_dataset),
    })

    eval_dataset = eval_features.remove_columns(["example_id", "offset_mapping"])

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        learning_rate=cfg.method.lr,
        weight_decay=cfg.method.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        eval_strategy="no",
        save_strategy=cfg.training.save_strategy,
        logging_steps=cfg.training.logging_steps,
        load_best_model_at_end=False,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        seed=cfg.training.seed,
        max_steps=cfg.training.max_steps,
        report_to="none",
    )

    data_collator = get_squad_data_collator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[TrackingCallback(tracker)],
    )

    reset_memory_stats()
    trainer.train()
    peak_memory = get_peak_memory()

    tracker.log_metrics({"peak_memory_gb": bytes_to_gb(peak_memory)})

    predictions = trainer.predict(eval_dataset)
    raw_preds = (predictions.predictions[0], predictions.predictions[1])

    text_predictions = postprocess_qa_predictions(eval_raw, eval_features, raw_preds)
    metrics = compute_squad_metrics(text_predictions, eval_raw)

    print(f"\nEvaluation results: {metrics}")
    tracker.log_metrics({f"final_{k}": v for k, v in metrics.items()})

    tracker.finish()
    return metrics["f1"]


if __name__ == "__main__":
    main()
