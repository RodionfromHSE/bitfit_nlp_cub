import time

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

from src.analysis import bytes_to_gb, count_parameters, get_peak_memory, reset_memory_stats
from src.tracking import ExperimentTracker


class InstrumentedTrainer(Trainer):
    """Trainer with memory and timing instrumentation."""

    def __init__(self, *args, experiment_tracker: ExperimentTracker | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_tracker = experiment_tracker
        self.epoch_times: list[float] = []
        self._epoch_start_time: float | None = None

    def train(self, *args, **kwargs):
        reset_memory_stats()
        start_time = time.perf_counter()

        result = super().train(*args, **kwargs)

        total_time = time.perf_counter() - start_time
        peak_memory = get_peak_memory()

        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(
                {
                    "total_training_time_seconds": total_time,
                    "peak_memory_gb": bytes_to_gb(peak_memory),
                }
            )

        return result


class TrackingCallback(TrainerCallback):
    """Callback to log metrics to experiment tracker."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self._epoch_start_time: float | None = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start_time = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start_time is not None:
            epoch_time = time.perf_counter() - self._epoch_start_time
            self.tracker.log_metrics({"epoch_time_seconds": epoch_time}, step=int(state.epoch))
            self._epoch_start_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            filtered_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            self.tracker.log_metrics(filtered_logs, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            filtered = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.tracker.log_metrics(filtered, step=state.global_step)


def create_training_args(cfg, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        learning_rate=cfg.method.lr,
        weight_decay=cfg.method.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy=cfg.training.save_strategy,
        logging_steps=cfg.training.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.task.metric,
        greater_is_better=True,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        seed=cfg.training.seed,
        max_steps=cfg.training.max_steps,
        report_to="none",
        remove_unused_columns=False,
    )

