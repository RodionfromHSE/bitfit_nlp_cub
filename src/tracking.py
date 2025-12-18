from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import mlflow
import wandb
from omegaconf import DictConfig, OmegaConf


@runtime_checkable
class ExperimentTracker(Protocol):
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def finish(self) -> None: ...


class WandbTracker:
    def __init__(
        self,
        project: str,
        entity: str | None = None,
        config: dict | None = None,
        run_name: str | None = None,
    ):
        self.run = wandb.init(project=project, entity=entity, config=config, name=run_name)

    def log_params(self, params: dict) -> None:
        wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        artifact = wandb.Artifact(name=path.replace("/", "_"), type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        wandb.finish()


class MLflowTracker:
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        config: dict | None = None,
        run_name: str | None = None,
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        if config:
            self.log_params(config)

    def log_params(self, params: dict) -> None:
        flat_params = _flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def finish(self) -> None:
        mlflow.end_run()


class NoOpTracker:
    """Tracker that does nothing, for testing or when tracking is disabled."""

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def finish(self) -> None:
        pass


def create_tracker(tracker_cfg: DictConfig, full_config: DictConfig) -> ExperimentTracker:
    config_dict = OmegaConf.to_container(full_config, resolve=True)
    run_name = _build_run_name(full_config)

    if tracker_cfg.backend == "wandb":
        return WandbTracker(
            project=tracker_cfg.project,
            entity=tracker_cfg.get("entity"),
            config=config_dict,
            run_name=run_name,
        )
    elif tracker_cfg.backend == "mlflow":
        return MLflowTracker(
            tracking_uri=tracker_cfg.tracking_uri,
            experiment_name=tracker_cfg.experiment_name,
            config=config_dict,
            run_name=run_name,
        )
    elif tracker_cfg.backend == "none":
        return NoOpTracker()
    else:
        raise ValueError(f"Unknown tracker backend: {tracker_cfg.backend}")


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _build_run_name(cfg: DictConfig) -> str:
    parts = []
    for path in ("task.dataset", "task.name", "method.name"):
        value = OmegaConf.select(cfg, path, default=None)
        if value:
            parts.append(value)

    lr = OmegaConf.select(cfg, "method.lr", default=None)
    if lr is not None:
        parts.append(f"lr{lr}")

    seed = OmegaConf.select(cfg, "training.seed", default=None)
    if seed is not None:
        parts.append(f"seed{seed}")

    return "-".join(str(p) for p in parts if p is not None)
