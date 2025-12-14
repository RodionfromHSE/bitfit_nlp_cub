import evaluate
import numpy as np


def get_compute_metrics(task_name: str):
    """Returns a compute_metrics function for the given GLUE task."""

    if task_name == "mrpc":
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_metric.compute(predictions=predictions, references=labels)
            f1 = f1_metric.compute(predictions=predictions, references=labels)
            return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

        return compute_metrics

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    return compute_metrics


def get_metric_for_best_model(task_name: str) -> str:
    """Returns the metric name to use for selecting the best model."""
    if task_name == "mrpc":
        return "f1"
    return "accuracy"

