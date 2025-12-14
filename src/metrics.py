import collections

import evaluate
import numpy as np
from datasets import Dataset


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


def postprocess_qa_predictions(
    examples: Dataset,
    features: Dataset,
    raw_predictions: tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> dict[str, str]:
    """Convert model outputs to text predictions."""
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        min_null_score = None
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = 0
            null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions


def compute_squad_metrics(predictions: dict[str, str], references: Dataset) -> dict[str, float]:
    """Compute SQuAD EM and F1 metrics."""
    squad_metric = evaluate.load("squad")

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    formatted_references = [{"id": ex["id"], "answers": ex["answers"]} for ex in references]

    results = squad_metric.compute(predictions=formatted_predictions, references=formatted_references)
    return {"exact_match": results["exact_match"], "f1": results["f1"]}

