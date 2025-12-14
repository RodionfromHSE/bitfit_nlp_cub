from functools import partial

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


def get_tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_glue_dataset(task_name: str) -> DatasetDict:
    return load_dataset("glue", task_name)


def tokenize_glue(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    sentence1_key: str,
    sentence2_key: str | None,
) -> dict:
    if sentence2_key is None:
        return tokenizer(
            examples[sentence1_key],
            truncation=True,
            max_length=MAX_LENGTH,
        )
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def prepare_glue_dataset(
    task_name: str,
    sentence1_key: str,
    sentence2_key: str | None,
    tokenizer: PreTrainedTokenizer | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and tokenize GLUE dataset, returns (train, validation) splits."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    dataset = load_glue_dataset(task_name)
    tokenize_fn = partial(
        tokenize_glue,
        tokenizer=tokenizer,
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key,
    )

    columns_to_remove = [c for c in dataset["train"].column_names if c != "label"]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=columns_to_remove)
    tokenized = tokenized.rename_column("label", "labels")

    return tokenized["train"], tokenized["validation"]


def get_data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)

