from functools import partial

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    PreTrainedTokenizer,
)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
MAX_LENGTH_QA = 384
DOC_STRIDE = 128


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


def load_squad_dataset() -> DatasetDict:
    return load_dataset("squad")


def prepare_squad_train(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """Tokenize SQuAD training examples with answer start/end positions."""
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=MAX_LENGTH_QA,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples["answers"][sample_idx]

        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        sequence_ids = tokenized.sequence_ids(i)
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_squad_eval(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """Tokenize SQuAD validation examples, keeping example_id and offset_mapping for evaluation."""
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=MAX_LENGTH_QA,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    tokenized["example_id"] = [examples["id"][sample_mapping[i]] for i in range(len(sample_mapping))]

    new_offset_mapping = []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        sequence_ids = tokenized.sequence_ids(i)
        new_offsets = [(o if sequence_ids[j] == 1 else None) for j, o in enumerate(offsets)]
        new_offset_mapping.append(new_offsets)
    tokenized["offset_mapping"] = new_offset_mapping

    return tokenized


def prepare_squad_dataset(
    tokenizer: PreTrainedTokenizer | None = None,
    train_size: int | None = None,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load and tokenize SQuAD dataset, returns (train, eval_tokenized, eval_raw)."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    dataset = load_squad_dataset()

    train_data = dataset["train"]
    if train_size is not None and train_size < len(train_data):
        train_data = train_data.shuffle(seed=seed).select(range(train_size))

    train_tokenized = train_data.map(
        partial(prepare_squad_train, tokenizer=tokenizer),
        batched=True,
        remove_columns=train_data.column_names,
    )

    val_data = dataset["validation"]
    val_tokenized = val_data.map(
        partial(prepare_squad_eval, tokenizer=tokenizer),
        batched=True,
        remove_columns=val_data.column_names,
    )

    return train_tokenized, val_tokenized, val_data


def get_squad_data_collator() -> DefaultDataCollator:
    return DefaultDataCollator()

