import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_NAME = "bert-base-uncased"


def create_model(method: str, num_labels: int, method_cfg: dict | None = None) -> nn.Module:
    """Factory function to create a model with the specified PEFT method."""
    if method == "full_ft":
        return _create_full_ft(num_labels)
    elif method == "bitfit":
        return _create_bitfit(num_labels)
    elif method == "bitfit_subset":
        return _create_bitfit_subset(num_labels)
    elif method == "lora":
        return _create_lora(num_labels, method_cfg or {})
    elif method == "prompt_tuning":
        return _create_prompt_tuning(num_labels, method_cfg or {})
    else:
        raise ValueError(f"Unknown method: {method}")


def create_qa_model(method: str) -> nn.Module:
    """Factory function to create a QA model with the specified method."""
    if method == "full_ft":
        return _create_qa_full_ft()
    elif method == "bitfit":
        return _create_qa_bitfit()
    else:
        raise ValueError(f"Unknown method for QA: {method}")


def _create_full_ft(num_labels: int) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


def _create_qa_full_ft() -> PreTrainedModel:
    return AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)


def _create_qa_bitfit() -> PreTrainedModel:
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    _freeze_except_bias_and_qa_head(model)
    return model


def _freeze_except_bias_and_qa_head(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "bias" in name or "qa_outputs" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def _create_bitfit(num_labels: int) -> PreTrainedModel:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    _freeze_except_bias_and_classifier(model)
    return model


def _freeze_except_bias_and_classifier(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "bias" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def _create_bitfit_subset(num_labels: int) -> PreTrainedModel:
    """Only train query bias, intermediate dense bias, and classifier."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    _freeze_except_bitfit_subset_and_classifier(model)
    return model


def _freeze_except_bitfit_subset_and_classifier(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        elif "attention.self.query.bias" in name:
            param.requires_grad = True
        elif "intermediate.dense.bias" in name:
            param.requires_grad = True


def _create_lora(num_labels: int, method_cfg: dict) -> nn.Module:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    lora_config = LoraConfig(
        r=method_cfg.get("r", 8),
        lora_alpha=method_cfg.get("lora_alpha", 16),
        lora_dropout=method_cfg.get("lora_dropout", 0.1),
        target_modules=list(method_cfg.get("target_modules", ["query", "value"])),
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(base_model, lora_config)
    return model


def _create_prompt_tuning(num_labels: int, method_cfg: dict) -> nn.Module:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    num_virtual_tokens = method_cfg.get("num_virtual_tokens", 20)
    return SoftPromptModel(base_model, num_virtual_tokens)


class SoftPromptModel(nn.Module):
    """Soft prompt tuning: prepend learned embeddings to input, freeze the rest of BERT."""

    def __init__(self, base_model: PreTrainedModel, num_virtual_tokens: int):
        super().__init__()
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        self.config = base_model.config

        hidden_size = base_model.config.hidden_size
        self.soft_prompt = nn.Embedding(num_virtual_tokens, hidden_size)
        nn.init.normal_(self.soft_prompt.weight, mean=0.0, std=0.02)

        self._freeze_base_except_classifier()

    def _freeze_base_except_classifier(self) -> None:
        for name, param in self.base_model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        input_embeds = self.base_model.bert.embeddings(input_ids)

        prompt_embeds = self.soft_prompt.weight.unsqueeze(0).expand(batch_size, -1, -1)
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_virtual_tokens, device=device)
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        outputs = self.base_model.bert(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )

        pooled_output = outputs.pooler_output
        logits = self.base_model.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states
        )

