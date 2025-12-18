"""Measure actual FLOPs for each PEFT method on a GLUE sample."""

import json
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis
from transformers import AutoTokenizer

from src.models import create_model

METHODS = ["full_ft", "lora", "bitfit", "bitfit_subset", "prompt_tuning"]
METHOD_DISPLAY = {
    "full_ft": "Full FT",
    "lora": "LoRA",
    "bitfit": "BitFit",
    "bitfit_subset": "BitFit Subset",
    "prompt_tuning": "Prompt Tuning",
}
METHOD_CONFIGS = {
    "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "target_modules": ["query", "value"]},
    "prompt_tuning": {"num_virtual_tokens": 20},
}


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def measure_forward_flops(model: torch.nn.Module, inputs: dict) -> int:
    """Measure forward pass FLOPs using fvcore."""
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, (inputs["input_ids"], inputs["attention_mask"]))
        return flops.total()


def measure_backward_flops_estimate(forward_flops: int, trainable_ratio: float) -> int:
    """Estimate backward pass FLOPs based on trainable parameter ratio.
    
    For full backward, FLOPs ≈ 2× forward (gradient computation + accumulation).
    For partial backward, scales with trainable params.
    """
    full_backward = forward_flops * 2
    return int(full_backward * trainable_ratio)


def create_sample_input(tokenizer, seq_length: int = 128) -> dict:
    """Create a sample input for FLOP measurement."""
    text = "This is a sample sentence for measuring FLOPs. " * 10
    inputs = tokenizer(
        text,
        max_length=seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def main():
    print("Measuring FLOPs for each PEFT method...\n")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = create_sample_input(tokenizer, seq_length=128)
    
    results = []
    
    for method in METHODS:
        print(f"Processing {METHOD_DISPLAY[method]}...")
        
        cfg = METHOD_CONFIGS.get(method, {})
        model = create_model(method, num_labels=2, method_cfg=cfg)
        
        trainable, total = count_trainable_params(model)
        trainable_ratio = trainable / total
        
        forward_flops = measure_forward_flops(model, inputs)
        backward_flops = measure_backward_flops_estimate(forward_flops, trainable_ratio)
        total_flops = forward_flops + backward_flops
        
        result = {
            "method": METHOD_DISPLAY[method],
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(trainable_ratio * 100, 4),
            "forward_gflops": round(forward_flops / 1e9, 2),
            "backward_gflops_est": round(backward_flops / 1e9, 2),
            "total_gflops": round(total_flops / 1e9, 2),
        }
        results.append(result)
        
        print(f"  Trainable: {trainable:,} / {total:,} ({trainable_ratio*100:.3f}%)")
        print(f"  Forward:  {forward_flops/1e9:.2f} GFLOPs")
        print(f"  Backward: {backward_flops/1e9:.2f} GFLOPs (estimated)")
        print(f"  Total:    {total_flops/1e9:.2f} GFLOPs")
        print()
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    output_path = Path("outputs/flops_measured.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<16} {'Forward (GFLOPs)':<18} {'Backward (GFLOPs)':<18} {'Total (GFLOPs)':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['method']:<16} {r['forward_gflops']:<18.2f} {r['backward_gflops_est']:<18.2f} {r['total_gflops']:<15.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

