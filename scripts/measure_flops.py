"""Measure actual FLOPs for each PEFT method on a GLUE sample."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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




def measure_flops_with_profiler(model: torch.nn.Module, inputs: dict) -> tuple[int, int]:
    """Measure forward and backward FLOPs using torch.profiler."""
    model.train()
    model.zero_grad()
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = torch.zeros(input_ids.shape[0], dtype=torch.long)
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        # forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    
    forward_flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)
    
    model.zero_grad()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        # forward + backward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
    
    total_flops = sum(e.flops for e in prof.key_averages() if e.flops > 0)
    backward_flops = total_flops - forward_flops
    
    return forward_flops, backward_flops


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


def create_flops_comparison_graph(df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar chart comparing FLOPs across methods."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df["method"].tolist()
    forward = df["forward_gflops"].tolist()
    backward = df["backward_gflops"].tolist()
    
    x = range(len(methods))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], forward, width, label="Forward", color="#2ecc71")
    bars2 = ax.bar([i + width/2 for i in x], backward, width, label="Backward", color="#e74c3c")
    
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("GFLOPs", fontsize=12)
    ax.set_title("FLOPs Comparison per Sample (Seq Length = 128)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend()
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / "flops_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "flops_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FLOPs comparison graph to {output_dir}/flops_comparison.{{png,pdf}}")


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
        
        # measure with profiler (actual)
        fwd_prof, bwd_prof = measure_flops_with_profiler(model, inputs)
        
        result = {
            "method": METHOD_DISPLAY[method],
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(trainable_ratio * 100, 4),
            "forward_gflops": round(fwd_prof / 1e9, 2),
            "backward_gflops": round(bwd_prof / 1e9, 2),
            "total_gflops": round((fwd_prof + bwd_prof) / 1e9, 2),
        }
        results.append(result)
        
        print(f"  Trainable: {trainable:,} / {total:,} ({trainable_ratio*100:.3f}%)")
        print(f"  Forward (profiler):  {fwd_prof/1e9:.2f} GFLOPs")
        print(f"  Backward (profiler): {bwd_prof/1e9:.2f} GFLOPs")
        print(f"  Total:    {(fwd_prof + bwd_prof)/1e9:.2f} GFLOPs")
        print()
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "flops_measured.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}/flops_measured.json")
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "flops_table.csv", index=False)
    print(f"Table saved to {output_dir}/flops_table.csv")
    
    create_flops_comparison_graph(df, output_dir)
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<16} {'Forward (GFLOPs)':<18} {'Backward (GFLOPs)':<18} {'Total (GFLOPs)':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['method']:<16} {r['forward_gflops']:<18.2f} {r['backward_gflops']:<18.2f} {r['total_gflops']:<15.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

