---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
color: #222222
style: |
  section {
    font-family: 'Segoe UI', Tahoma, sans-serif;
  }
  h1 {
    color: #1e3a5f;
  }
  h2 {
    color: #2d5986;
  }
  table {
    font-size: 0.65em;
    width: 100%;
  }
  th {
    background-color: #1e3a5f;
    color: white;
  }
  td {
    color: #222222;
  }
  strong {
    color: #c0392b;
  }
  blockquote {
    border-left: 4px solid #2d5986;
    background-color: #f0f4f8;
    padding: 10px 20px;
  }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
---

# BitFit & PEFT Experiments

## Comparing Parameter-Efficient Fine-Tuning Methods

**Base Model:** BERT-base-uncased (109M parameters)  
**Tasks:** GLUE (SST-2, MRPC, RTE) & SQuAD v1.1

---

# Agenda

1. **Part 0:** Performance Indicators & Metrics
2. **Part I:** GLUE Experiments
3. **Part II:** SQuAD Data-Size Experiments
4. **Conclusions**

---

# Part 0: Performance Indicators

## Metrics Used

| Task Type | Primary Metric | Secondary Metric |
|-----------|---------------|------------------|
| SST-2 (Sentiment) | Accuracy | - |
| MRPC (Paraphrase) | F1 Score | Accuracy |
| RTE (Entailment) | Accuracy | - |
| SQuAD (QA) | Exact Match (EM) | F1 Score |

---

# Methods Overview

| Method | Trainable Params | % of Total | Description |
|--------|-----------------|------------|-------------|
| Full FT | 109,483,778 | 100% | All parameters |
| BitFit | 104,450 | **0.095%** | All bias terms + classifier |
| BitFit Subset | 47,618 | **0.043%** | Query bias + intermediate dense bias + classifier |
| LoRA (r=8) | 296,450 | **0.27%** | Low-rank adapters on Q,V |
| Prompt Tuning | 16,898 | **0.015%** | 20 virtual tokens + classifier |

---

# Memory Efficiency

| Method | Memory (GB) | Relative to Full FT |
|--------|-------------|---------------------|
| Full FT | ~2.1-2.2 | 1.0x |
| BitFit | ~1.0-1.1 | **~0.5x** |
| BitFit Subset | ~1.0 | **~0.5x** |
| LoRA | ~0.9-1.2 | **~0.5x** |
| Prompt Tuning | ~0.9-1.2 | **~0.5x** |

All PEFT methods achieve **~50% memory reduction** compared to Full FT

---

# Part I: GLUE Experiments

## Experiment Setup

- **Tasks:** SST-2, MRPC, RTE
- **Methods:** Full FT, BitFit, BitFit Subset, LoRA, Prompt Tuning
- **Training:** 3 epochs, batch size 16, seed 42

---

# Part I: Results

| Method | SST-2 (Acc) | MRPC (F1) | RTE (Acc) | Avg | Trainable % |
|--------|-------------|-----------|-----------|-----|-------------|
| **Full FT** | **92.09%** | **90.02%** | **67.51%** | **83.21%** | 100% |
| LoRA | 92.20% | 88.32% | 61.01% | 80.51% | 0.27% |
| BitFit | 90.60% | 88.01% | 61.37% | 79.99% | 0.095% |
| BitFit Subset | 90.25% | 81.94% | 57.04% | 76.41% | 0.043% |
| Prompt Tuning | 88.19% | 81.79% | 57.40% | 75.79% | 0.015% |

---

# Part I: Key Observations

1. **Full FT dominates** across all tasks (as expected)

2. **LoRA is competitive** — matches/exceeds Full FT on SST-2

3. **BitFit performs well** — ~96-98% of Full FT with only 0.095% params

4. **Task complexity matters:**
   - SST-2 (simple): All methods ~88-92%
   - MRPC (medium): Larger gap appears
   - RTE (hard/small): Biggest gap — 67.5% vs ~57-61%

---

# Part I: Parameter Efficiency

![center width:900px](outputs/parameter_efficiency.png)

---

# Part I: FLOPs Comparison

![center width:900px](outputs/flops_comparison.png)

---

# Part I: Results Visualization

![center width:1000px](outputs/part1_results.png)

---

# Part II: SQuAD Data-Size Experiments

## Hypothesis

> **BitFit should avoid overparametrization and perform better or on par with Full FT when training data is limited.**

**Rationale:** With 0.095% vs 100% trainable parameters, BitFit should:
- Be less prone to overfitting on small datasets
- Achieve better generalization with limited data
- Eventually converge to similar performance as data increases

---

# Part II: Experiment Setup

- **Task:** SQuAD v1.1 (Question Answering)
- **Methods:** Full FT vs BitFit
- **Train Sizes:** 500, 1K, 2K, 3K, 5K, 7K, 10K, 15K, 20K, 88.5K (full)
- **Training:** Fixed number of steps (2000 for small datasets, proportionally more for larger), batch size 12

**Key:** Equal training steps ensures proper convergence across all dataset sizes

---

# Part II: ⚠️ Note on Previous Results

In our initial experiments, we made a **methodological error**: we used the same number of epochs for all dataset sizes. Smaller datasets saw far fewer training steps → poor convergence for BitFit.

**Previous (flawed) results — too few steps for small datasets:**

| Train Size | Full FT (EM) | Full FT (F1) | BitFit (EM) | BitFit (F1) | Δ F1 |
|------------|--------------|--------------|-------------|-------------|------|
| 1,000 | 20.52% | 31.71% | 7.13% | 14.72% | **-16.99%** |
| 5,000 | 56.73% | 67.75% | 38.00% | 50.90% | **-16.85%** |
| 10,000 | 66.54% | 76.71% | 50.32% | 62.64% | **-14.07%** |

---

# Part II: Fixed Methodology

After fixing the training steps (equal steps across all sizes), BitFit performance improved significantly on small datasets.

**Key insight:** The flawed methodology unfairly penalized BitFit because:
- BitFit needs more iterations to converge (fewer params = smaller updates)
- With fixed epochs, small datasets = very few steps
- BitFit couldn't converge properly

---

# Part II: Results (F1 Score) — Corrected

| Train Size | Full FT (F1) | BitFit (F1) | Δ F1 |
|------------|--------------|-------------|------|
| 500 | 45.54% | 55.31% | **+9.77%** |
| 1,000 | 54.08% | 59.32% | **+5.24%** |
| 2,000 | 63.03% | 62.31% | -0.72% |
| 3,000 | 67.78% | 63.15% | -4.63% |
| 5,000 | 73.96% | 62.80% | -11.16% |
| 7,000 | 76.17% | 64.46% | -11.71% |
| 10,000 | 79.36% | 67.89% | -11.47% |
| 15,000 | 81.44% | 70.69% | -10.75% |
| 20,000 | 83.04% | 72.34% | -10.70% |
| 88,524 (full) | 88.01% | 76.77% | -11.24% |

---

# Part II: Results (Exact Match) — Corrected

| Train Size | Full FT (EM) | BitFit (EM) | Δ EM |
|------------|--------------|-------------|------|
| 500 | 34.45% | 42.10% | **+7.65%** |
| 1,000 | 42.71% | 46.53% | **+3.82%** |
| 2,000 | 51.19% | 49.78% | -1.41% |
| 3,000 | 56.37% | 51.14% | -5.23% |
| 5,000 | 63.37% | 50.72% | -12.65% |
| 7,000 | 66.20% | 52.78% | -13.42% |
| 10,000 | 69.36% | 56.52% | -12.84% |
| 15,000 | 72.10% | 59.38% | -12.72% |
| 20,000 | 73.71% | 61.27% | -12.44% |
| 88,524 (full) | 80.74% | 65.94% | -14.80% |

---

# Part II: Results Visualization

![center width:1000px](outputs/part2_results.png)

---

# Part II: Key Finding

## BitFit Outperforms Full FT at Very Low Data!

**The hypothesis is PARTIALLY confirmed:**

| Data Regime | Winner | Gap |
|-------------|--------|-----|
| ≤1K samples | **BitFit** | +5-10% F1 |
| ~2K samples | Tie | ~0% |
| ≥3K samples | **Full FT** | ~11% F1 |

---

# Part II: Conclusion

**BitFit wins in extreme low-data regimes!** This supports the overparametrization hypothesis:

1. **At 500 samples:** BitFit F1: 55.31% vs Full FT F1: 45.54% (**+9.77%**)
2. **At 1,000 samples:** BitFit F1: 59.32% vs Full FT F1: 54.08% (**+5.24%**)
3. **Crossover point ~2K samples:** Methods converge
4. **Beyond 3K:** Full FT dominates with consistent ~11% F1 advantage

Full FT overfits on very small datasets, while BitFit's parameter constraint acts as **regularization**.

---

# Summary

## Key Takeaways

1. **PEFT methods work** — achieve ~96-98% of Full FT with <1% parameters
2. **LoRA best trade-off** — competitive performance with 0.27% params
3. **BitFit shines in low-data** — outperforms Full FT with ≤1K samples
4. **Task complexity matters** — simple tasks benefit more from PEFT
5. **Memory savings** — all PEFT methods reduce memory by ~50%

