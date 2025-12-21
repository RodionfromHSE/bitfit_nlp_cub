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
    font-size: 0.62em;
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

## Parameter-Efficient Fine-Tuning on BERT

**Base Model:** BERT-base-uncased (109M parameters)  
**Tasks:** GLUE (SST-2, MRPC, RTE) + SQuAD v1.1

**Presenters:** _Name A_ & _Name B_

<!--
Speaker A:
- Quick framing: we compare PEFT methods vs full fine-tuning on classification + QA.
- Set expectations: “same pipeline, different trainable parameters”.
-->

---

# Why PEFT?

Full fine-tuning (FT) updates **all 109M parameters** → expensive to train, store, and iterate.

**PEFT (Parameter-Efficient Fine-Tuning)** aims to:
- Train **≤0.27%** of parameters (our methods)
- Reduce peak training memory (≈ **0.5×** in our setup)
- Keep accuracy close to Full FT

<!--
Speaker A:
- Motivate with practical constraints: GPU memory + faster experimentation.
- Don’t promise “always better”; promise a trade-off study.
-->

---

# Research Questions (RQ)

1. **GLUE:** How close can PEFT get to Full FT?
2. **Trade-off:** Which method gives the best performance per trainable parameter?
3. **SQuAD low-data:** When does **BitFit** beat Full FT?

<!--
Speaker A:
- These will be answered explicitly again on the conclusions slide.
-->

---

# What We Did

- Implemented **BitFit**, **BitFit Subset**, **LoRA (r=8)**, **Prompt Tuning (20 tokens)** + **Full FT**
- Used one shared training pipeline (same tokenizer, datasets, metrics)
- Measured: **quality** (Acc/F1/EM), **trainable %**, **FLOPs**, **peak memory**

<!--
Speaker A:
- Emphasize “single codebase / controlled comparison”.
-->

---

# Roadmap

1. **Setup:** model, datasets, metrics, training protocol  
2. **Methods:** what each method changes + how implemented in code  
3. **Results:** GLUE → then SQuAD low-data  
4. **Takeaways:** decision rule + limitations

<!--
Speaker A:
- Tell them what to listen for: results + why we trust them (protocol).
-->

---

# Setup: Base Model & Tasks

**Model:** `bert-base-uncased` (Transformers)

**Tasks**
- **GLUE**: SST-2 (sentiment), MRPC (paraphrase), RTE (entailment)
- **SQuAD v1.1**: extractive question answering (predict answer span)

<!--
Speaker A:
- This is “one backbone, multiple tasks”.
- Mention: classification uses `AutoModelForSequenceClassification`, QA uses `AutoModelForQuestionAnswering`.
-->

---

# Setup: Datasets (sizes)

| Dataset | Split | Size |
|---|---:|---:|
| SST-2 | train / val | 67,349 / 872 |
| MRPC | train / val | 3,668 / 408 |
| RTE | train / val | 2,490 / 277 |
| SQuAD v1.1 | train / val | 87,599 / 10,570 |

**SQuAD size sweep (train):** 500, 1K, 2K, 3K, 5K, 7K, 10K, 15K, 20K, **88,524 (full\*)**

\*Full run corresponds to **88,524 tokenized training windows** after QA sliding-window tokenization.

<!--
Speaker A:
- Key story: SST-2 is large; MRPC/RTE are small → harder to match with PEFT.
- Clarify “88,524” is the tokenized training set size used in our logs/plots (not raw examples).
-->

---

# Setup: Metrics

| Task Type | Primary Metric | Secondary Metric |
|---|---|---|
| SST-2 | Accuracy | - |
| MRPC | F1 Score | Accuracy |
| RTE | Accuracy | - |
| SQuAD | Exact Match (EM) | F1 Score |

**Notes:** Higher is better; Acc = accuracy, EM = exact match

<!--
Speaker A:
- Explain MRPC: we report F1 primarily.
- Explain SQuAD: EM is strict; F1 is softer overlap.
-->

---

# Setup: Training Protocol (fair comparison)

**Shared**
- Same tokenizer / preprocessing pipeline
- Mixed precision enabled when GPU is available (fp16)

**GLUE**
- **3 epochs**, batch size **16**, seed **42**
- Sequence length: **128**

**SQuAD**
- Batch size **12**, seed **42**
- Max length **384**, doc stride **128**
- **Step-based** schedule: small datasets get enough optimizer updates (min **2000** steps)

<!--
Speaker A:
- This slide is about comparability: same backbone, same data pipeline, controlled training.
- Mention: method-specific learning rates are set via configs (can show in appendix if asked).
-->

---

# Methods Overview (with parameter counts)

| Method | Trainable Params | % of Total | What is trained |
|--------|------------------:|-----------:|---|
| Full FT | 109,483,778 | 100% | All parameters |
| BitFit | 104,450 | **0.095%** | All bias terms + classifier |
| BitFit Subset | 47,618 | **0.043%** | Query bias + intermediate dense bias + classifier |
| LoRA (r=8) | 296,450 | **0.27%** | Low-rank adapters on Q,V |
| Prompt Tuning | 16,898 | **0.015%** | 20 virtual tokens + classifier |

<!--
Speaker A:
- This table is the anchor: everything else explains “what do these trainable params mean?”
-->

---

# How We Implement PEFT (in our code)

All methods are implemented by **freezing/unfreezing parameters** (via `requires_grad`) or adding small trainable modules.

**Summary (sequence classification)**
- **Full FT:** nothing frozen
- **BitFit:** train parameters whose name contains `bias` or `classifier`
- **BitFit Subset:** train `attention.self.query.bias`, `intermediate.dense.bias`, `classifier`
- **LoRA:** inject adapters into attention `query` and `value` (rank **r=8**)
- **Prompt Tuning:** learn **20** prompt embeddings and prepend to input embeddings

<!--
Speaker A:
- Point to `src/models.py` as the single implementation entrypoint.
- Keep it conceptual; details per method come next.
-->

---

# Full Fine-Tuning (baseline)

**What changes:** all weights are updated.

**Why we need it:** best-case performance, but highest cost.

Trainable parameters: **109,483,778 (100%)**

<!--
Speaker A:
- Baseline for quality; everything else is “how much do we lose (or gain) when training less?”
-->

---

# BitFit (bias-only fine-tuning)

**Core idea:** fine-tune only bias terms (and the task head)

For a linear layer: **y = Wx + b** → **y' = Wx + (b + Δb)**

**Trainable params:** **104,450 (0.095%)**

<!--
Speaker A:
- Explain intuition: bias shifts can change activation patterns (especially with GELU).
- Implementation detail (if asked): `requires_grad=True` for params with “bias” or “classifier”.
-->

---

# BitFit Subset (even fewer biases)

**What we train (subset):**
- Attention **query bias**
- Feed-forward **intermediate dense bias**
- Classifier head

**Trainable params:** **47,618 (0.043%)**

**Why:** push parameter efficiency further and test which biases matter most.

<!--
Speaker A:
- Mention this is a targeted subset: Q bias affects attention, intermediate bias affects MLP gating.
-->

---

# LoRA (Low-Rank Adaptation)

**Idea:** keep base weights frozen, learn low-rank updates  
\(\Delta W \approx A B\) where rank(\(\Delta W\)) = **r**

**Our setup**
- **r = 8**, `alpha = 16`, dropout `0.1`
- Adapters on attention **Q** and **V**

**Trainable params:** **296,450 (0.27%)**

<!--
Speaker A:
- LoRA is “small weight updates in key submodules” (Q/V).
- Implementation is via `peft.get_peft_model(...)` with `LoraConfig`.
-->

---

# Prompt Tuning (soft prompts)

**Idea:** learn a small set of embeddings that steer the frozen model.

**Our setup**
- **20 virtual tokens** (learned embeddings)
- Base model frozen except classifier

**Trainable params:** **16,898 (0.015%)**

<!--
Speaker A:
- In code we prepend learned embeddings to the input embedding sequence and extend the attention mask.
- It’s extremely parameter-efficient, but capacity is limited.
-->

---

# Efficiency Metrics (memory + compute)

| Method | Memory (GB) | Relative to Full FT |
|--------|-------------|---------------------|
| Full FT | ~2.1-2.2 | 1.0x |
| BitFit | ~1.0-1.1 | **~0.5x** |
| BitFit Subset | ~1.0 | **~0.5x** |
| LoRA | ~0.9-1.2 | **~0.5x** |
| Prompt Tuning | ~0.9-1.2 | **~0.5x** |

In our setup, PEFT reduces peak training memory by **~50%** vs Full FT.

<!--
Speaker A:
- Use this as the “systems” motivation before we show accuracy.
- Handoff to Speaker B for results.
-->

---

# Part 1 — GLUE Experiments

**Goal:** compare Full FT vs PEFT on 3 classification tasks.

<!--
Speaker B:
- Quick transition: “now that methods are clear, here are the results”.
-->

---

# GLUE: Experiment Setup

- **Tasks:** SST-2, MRPC, RTE
- **Methods:** Full FT, BitFit, BitFit Subset, LoRA, Prompt Tuning
- **Training:** 3 epochs, batch size 16, seed 42

<!--
Speaker B:
- Remind: same base model + same pipeline; only trainable parameters change.
-->

---

# GLUE: Results (final metrics)

| Method | SST-2 (Acc) | MRPC (F1) | RTE (Acc) | Avg | Trainable % |
|--------|-------------|-----------|-----------|-----|-------------|
| **Full FT** | **92.09%** | **90.02%** | **67.51%** | **83.21%** | 100% |
| LoRA | 92.20% | 88.32% | 61.01% | 80.51% | 0.27% |
| BitFit | 90.60% | 88.01% | 61.37% | 79.99% | 0.095% |
| BitFit Subset | 90.25% | 81.94% | 57.04% | 76.41% | 0.043% |
| Prompt Tuning | 88.19% | 81.79% | 57.40% | 75.79% | 0.015% |

<!--
Speaker B:
- Point out: Full FT best Avg; LoRA slightly best on SST-2; MRPC/RTE expose gaps.
-->

---

# GLUE: Key Takeaways

1. **Full FT is best overall (Avg 83.21)**, but **LoRA matches it on SST-2** (92.20 vs 92.09)
2. **LoRA is the strongest PEFT here** — Avg 80.51 with 0.27% trainable params
3. **BitFit is close behind** — Avg 79.99 with 0.095% trainable params
4. **Harder/smaller tasks amplify gaps** (RTE shows the biggest separation)

<!--
Speaker B:
- Keep it tight: 30–45 seconds. Move quickly to trade-off plots.
-->

---

# GLUE: Performance vs Trainable Parameters

![center width:900px](outputs/parameter_efficiency.png)

<!--
Speaker B:
- Tell a story: “how much performance per parameter do we get?”
- Emphasize the Pareto frontier (LoRA/BitFit vs Prompt/Subsets).
-->

---

# GLUE: FLOPs Comparison

![center width:900px](outputs/flops_comparison.png)

FLOPs = floating-point operations (lower is better)

<!--
Speaker B:
- Mention: measured per sample (seq length 128) to compare relative compute.
-->

---

# GLUE: Visual Summary

![center width:1000px](outputs/part1_results.png)

<!--
Speaker B:
- One-sentence wrap: “LoRA best PEFT overall; BitFit very strong for its size.”
- Transition: “Now, what happens when data is limited on QA?”
-->

---

# Part 2 — SQuAD Low-Data Experiments

**Goal:** test whether BitFit acts as regularization in extreme low-data regimes.

<!--
Speaker B:
- Set expectation: the main story is the crossover as data increases.
-->

---

# SQuAD: Hypothesis

> **BitFit can generalize better than Full FT in extreme low-data regimes, but Full FT should win once data is sufficient.**

**Rationale:** With **0.095% vs 100%** trainable parameters, BitFit should:
- Be less prone to overfitting on very small datasets
- Provide a strong regularized baseline
- Trade peak performance for efficiency as data grows

<!--
Speaker B:
- Make it falsifiable: “we expect a crossover”.
-->

---

# SQuAD: Experiment Setup

- **Task:** SQuAD v1.1 (Question Answering)
- **Methods:** Full FT vs BitFit
- **Train sizes:** 500 → 20K → **88,524 (full)**
- **Training:** step-based max steps (≥ **2000** for small), batch size **12**

<!--
Speaker B:
- Keep focus: only two methods to isolate “bias-only vs full capacity”.
-->

---

# ⚠️ Methodological Pitfall (what we fixed)

If we train with the **same epochs** for every train size:
- Small datasets get **far fewer steps**
- BitFit is **under-trained** (appears much worse than it is)

**Previous (flawed) results — too few steps:**

| Train Size | Full FT (EM) | Full FT (F1) | BitFit (EM) | BitFit (F1) | Δ F1 |
|------------|--------------|--------------|-------------|-------------|------|
| 1,000 | 20.52% | 31.71% | 7.13% | 14.72% | **-16.99%** |
| 5,000 | 56.73% | 67.75% | 38.00% | 50.90% | **-16.85%** |
| 10,000 | 66.54% | 76.71% | 50.32% | 62.64% | **-14.07%** |

<!--
Speaker B:
- Strong credibility moment: we found and corrected a confound.
- Core message: you must control for optimizer updates when varying dataset size.
-->

---

# Corrected Protocol: Step-Based Training

After switching to a **step-based** schedule, BitFit performance improves substantially at low data.

**Key point:** controlling for optimizer steps makes the comparison meaningful.

_Insert schematic (optional): “fixed epochs” vs “fixed steps” timeline._

<!--
Speaker B:
- If asked: our policy enforces a minimum step budget (2000) and scales steps for larger datasets.
-->

---

# SQuAD: Results (scaling curve)

![center width:1000px](outputs/part2_results.png)

<!--
Speaker B:
- Walk the audience through 3 regimes: very low data (BitFit wins), crossover (~2K), large data (Full FT wins).
-->

---

# SQuAD: Key Numbers (corrected)

| Train Size | Full FT (F1) | BitFit (F1) | Full FT (EM) | BitFit (EM) |
|------------|--------------|-------------|--------------|-------------|
| 500 | 45.54% | **55.31%** | 34.45% | **42.10%** |
| 1,000 | 54.08% | **59.32%** | 42.71% | **46.53%** |
| 2,000 | **63.03%** | 62.31% | **51.19%** | 49.78% |
| 3,000 | **67.78%** | 63.15% | **56.37%** | 51.14% |
| 88,524 (full) | **88.01%** | 76.77% | **80.74%** | 65.94% |

<!--
Speaker B:
- Emphasize the punchline with numbers: +9.77 F1 at 500; +5.24 F1 at 1K; then Full FT pulls ahead.
-->

---

# SQuAD: Regime Summary

| Data Regime | Winner | Gap |
|-------------|--------|-----|
| ≤1K samples | **BitFit** | +5-10% F1 |
| ~2K samples | Tie | ~0% |
| ≥3K samples | **Full FT** | ~11% F1 |

<!--
Speaker B:
- This slide is what the audience should remember if they remember only one thing.
- Handoff back to Speaker A for wrap-up.
-->

---

# Interpretation (why the crossover happens)

**Why BitFit can win at very low data**
- Fewer trainable params → stronger implicit regularization
- Bias shifts can re-route existing pretrained features without overfitting

**Why Full FT wins with more data**
- More capacity to adapt internal representations
- With enough data, that extra capacity is beneficial

<!--
Speaker A:
- Keep it tied to what we saw (crossover), not generic PEFT hype.
-->

---

# Practical Decision Rule (heuristic)

- **Need best overall quality + enough data:** Full FT
- **Constrained training but want strong GLUE performance:** LoRA (r=8)
- **Extreme low-data (≤1K) QA:** BitFit is a strong baseline
- **If you’re unsure:** start with BitFit/LoRA, then Full FT if you can afford it

<!--
Speaker A:
- Call it a heuristic because: single seed + limited hyperparameter tuning.
-->

---

# Limitations / Threats to Validity

- Single seed (**42**) → variance not measured
- Minimal per-method hyperparameter tuning
- LoRA rank fixed (**r=8**) and prompt length fixed (**20**)
- Only BERT-base and a small set of tasks

<!--
Speaker A:
- Say this confidently; it increases trust.
-->

---

# Next Steps

- Run multiple seeds and report confidence intervals
- Tune LoRA ranks / BitFit subsets per task
- Expand to more tasks and larger backbones
- Add wall-clock throughput + cost analysis

<!--
Speaker A:
- Keep it short; this is a “future work” slide, not a new section.
-->

---

# Conclusions (answers to RQs)

1. **GLUE:** PEFT is competitive — LoRA/BitFit are within ~3 Avg points of Full FT with ≤0.27% trainable params  
2. **Trade-off:** **LoRA** is the best overall PEFT in our GLUE setup; **BitFit** is close with fewer params  
3. **SQuAD low-data:** **BitFit wins at ≤1K**, ties ~2K, and Full FT wins beyond ~3K with ~11% F1 gap

<!--
Speaker A:
- Mirror the RQ slide: answer in the same order.
-->

---

# Questions?

<!--
Both:
- Speaker A takes implementation/method questions.
- Speaker B takes results/plots questions.
-->

---

# Appendix: SQuAD Results Table (F1, corrected)

| Train Size | Full FT (F1) | BitFit (F1) | Δ F1 |
|------------|--------------|-------------|------|
| 500 | 45.54% | 55.31% | **+9.77%** |
| 1,000 | 54.08% | 59.32% | **+5.24%** |
| 2,000 | 63.03% | 62.31% | **-0.72%** |
| 3,000 | 67.78% | 63.15% | **-4.63%** |
| 5,000 | 73.96% | 62.80% | **-11.16%** |
| 7,000 | 76.17% | 64.46% | **-11.71%** |
| 10,000 | 79.36% | 67.89% | **-11.47%** |
| 15,000 | 81.44% | 70.69% | **-10.75%** |
| 20,000 | 83.04% | 72.34% | **-10.70%** |
| 88,524 (full) | 88.01% | 76.77% | **-11.24%** |

<!--
Optional backup:
- If someone asks “what about 7K / 10K / 15K?”, this slide has the full sweep.
-->

---

# Appendix: SQuAD Results Table (EM, corrected)

| Train Size | Full FT (EM) | BitFit (EM) | Δ EM |
|------------|--------------|-------------|------|
| 500 | 34.45% | 42.10% | **+7.65%** |
| 1,000 | 42.71% | 46.53% | **+3.82%** |
| 2,000 | 51.19% | 49.78% | **-1.41%** |
| 3,000 | 56.37% | 51.14% | **-5.23%** |
| 5,000 | 63.37% | 50.72% | **-12.65%** |
| 7,000 | 66.20% | 52.78% | **-13.42%** |
| 10,000 | 69.36% | 56.52% | **-12.84%** |
| 15,000 | 72.10% | 59.38% | **-12.72%** |
| 20,000 | 73.71% | 61.27% | **-12.44%** |
| 88,524 (full) | 80.74% | 65.94% | **-14.80%** |

---

# Appendix: Hyperparameters (from configs)

**GLUE training:** 3 epochs, batch size 16, warmup ratio 0.1, weight decay 0.01, fp16 (GPU), seed 42  
**SQuAD training:** batch size 12, warmup ratio 0.1, weight decay 0.01, fp16 (GPU), seed 42, step-based `max_steps`

| Method | Learning rate | Key method params |
|---|---:|---|
| Full FT | 2e-5 | - |
| BitFit | 5e-4 | - |
| BitFit Subset | 5e-4 | - |
| LoRA | 2e-4 | r=8, alpha=16, dropout=0.1, target={query,value} |
| Prompt Tuning | 5e-4 | num_virtual_tokens=20 |

---

# Appendix: Implementation Snippets (src/models.py)

**BitFit (sequence classification)**
```py
param.requires_grad = ("bias" in name) or ("classifier" in name)
```

**BitFit Subset (sequence classification)**
```py
train: classifier, attention.self.query.bias, intermediate.dense.bias
```

**LoRA (sequence classification)**
```py
LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query","value"])
```

**Prompt tuning (sequence classification)**
```py
combined = concat([soft_prompt_embeds, input_embeds])
```
