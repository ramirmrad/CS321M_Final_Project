# Inference Method as a Measurement Facet: Benchmark Validity Under Speculative Decoding

**CS 321M Final Project — Rami Ratl Mrad — Stanford University**

This repository contains all code, pre-collected logit caches, result CSVs, and figures for the paper *Inference Method as a Measurement Facet: Benchmark Validity Under Speculative Decoding*. The project studies whether speculative decoding introduces construct-irrelevant measurement variance in LLM benchmark scores, using a logit-replay design across 500 MMLU items, 500 HellaSwag items, and 200 GSM8K items for two draft–target model pairs.

---

## Repository Structure

```
.
├── code_submission.ipynb          # Main analysis notebook (end-to-end reproduction)
├── writeup.tex              # LaTeX source for the final manuscript
├── proposal.tex             # LaTeX source for the pre-analysis plan
├── results/                 # All output CSVs and figures (pre-computed)
│   ├── a1_summary_final.csv          # Accuracy by condition (Analysis 1)
│   ├── a1_mcnemar_final.csv          # McNemar test results with Bonferroni correction
│   ├── a2_gtheory_final.csv          # G-theory variance components (Analysis 2)
│   ├── a4_construct_final.csv        # Construct-level accuracy changes (Analysis 3)
│   ├── condition_impact.csv          # Per-condition accuracy, flip rate, token change rate
│   ├── mechanism_diagnostics.csv     # Per-item logit statistics (entropy, JS div, etc.)
│   ├── mechanism_flip_associations.csv # Point-biserial correlations: predictors vs flips
│   ├── score_flips.csv               # Item-level score flip indicators
│   ├── accuracy_gaps.csv             # Target vs draft accuracy gaps
│   ├── temp_correlations.csv         # Temperature–accuracy Pearson correlations
│   ├── temp_gtheory.csv              # G-theory for the temperature facet
│   ├── acceptance_rate_data.csv      # Regression inputs (Analysis 4)
│   ├── reg_model1.csv / reg_model2.csv / reg_model3.csv  # OLS regression outputs
│   ├── fig_spec_threshold_validity.pdf       # Figure: spec threshold effects
│   ├── fig_spec_vs_temperature_positive_control.pdf  # Figure: spec vs temperature
│   ├── fig_mechanism_flip_predictors.pdf     # Figure: mechanism heatmaps
│   ├── fig_temp_accuracy_curve.pdf           # Figure: temperature accuracy curves
│   ├── fig_temp_gtheory.pdf                  # Figure: temperature G-theory
│   ├── fig1_accuracy_comparison.pdf          # Figure: accuracy by condition
│   ├── fig2_gtheory_comparison.pdf           # Figure: G-theory bar chart
│   └── fig4_construct_comparison.pdf         # Figure: construct stability
└── cache/
    └── logits/              # Pre-collected .npz logit caches (one per model × benchmark)
```

---

## Environment Setup

**Python version:** 3.10+

Install all dependencies with:

```bash
pip install transformers accelerate datasets torch numpy scipy pandas \
            matplotlib seaborn tqdm statsmodels scikit-learn
```

Or, to install from within the notebook, run **§ 0** (the first code cell), which calls `pip install` automatically.

**No GPU is required to reproduce the analysis.** GPU is only needed if you wish to re-collect the logit caches from scratch (see Stage 1 below).

---

## Reproducing the Results

Reproduction has two stages. **Stage 2 alone is sufficient to reproduce all reported results and figures** using the pre-collected logit caches provided in `cache/logits/`.

### Stage 1 — Logit Collection (optional, GPU required)

This stage generates the `.npz` logit caches by running forward passes through the draft and target models on each benchmark item.

**Requirements:**
- NVIDIA GPU (the paper used an L4)
- HuggingFace access token for the Llama models (`meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`)

**Steps:**
1. Log in to HuggingFace: `huggingface-cli login`
2. Open `code_submission.ipynb`
3. In **§ 1 (Configuration)**, set `RUN_LOGIT_COLLECTION = True`
4. Run all cells in order

Logit collection writes `.npz` files to `cache/logits/`. Expected runtime: **~2–4 hours** per model pair on an L4 GPU across all three benchmarks.

### Stage 2 — Analysis (no GPU required)

This stage replays the saved logits, runs all decoding simulations, and regenerates all CSVs and figures.

**Steps:**
1. Open `code_submission.ipynb`
2. Ensure `RUN_LOGIT_COLLECTION = False` in **§ 1** (this is the default)
3. Run all cells in order from top to bottom

All outputs are written to `results/`. **Expected runtime: ~5–15 minutes** on a standard laptop CPU.

### Quickstart (analysis only, small-scale check)

To verify the pipeline runs correctly without waiting for the full analysis, set the following in **§ 1**:

```python
N_GSM8K = 20    # reduce from 200
N_FULL  = 50    # reduce from 500
```

Then run all cells. This uses only 20 GSM8K and 50 MMLU/HellaSwag items and completes in under a minute. Note that results will differ from the paper; restore the original values to reproduce reported numbers.

---

## Which Cells Produce Which Outputs

| Notebook section | Key outputs |
|---|---|
| § 5 (Simulation) | `raw_results.csv`, `draft_standalone.csv`, `accuracy_gaps.csv` |
| § 6 (Analysis Dataset) | `analysis_dataset.csv`, `condition_impact.csv`, `a1_summary_final.csv` |
| § 7 (Validity Diagnostics) | `mechanism_diagnostics.csv`, `score_flips.csv`, `mechanism_flip_associations.csv`, `fig_spec_threshold_validity.pdf`, `fig_spec_vs_temperature_positive_control.pdf`, `fig_mechanism_flip_predictors.pdf` |
| § 8 (Temperature) | `temp_correlations.csv`, `temp_gtheory.csv`, `fig_temp_accuracy_curve.pdf`, `fig_temp_gtheory.pdf` |
| § 9 (Analysis 1) | `a1_mcnemar_final.csv`, `fig1_accuracy_comparison.pdf` |
| § 10 (Analysis 2) | `a2_gtheory_final.csv`, `fig2_gtheory_comparison.pdf` |
| § 11 (Analysis 3) | `a4_construct_final.csv`, `fig4_construct_comparison.pdf` |
| § 12 (Analysis 4) | `acceptance_rate_data.csv`, `reg_model1.csv`, `reg_model2.csv`, `fig6_acceptance_moderation.pdf` |

---

## Datasets

All benchmark data is loaded automatically from HuggingFace Datasets at runtime:

| Benchmark | HuggingFace ID | Split | Items used |
|---|---|---|---|
| MMLU | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) | `test` | 500 |
| HellaSwag | [`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag) | `validation` | 500 |
| GSM8K | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | `test` | 200 |

No manual data download is required.

---

## Models

| Role | Model | HuggingFace ID |
|---|---|---|
| Draft | Qwen2.5-0.5B | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Target | Qwen2.5-7B | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Draft | Llama-3.2-1B | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| Target | Llama-3.1-8B | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |

Models are downloaded automatically by HuggingFace Transformers during logit collection. The Llama models require accepting the Meta license on HuggingFace and logging in with `huggingface-cli login`.

---

## Computational Requirements

| Stage | Hardware | Estimated Runtime |
|---|---|---|
| Logit collection (Stage 1) | GPU recommended (paper: NVIDIA L4, 24 GB VRAM) | ~2–4 hours per model pair |
| Analysis & figures (Stage 2) | CPU only | ~5–15 minutes |

The `cache/logits/` directory contains pre-collected `.npz` files so that Stage 2 can be run without GPU access.

---

## Reproducibility Notes

- All random seeds are fixed: `SEED = 42` (Python, NumPy, PyTorch).
- All stochastic processes (item shuffling, token sampling, simulation RNGs) use `SEED`-derived seeds documented in § 1 of the notebook.
- The `results/` directory contains pre-computed CSVs and figures. Running Stage 2 will overwrite them with identical values.
- Floating-point results may differ by ±1 ULP across hardware/OS due to non-associative floating-point arithmetic, but all reported statistics (accuracies, p-values, correlation coefficients) should match to at least 3 significant figures.
