# Cross-Lingual Alignment Without Categorical Separability:
# Temporal Safety Thresholds for Mental Health Risk Detection

EACL 2027 Anonymous Submission (ARR August 2026)

## Overview

This repository contains code for:

- **Cross-lingual semantic analysis (RQ1–RQ2)**: Wasserstein distance, ARI, Silhouette under K-Means
- **Language-controlled non-linear clustering (HDBSCAN)**: pooled (KO+EN) vs. monolingual re-analysis isolating the language confound identified in the pooled setting
- **Temporal MSW analysis (RQ3)**: Kaplan–Meier survival modeling, density-based proxy for cross-lingual temporal alignment
- **Supervised baselines**: TF-IDF, multilingual BERT variants, GPT-4, klue/RoBERTa
- **Sequential concatenation baseline**: k-turn XLM-R (k=3,5,10)
- **Temporal aggregation baseline**: majority-vote window
- **MSW-informed detection coverage**: session-level recall/F1 vs. turn budget

## Requirements

```bash
pip install transformers torch scikit-learn
pip install sentence-transformers lifelines hdbscan
pip install pandas numpy scipy matplotlib
pip install anthropic  # optional: for GPT-4 baseline
```

## Data

### Korean counseling data

Due to AI Hub Terms of Use (Korean residents only), raw and processed text data cannot be redistributed.

To reproduce results:
1. Register at https://www.aihub.or.kr
2. Download Dataset No. 71806 (1,661 sessions total)
3. Run preprocessing: `python 01_bert_training.py --data YOUR_DATA_PATH`

Note: This study used 174 full sessions (mean 286 total turns), from which 2,833 labeled utterances were extracted for classification, and 173 sessions (≥84 total turns) for temporal survival analysis.

### English counseling data

- Dataset: Mental Health Counseling Conversations
- URL: https://github.com/nbertagnolli/counsel-chat
- License: MIT

### Cross-corpus validation

- ESConv: https://github.com/thu-coai/Emotional-Support-Conversation
- AnnoMI: https://github.com/uccollab/AnnoMI

## Pipeline

Note: Steps 1–3 and 7 require the Korean AI Hub dataset (registration required; Korean residents only). Pre-computed results are available in `data/ko_dialogue_stats_endpoint.csv` for direct reproduction of MSW figures.

### Step 1: Train BERT classifiers

```bash
python 01_bert_training.py \
    --data bert_training_data.csv \
    --output_dir ./models
```

### Step 2: Extract risk scores (N=173, τ=0.90)

```bash
python 02_score_extraction.py \
    --jsonl ko_raw_all.jsonl \
    --model_dir ./models \
    --threshold 0.90 \
    --min_turns 84 \
    --output ko_turn_scores.csv
```

Threshold τ=0.90 was calibrated to reproduce original MSW estimates on a held-out subset (reproducing MSW₀.₅=12 within ±1 turn). Risk onset: P(L1|u_t) ≥ 0.5 or P(L2|u_t) ≥ 0.5 per client turn.

### Step 3: MSW survival analysis

```bash
python 03_msw_analysis.py \
    --scores ko_turn_scores.csv \
    --esconv ESConv.json \
    --annomi data/AnnoMI-simple.csv \
    --output ko_dialogue_stats_endpoint.csv
```

**Quick Start (without Korean data):**

```bash
python 03_msw_analysis.py \
    --scores data/ko_dialogue_stats_endpoint.csv \
    --esconv ESConv.json \
    --annomi data/AnnoMI-simple.csv
```

### Step 4: Translation baseline

```bash
python 04_translation_baseline.py \
    --data bert_training_data.csv \
    --n_sample 100
```

### Step 5: Supervised + Sequential baselines

Full run (requires Korean data + GPU):

```bash
python 05_supervised_baselines.py \
    --data bert_training_data.csv \
    --openai_key YOUR_KEY
```

**Note on Appendix L1-only detection table**: The L1 (severe-risk)
binary detection results reported in the appendix were obtained
by modifying `05_supervised_baselines.py`'s label definition from
`y = df['label'].values` (main risk/no-risk target, used for
Table 14) to `y = df['L1'].values.astype(int)`, then re-running
the script. This variant is not included as a separate file but
can be reproduced with this one-line change.

```
### Step 6: MSW-informed detection coverage

```bash
python 06_detection_coverage.py \
    --model_dir ./models/L2 \
    --jsonl ko_raw_all.jsonl \
    --min_client_turns 84 \
    --output detection_coverage.csv
```

### Step 7: Language-controlled HDBSCAN clustering

```bash
python 07_hdbscan_language_controlled.py \
    --data bert_training_data.csv \
    --embedding_mode cls_pooling \
    --min_cluster_sizes 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
    --output hdbscan_en_grid.csv
```

Uses CLS-token pooling (`last_hidden_state[:,0,:]`, L2-normalized) — **not** SentenceTransformer's default mean pooling — matching the embedding extraction used throughout the paper. Both pooled (KO+EN) and monolingual (EN-only, KO-only) conditions are computed; Korean-only clustering is reported as inconclusive due to insufficient cluster mass at min_cluster_size≥10.

## Expected Results

### Supervised Baselines (single-turn)

| Method | F1 | ±std |
|---|---|---|
| Random | 0.505 | 0.012 |
| Majority class | 0.690 | 0.000 |
| TF-IDF + LR | **0.793** | 0.013 |
| XLM-R + LR | 0.782 | 0.012 |
| LaBSE + LR | 0.768 | 0.013 |
| mBERT + LR | 0.763 | 0.011 |
| GPT-4 zero-shot | 0.699 | — |
| klue/RoBERTa (KO-only†) | 0.684 | 0.023 |

†Korean-only fine-tuning (n=2,833 labeled utterances from 174 sessions; 5-fold CV)
GPT-4: KO=0.614, EN=0.776 (n=200)

### Temporal Aggregation Baselines (majority vote, KO-only)

| Window | F1 | ±std |
|---|---|---|
| 8 turns | 0.664 | 0.017 |
| 16 turns | 0.656 | 0.013 |
| 32 turns | 0.615 | 0.013 |

### Sequential Concatenation Baselines (XLM-R, k-turn, KO-only)

| k | F1 | ±std |
|---|---|---|
| k=3 | 0.767 | 0.042 |
| k=5 | 0.777 | 0.022 |
| k=10 | 0.784 | 0.020 |

Prior k client turns concatenated with current turn via [SEP]; truncated to 512 tokens (right); 5-fold CV, seed=42. Neither majority-vote nor k-turn concatenation overcomes F1=0.80.

### MSW Thresholds (Korean, N=173, ≥84 total turns, τ=0.90)

| Category | MSW₀.₅ [95% CI] | MSW₀.₉ [95% CI] | N |
|---|---|---|---|
| Overall | 12 [8, 16] | 32 [24, 42] | 173 |
| Depression | 10 [6, 14] | 32 [18, 48] | 51 |
| Anxiety | 8 [4, 12] | 23 [14, 36] | 47 |
| Addiction | 14 [8, 20] | 30 [20, 44] | 52 |
| Normative | 14 [6, 24] | 42 [28, 58] | 23 |

Population-level thresholds; not clinical decision rules.

### MSW-Informed Detection Coverage

(klue/RoBERTa, N=154, ≥84 client turns, any-risk-until-t)

| Turn Budget | Recall | F1 |
|---|---|---|
| 4 | 0.526 | 0.689 |
| 8 | 0.721 | 0.838 |
| 12 (MSW₀.₅) | 0.857 | 0.923 |
| 16 | 0.935 | 0.966 |
| 24 | 0.968 | 0.983 |
| 32 (MSW₀.₉) | 0.987 | 0.993 |
| 48 | 1.000 | 1.000 |

Rapid recall gain between turns 4–12 aligns with high-hazard KM phase; saturation near turn 32 validates MSW₀.₉.

### Cross-lingual Temporal Validation

| Method | r | p | MSW₀.₉ Error (% vs. GT) |
|---|---|---|---|
| Density proxy (ours) | 0.993 | <.001 | 10% |
| Translation baseline | 0.820 | <.001 | — |
| Shuffled control | ≈0.09 | n.s. | — |
| Uniform | 0.910 | — | 72% |
| Linear | 0.648 | — | 65% |
| Random-mono | 0.999 | — | 64% |

GT = empirical Korean multi-turn MSW₀.₉ = 209 turns. Proxy: MSW₀.₉=189 (10% error); Random-mono: MSW₀.₉=344 (64% error).

### Per-Encoder Semantic Metrics (K=3, L1–L3, K-Means)

| Encoder | W (avg) | ARI | Silhouette |
|---|---|---|---|
| mBERT | 0.203 | 0.027 | 0.384† |
| XLM-R | 0.042 | 0.022 | 0.324† |
| LaBSE | 0.013 | 0.044 | 0.071 |

†High Silhouette in mBERT/XLM-R reflects language-identity clustering (KO vs. EN), not risk-level separation (confirmed by ARI≈0).

### Language-Controlled HDBSCAN (English-only, N=2,700, CLS-pooled LaBSE)

| min_cluster_size | Clusters | Noise% | ARI |
|---|---|---|---|
| 6 | 49 | 73.6% | 0.089 |
| 7 | 40 | 76.6% | 0.113 |
| 8 | 27 | 80.7% | 0.164 |
| 9 | 20 | 83.5% | 0.210 |
| **10** | **16** | **84.6%** | **0.248** |
| 11 | 3 | 60.5% | 0.022 |
| 12 | 3 | 63.3% | 0.024 |
| 13 | 3 | 65.5% | 0.028 |
| 14 | 3 | 67.3% | 0.030 |
| 15 | 7 | 89.3% | 0.481‡ |
| 16 | 3 | 69.7% | 0.036 |
| 17 | 3 | 71.1% | 0.039 |
| 18 | 3 | 71.8% | 0.041 |
| 19 | 3 | 73.2% | 0.047 |
| 20 | 3 | 74.2% | 0.043 |

Permutation test (min_size=10, n=1,000): observed ARI=0.248, p<0.001.

‡Isolated exception: 7 near-perfectly pure micro-clusters covering only 10.7% of the corpus at 89.3% noise — a boundary artefact in HDBSCAN's hierarchical cluster selection (neighbouring integers 14 and 16 both show near-null ARI with stable cluster counts), not evidence of a resolution-dependent trend. Excluded from main-text summary; reported in the paper's Appendix for transparency.

**Pooled (KO+EN) clustering** yields ARI≈0 across all tested min_cluster_size values (5, 10, 20, 50, 100), but clusters are 81–100% monolingual — the null result reflects a language-identity axis, not evidence against risk-level structure.

**Korean-only clustering** was inconclusive (100% noise at min_cluster_size≥10) due to smaller sample size and class imbalance relative to the English subset.

At min_cluster_size=10, no-distress (L0) utterances are recovered with perfect cluster purity (32 of 111, inconsistent with L0's 4.1% base rate); moderate (L2) and mild (L3) are largely separated (11 of 16 clusters at 100% purity); severe-risk (L1, n=65) is never the dominant label in any cluster.

### ESConv Direct Comparison (multi-turn)

| Corpus | Lang | N | MSW₀.₅ | logrank p |
|---|---|---|---|---|
| KO-Dep | KO | 51 | 10 | 0.13 (ns) |
| EN-Dep | EN | 510 | 7 | |
| KO-Anx | KO | 47 | 8 | <0.001*** |
| EN-Anx | EN | 381 | 14 | |
| AnnoMI | EN | 133 | >90 | — |

## Annotation Guidelines

See `annotation_guidelines.md` for:
- L0–L3 level definitions with boundary cases
- Korean honorific and somatic metaphor handling
- Negation rules
- 20 annotated boundary examples

**Risk taxonomy:**

| Level | Label | Description |
|---|---|---|
| L0 | No distress | No psychological suffering |
| L1 | Severe | Active suicidal intent, self-harm, or planning |
| L2 | Moderate | Depressive symptoms, hopelessness, passive ideation |
| L3 | Mild | Negative affect without suicidal content |

**Binary threshold:**
- Severe/moderate risk: L1–L2
- No or mild distress: L0, L3

**Inter-rater reliability:**
- LLM vs. expert: κ=0.41 (binary), κ=0.28 (4-level)
- Lay annotator vs. expert: κ=0.49 (binary, n=100)
- Cross-lingual calibration: κ=0.71 (n=100)

## Compute Requirements

| Task | Hardware | Time |
|---|---|---|
| BERT training | NVIDIA T4 (16GB) | ~2h |
| Embedding extraction (N=6,345) | T4 | ~20min/encoder |
| Survival analysis + baselines | CPU | <10min |
| Sequential baseline (k=3,5,10) | T4 | ~3h |
| Detection coverage (Step 6) | T4 | ~5min |
| HDBSCAN grid + permutation test (Step 7) | CPU | <10min |

All results reproducible with seed=42 throughout.

## Citation

```bibtex
@inproceedings{anonymous2027eacl,
  title     = {Cross-Lingual Alignment Without
               Categorical Separability:
               Temporal Safety Thresholds for
               Mental Health Risk Detection},
  author    = {Anonymous},
  booktitle = {Proceedings of the 2027 Conference
               of the European Chapter of the
               Association for Computational
               Linguistics (EACL)},
  year      = {2027}
}
```

## License

- Code: MIT License
- Data: Subject to original dataset licenses (AI Hub Terms of Use; MIT for English corpus)
