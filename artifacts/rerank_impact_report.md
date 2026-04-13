# Rerank impact report (Phase 3)

**Source traces:** `/Users/ppavuluri/Desktop/retreival_systems/llm-retrieval-systems/artifacts/retrieval_traces/labeled_eval_run.jsonl`
**Rows loaded:** 60
**Rows with `rerank_applied`:** 24

## 1. Metrics by mode (overall)

| Mode | P@k | R@k | MRR | n |
|------|-----|-----|-----|---|
| vector | 0.233 | 0.639 | 0.642 | 12 |
| hybrid | 0.267 | 0.708 | 0.771 | 12 |
| auto | 0.233 | 0.639 | 0.642 | 12 |
| auto_rerank | 0.217 | 0.625 | 0.611 | 12 |
| hybrid_rerank | 0.233 | 0.653 | 0.611 | 12 |

## 2. Metrics by `query_family`

### Family: `abstract_complaint_summary`

| Mode | P@k | R@k | MRR |
|------|-----|-----|-----|
| vector | 0.000 | 0.000 | 0.000 |
| hybrid | 0.100 | 0.250 | 0.500 |
| auto | 0.000 | 0.000 | 0.000 |
| auto_rerank | 0.100 | 0.250 | 0.250 |
| hybrid_rerank | 0.100 | 0.250 | 0.250 |

### Family: `exact_issue_lookup`

| Mode | P@k | R@k | MRR |
|------|-----|-----|-----|
| vector | 0.275 | 0.833 | 0.812 |
| hybrid | 0.275 | 0.833 | 0.875 |
| auto | 0.275 | 0.833 | 0.812 |
| auto_rerank | 0.250 | 0.792 | 0.729 |
| hybrid_rerank | 0.250 | 0.792 | 0.729 |

### Family: `rating_scoped_summary`

| Mode | P@k | R@k | MRR |
|------|-----|-----|-----|
| vector | 0.400 | 0.667 | 1.000 |
| hybrid | 0.400 | 0.667 | 1.000 |
| auto | 0.400 | 0.667 | 1.000 |
| auto_rerank | 0.000 | 0.000 | 0.000 |
| hybrid_rerank | 0.000 | 0.000 | 0.000 |

### Family: `value_complaint`

| Mode | P@k | R@k | MRR |
|------|-----|-----|-----|
| vector | 0.200 | 0.333 | 0.200 |
| hybrid | 0.400 | 0.667 | 0.250 |
| auto | 0.200 | 0.333 | 0.200 |
| auto_rerank | 0.400 | 0.667 | 1.000 |
| hybrid_rerank | 0.600 | 1.000 | 1.000 |

## 3. Rerank vs baseline (aggregate deltas)

### auto_rerank vs auto

| Metric | Base | Rerank | Δ abs | Δ % |
|--------|------|--------|-------|-----|
| P@k | 0.2333 | 0.2167 | -0.0167 | -7.1% |
| R@k | 0.6389 | 0.6250 | -0.0139 | -2.2% |
| MRR | 0.6417 | 0.6111 | -0.0306 | -4.8% |

### hybrid_rerank vs hybrid

| Metric | Base | Rerank | Δ abs | Δ % |
|--------|------|--------|-------|-----|
| P@k | 0.2667 | 0.2333 | -0.0333 | -12.5% |
| R@k | 0.7083 | 0.6528 | -0.0556 | -7.8% |
| MRR | 0.7708 | 0.6111 | -0.1597 | -20.7% |

## 4. Rerank effectiveness (gold in top-k, pre vs post)

Counts use trace flags `gold_in_pre_rerank_top_k` and `gold_in_post_rerank_top_k` on **auto_rerank** and **hybrid_rerank** rows with non-empty gold labels.

### Overall (all rerank eval rows)

- **Helped** (pre=False, post=True): **1**
- **Unchanged** (same gold-in-top-k flag): **21**
- **Hurt** (pre=True, post=False): **2**

### By `query_family`

| Family | helped | unchanged | hurt |
|--------|--------|-----------|------|
| abstract_complaint_summary | 1 | 3 | 0 |
| exact_issue_lookup | 0 | 16 | 0 |
| rating_scoped_summary | 0 | 0 | 2 |
| value_complaint | 0 | 2 | 0 |

## 5. Representative examples

### Reranking clearly helped (2)

#### Example 1: `l07`

- **Query:** does not work ineffective deodorant
- **query_family:** `abstract_complaint_summary`
- **eval_mode:** `auto_rerank`
- **pre-rerank top-5 chunk IDs:** `['B07WDL82Z4_12_0', 'B071YYMZ19_8_0', 'B07WDL82Z4_20_0', 'B071YYMZ19_0_0', 'B07WDL82Z4_17_0']`
- **post-rerank top-5 chunk IDs:** `['B071YYMZ19_7_0', 'B071YYMZ19_3_0', 'B071YYMZ19_0_0', 'B07WDL82Z4_17_0', 'B071YYMZ19_1_0']`
- **Gold moved into top-k:** True (pre=False, post=True)
- **Note:** Order changed within top-k (e.g. 'B071YYMZ19_0_0' moved relative to retrieval ranking).

### Reranking did nothing observable (2)

#### Example 1: `l03`

- **Query:** skin irritation rash or burn from product
- **query_family:** `exact_issue_lookup`
- **eval_mode:** `auto_rerank`
- **pre-rerank top-5 chunk IDs:** `[]`
- **post-rerank top-5 chunk IDs:** `[]`
- **Gold moved into top-k:** False (pre=False, post=False)
- **Note:** Top-k chunk IDs and order unchanged after rerank.

#### Example 2: `l01`

- **Query:** counterfeit or fake product complaints
- **query_family:** `exact_issue_lookup`
- **eval_mode:** `auto_rerank`
- **pre-rerank top-5 chunk IDs:** `['B071YYMZ19_2_0', 'B071YYMZ19_4_0', 'B07WDL82Z4_12_0', 'B07WDL82Z4_18_0', 'B071YYMZ19_1_0']`
- **post-rerank top-5 chunk IDs:** `['B071YYMZ19_2_0', 'B07WDL82Z4_12_0', 'B071YYMZ19_1_0', 'B071YYMZ19_4_0', 'B07WDL82Z4_13_0']`
- **Gold moved into top-k:** False (pre=True, post=True)
- **Note:** Order changed within top-k (e.g. 'B07WDL82Z4_12_0' moved relative to retrieval ranking).

### Reranking hurt (up to 1)

#### Example 1: `l12`

- **Query:** summary of negative one-star experiences
- **query_family:** `rating_scoped_summary`
- **eval_mode:** `auto_rerank`
- **pre-rerank top-5 chunk IDs:** `['B071YYMZ19_4_0', 'B07NQN58ZC_29_0', 'B07NQN58ZC_33_0', 'B07WDL82Z4_21_0', 'B071YYMZ19_7_0']`
- **post-rerank top-5 chunk IDs:** `['B071YYMZ19_7_0', 'B071YYMZ19_3_0', 'B07NQN58ZC_29_0', 'B071YYMZ19_3_1', 'B071YYMZ19_6_0']`
- **Gold moved into top-k:** False (pre=True, post=False)
- **Note:** Order changed within top-k (e.g. 'B071YYMZ19_7_0' moved relative to retrieval ranking).

---

_Generated by `scripts/analyze_rerank_impact.py`. Re-run `eval_labeled_retrieval.py` then this script to refresh._
