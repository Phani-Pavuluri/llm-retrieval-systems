# ML capabilities roadmap

## Goal

Evolve the system into a **data science assistant** that can support **descriptive**, **diagnostic**, **predictive**, **causal**, and **decision-support** workflows — **beyond** retrieval-only Q&A.

---

## Why this is separate from the current pipeline

The **current** stack is optimized for:

- Retrieval and ranking  
- **Grounded** answering and summarization  
- Explainability of **that** path (Phase 5.1)  

Future **ML capabilities** add **computation-heavy** analysis that should not be faked inside an LLM prompt.

---

## Tool abstraction (formal target)

A **tool** is any composable unit the router can invoke **by name** with a **typed contract**. Phase 6 work should converge on:

| Piece | Meaning |
|-------|---------|
| **Input schema** | Structured args (filters, metric defs, horizons, dataset handles) — not raw prose. |
| **Execution engine** | Where work runs: **Python** (pandas / sklearn / statsmodels), **SQL** (warehouse), **registered model** artifact, etc. |
| **Output schema** | Tabular result + diagnostics (metrics, intervals, warnings) suitable for JSON. |
| **Explanation contract** | What the LLM (or UI) may say: summarize **tool output fields**, never invent numbers absent from output. |

**Example tool categories (illustrative):**

- **Retrieval tool** — the existing RAG path (already the default “tool”).  
- **Aggregation tool** — grouped metrics, funnels, cohort cuts.  
- **Forecasting tool** — series → forecast object + backtest metadata.  
- **Experiment analysis tool** — design-aware effect estimates from tabular inputs.

This keeps Phase 6 from becoming ad-hoc “prompt the model to do stats.”

---

## Capability categories

### Descriptive analytics

- Aggregations, groupby-style breakdowns  
- Funnels, cohort-style views  
- Summary statistics  

### Diagnostic analysis

- Contribution / driver-style analysis  
- Regression diagnostics, residuals  
- Feature importance (model-based)  
- Change vs baseline  

### Prediction

- Classification, regression  
- Risk scoring  
- Demand or rating prediction (where data supports it)  

### Time-series forecasting

- Baseline / naive forecasts  
- ARIMA / SARIMA, Prophet-style components  
- Lag-based regression, seasonality / trend decomposition  

### Anomaly detection

- Z-score / residual thresholds  
- Change-point style signals  
- Isolation forest, monitoring-style alerts  

### Clustering / segmentation

- k-means, hierarchical, density-based methods  
- Embedding-based grouping  

### NLP / text analytics

- Topic extraction, sentiment, text classification  
- Theme detection, similarity / dedup  

### Experimentation / causal inference

- A/B testing, CUPED  
- Diff-in-diff, uplift / treatment effects  
- Quasi-experimental designs where appropriate  

### Optimization / decision support

- Constrained optimization, allocation / budgets  
- Scenario and trade-off exploration  

### Explainability / evaluation / monitoring (for ML outputs)

- Feature importance, calibration checks  
- Drift, forecast backtesting  
- Slice analysis, error analysis  

---

## Phased implementation plan

### Phase 6.1 (first)

- Descriptive + diagnostic analytics  
- Basic forecasting  
- Simple supervised prediction  

### Phase 6.2

- Anomaly detection  
- Clustering / segmentation  
- Text analytics jobs composable with retrieval  

### Phase 6.3

- Experimentation / causal inference  
- Optimization / decision support  
- More advanced forecasting  

---

## Design principles

- **LLM:** intent, **routing**, summarization, and **explanation** of results.  
- **Tools:** actual **numeric / statistical / ML** computation.  
- **Retrieval** and **analytics** stay **distinct but composable**.  
- **No fake predictions** from the LLM alone.  

---

## Long-term architecture

```
Query → Intent Router → Tool Selection →
    Retrieval path  |  Analytics / ML path
→ Result → LLM explanation
```

See **SYSTEM_EVOLUTION.md §13** and **`PRODUCT_ROADMAP.md`** for how product layers expose this.
