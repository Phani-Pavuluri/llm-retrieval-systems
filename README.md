# LLM Retrieval Systems (RAG shell)

Rule-based **query → retrieval → (optional) rerank → prompt → LLM** pipeline over chunked review text, with **traces**, **manual answer eval**, and **selective reranking**.

## Stack (short)

`QueryParser` → `Retriever` / `retrieval_strategy` → optional **cross-encoder rerank** (`rerank_policy` gates it) → `prompt_builder` → Ollama or OpenAI.

## Docs

**System evolution (phases, metrics, limitations):** [SYSTEM_EVOLUTION.md](SYSTEM_EVOLUTION.md)  
**Roadmaps:** [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md) (product / Phase 5.x), [ML_CAPABILITIES_ROADMAP.md](ML_CAPABILITIES_ROADMAP.md) (ML / Phase 6)

| File | Purpose |
|------|---------|
| [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) | Architecture & data flow (interview-style) |
| [SYSTEM_EVOLUTION.md](SYSTEM_EVOLUTION.md) | Phased changes, problems, fixes, **measured** impact |
| [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md), [ML_CAPABILITIES_ROADMAP.md](ML_CAPABILITIES_ROADMAP.md) | Product (Phase 5.x) vs ML / analytics (Phase 6) forward plan |
| [DAILY_LOG.md](DAILY_LOG.md) | Append-only change log |

## Quick start

```bash
pip install -r requirements.txt
# Place sample data under data/ per config, then:
PYTHONPATH=. python scripts/build_index.py
PYTHONPATH=. python scripts/run_query.py "Your question"
```

Answer eval (12 labeled queries): `scripts/run_answer_eval.py` → summarize with `scripts/summarize_answer_eval.py` (labels in `eval/answer_eval_labeled.json`).

## Author

Phani Pavuluri
