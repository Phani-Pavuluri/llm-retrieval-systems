# Run commands (copy-paste)

**Assume you are in the repository root** (directory containing `Makefile`, `src/`, `scripts/`). **`PYTHONPATH=.`** is required so `src` and `ui` resolve.

---

## API

```bash
PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## Chat UI

```bash
PYTHONPATH=. streamlit run ui/chat_ui.py
```

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## Retrieval eval

```bash
PYTHONPATH=. python scripts/eval_labeled_retrieval.py
```

## Rerank analysis

```bash
PYTHONPATH=. python scripts/analyze_rerank_impact.py
```

## Selective rerank validation

```bash
PYTHONPATH=. python scripts/validate_selective_rerank.py --run-eval
```

## Answer eval (generate answers)

```bash
PYTHONPATH=. python scripts/run_answer_eval.py
```

## Answer eval summary

```bash
PYTHONPATH=. python scripts/summarize_answer_eval.py
```

---

**Daily driver:** keep this file minimal. Explanations live in **`RUN_GUIDE.md`**.
