# Optional shortcuts — run from repo root (same as RUN_COMMANDS.md).
.PHONY: help api ui test eval rerank-analyze rerank-val answer-eval answer-summary

help:
	@echo "Targets: api | ui | test | eval | rerank-analyze | rerank-val | answer-eval | answer-summary"
	@echo "See RUN_COMMANDS.md (commands) and RUN_GUIDE.md (explanations)."

api:
	PYTHONPATH=. uvicorn src.api:app --host 127.0.0.1 --port 8000

ui:
	PYTHONPATH=. streamlit run ui/chat_ui.py

test:
	PYTHONPATH=. python -m unittest discover -s tests -v

eval:
	PYTHONPATH=. python scripts/eval_labeled_retrieval.py

rerank-analyze:
	PYTHONPATH=. python scripts/analyze_rerank_impact.py

rerank-val:
	PYTHONPATH=. python scripts/validate_selective_rerank.py --run-eval

answer-eval:
	PYTHONPATH=. python scripts/run_answer_eval.py

answer-summary:
	PYTHONPATH=. python scripts/summarize_answer_eval.py
