"""
Append-only JSONL logs for answer generation (Phase 4).

Separate from retrieval traces: one line per RAG answer, for debugging prompts and models.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config


def default_answer_trace_path() -> Path:
    base = getattr(config, "ANSWER_TRACE_DIR", config.PROJECT_ROOT / "artifacts" / "answer_traces")
    base.mkdir(parents=True, exist_ok=True)
    return Path(base) / "answers.jsonl"


def append_answer_trace(record: dict[str, Any], path: Path | None = None) -> Path:
    """
    Write one JSON object as a line. Returns the path written.

    `record` should already include any fields the caller wants; this function adds "ts" in UTC.
    """
    out = path or default_answer_trace_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return out.resolve()
