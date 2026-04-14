"""
Phase 5.4 — Bounded conversation context (client-supplied; no server session DB).

At most MAX_TURNS recent turns; each turn is a small structured record (no LLM summaries).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_TURNS = 5


class TurnRecord(BaseModel):
    """One completed turn for carry-forward into the resolver."""

    user_query_raw: str = ""
    resolved_query: str = ""
    query_family: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    answer_summary: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    explain_used: bool = False

    @field_validator("chunk_ids", mode="before")
    @classmethod
    def _coerce_chunk_ids(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return []


class ConversationContext(BaseModel):
    """Recent turns only; passed on each request — backend stays stateless."""

    turns: list[TurnRecord] = Field(default_factory=list)

    @field_validator("turns", mode="before")
    @classmethod
    def _cap_turns(cls, v: Any) -> list[Any]:
        if not v:
            return []
        if isinstance(v, list):
            return v[-MAX_TURNS:]
        return []

    def last_turn(self) -> TurnRecord | None:
        return self.turns[-1] if self.turns else None


def parse_conversation_context(raw: dict[str, Any] | None) -> ConversationContext | None:
    """Validate client JSON; returns None if missing or empty."""
    if not raw:
        return None
    try:
        ctx = ConversationContext.model_validate(raw)
    except Exception:
        return None
    if not ctx.turns:
        return None
    return ctx
