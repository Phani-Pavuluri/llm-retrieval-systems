"""Tests for ui/chat_helpers (no Streamlit)."""
from __future__ import annotations

import unittest

from ui.chat_helpers import (
    build_query_json,
    confidence_markdown_lines,
    conversation_context_from_turn_history,
    evidence_score_parts,
    format_api_error,
    metadata_markdown_lines,
    reasoning_summary_lines,
)


class TestBuildQueryJson(unittest.TestCase):
    def test_minimal(self) -> None:
        j = build_query_json(query="  hello  ", explain=False)
        self.assertEqual(j, {"query": "hello", "explain": False})

    def test_explain_and_overrides(self) -> None:
        j = build_query_json(
            query="q",
            explain=True,
            llm_backend="ollama",
            llm_model="mistral",
            k=7,
            selective_rerank=False,
            rerank_model="cross-encoder/x",
            rerank_top_n=8,
        )
        self.assertTrue(j["explain"])
        self.assertEqual(j["llm_backend"], "ollama")
        self.assertEqual(j["llm_model"], "mistral")
        self.assertEqual(j["k"], 7)
        self.assertFalse(j["selective_rerank"])
        self.assertEqual(j["rerank_model"], "cross-encoder/x")
        self.assertEqual(j["rerank_top_n"], 8)

    def test_omits_none_optional(self) -> None:
        j = build_query_json(query="x", explain=False, llm_backend=None, k=None)
        self.assertNotIn("llm_backend", j)
        self.assertNotIn("k", j)

    def test_conversation_context_passed(self) -> None:
        ctx = {"turns": [{"user_query_raw": "a", "resolved_query": "b"}]}
        j = build_query_json(query="q", explain=False, conversation_context=ctx)
        self.assertEqual(j["conversation_context"], ctx)

    def test_query_planner_only_when_true(self) -> None:
        j_off = build_query_json(query="q", explain=False, query_planner=False)
        self.assertNotIn("query_planner", j_off)
        j_on = build_query_json(query="q", explain=False, query_planner=True)
        self.assertTrue(j_on["query_planner"])


class TestConversationContextFromHistory(unittest.TestCase):
    def test_builds_turns(self) -> None:
        turns = [
            {
                "query": " hello ",
                "ok": True,
                "data": {
                    "answer": "x" * 500,
                    "metadata": {
                        "resolved_query": "resolved hello",
                        "query_family": "value_complaint",
                        "filters_applied": {"review_rating": 1},
                        "chunk_ids_used": ["a", "b"],
                    },
                },
            }
        ]
        ctx = conversation_context_from_turn_history(turns)
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(len(ctx["turns"]), 1)
        row = ctx["turns"][0]
        self.assertEqual(row["user_query_raw"], "hello")
        self.assertEqual(row["resolved_query"], "resolved hello")
        self.assertEqual(row["query_family"], "value_complaint")
        self.assertEqual(row["filters"], {"review_rating": 1})
        self.assertEqual(row["chunk_ids"], ["a", "b"])
        self.assertTrue(row["answer_summary"].endswith("…"))

    def test_skips_failed_turns(self) -> None:
        self.assertIsNone(
            conversation_context_from_turn_history([{"query": "q", "ok": False}])
        )


class TestEvidenceScoreParts(unittest.TestCase):
    def test_scores(self) -> None:
        parts = evidence_score_parts(
            {"rerank_score": 0.9, "semantic_score": 0.5, "foo": 1}
        )
        self.assertTrue(any("rerank_score=" in p for p in parts))
        self.assertTrue(any("semantic_score=" in p for p in parts))


class TestReasoningSummaryLines(unittest.TestCase):
    def test_ordered(self) -> None:
        rs = {
            "query_family": "exact_issue_lookup",
            "retrieval_mode": "vector",
            "rerank_applied": False,
            "summary_line": "Summary here.",
            "filters_applied": {},
            "strategy_reason": "",
        }
        lines = reasoning_summary_lines(rs)
        text = "\n".join(lines)
        self.assertIn("retrieval_mode", text)
        self.assertIn("Summary here.", text)


class TestConfidenceMarkdown(unittest.TestCase):
    def test_reasons(self) -> None:
        lines = confidence_markdown_lines(
            {
                "confidence_label": "low",
                "confidence_score": 0.2,
                "confidence_reasons": ["a", "b"],
            }
        )
        self.assertTrue(any("low" in ln for ln in lines))
        self.assertTrue(any("a" in ln for ln in lines))


class TestMetadataMarkdown(unittest.TestCase):
    def test_order(self) -> None:
        meta = {
            "query_family": "f",
            "llm_backend": "ollama",
            "prompt_template_id": "tid",
            "llm_model": "m",
            "selective_rerank_effective": True,
        }
        lines = metadata_markdown_lines(meta)
        joined = "\n".join(lines)
        self.assertIn("llm_backend", joined)
        self.assertIn("query_family", joined)


class TestFormatApiError(unittest.TestCase):
    def test_shape(self) -> None:
        c, m, d = format_api_error(
            {"error": {"code": "invalid_request", "message": "bad", "details": [1]}}
        )
        self.assertEqual(c, "invalid_request")
        self.assertEqual(m, "bad")
        self.assertEqual(d, [1])


if __name__ == "__main__":
    unittest.main()
