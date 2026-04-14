"""Phase 5.2 API — validation, response shape, error mapping (mocked pipeline)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api import app
from src.retrieval_request import RetrievalRequest


def _sample_request() -> RetrievalRequest:
    return RetrievalRequest(
        query_text="hello",
        top_k=5,
        task_type="general_qa",
        query_family="exact_issue_lookup",
    )


def _pipeline_return(*, explain: bool) -> dict:
    base = {
        "answer": "Example answer.",
        "request": _sample_request(),
        "prompt_template_id": "family_exact_issue_lookup",
        "prompt_template_label": "Exact",
        "llm_backend": "ollama",
        "llm_model": "llama3",
    }
    if explain:
        base["explanation"] = {
            "evidence": [
                {
                    "chunk_id": "c1",
                    "source_id": "B00X",
                    "chunk_text": "sample",
                    "rank_position": 1,
                }
            ],
            "reasoning_summary": {"summary_line": "vector", "rerank_applied": False},
            "confidence": {
                "confidence_label": "medium",
                "confidence_score": 0.5,
                "confidence_reasons": [],
            },
        }
    return base


class TestQueryAPI(unittest.TestCase):
    def tearDown(self) -> None:
        app.state.pipeline = None

    def test_health(self) -> None:
        with TestClient(app) as client:
            r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_invalid_request_missing_query(self) -> None:
        with TestClient(app) as client:
            r = client.post("/query", json={"explain": False})
        self.assertEqual(r.status_code, 400)
        body = r.json()
        self.assertEqual(body["error"]["code"], "invalid_request")
        self.assertIn("message", body["error"])

    def test_invalid_request_empty_query(self) -> None:
        with TestClient(app) as client:
            r = client.post("/query", json={"query": ""})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"]["code"], "invalid_request")

    def test_explain_false_explanation_null(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.return_value = _pipeline_return(explain=False)
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post("/query", json={"query": "What is this?"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["answer"], "Example answer.")
        self.assertIsNone(data["explanation"])
        self.assertIn("metadata", data)
        self.assertEqual(data["metadata"]["query_family"], "exact_issue_lookup")
        self.assertFalse(data["metadata"]["is_followup"])
        self.assertEqual(data["metadata"]["resolved_query"], "What is this?")
        mock_p.answer.assert_called_once()
        _args, call_kw = mock_p.answer.call_args
        self.assertFalse(call_kw.get("explain", False))

    def test_conversation_context_scope_merge_calls_pipeline(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.return_value = _pipeline_return(explain=False)
        conv = {
            "turns": [
                {
                    "user_query_raw": "Summarize complaints",
                    "resolved_query": "Summarize complaints",
                    "query_family": "abstract_complaint_summary",
                    "filters": {},
                    "answer_summary": "…",
                    "chunk_ids": [],
                    "explain_used": False,
                }
            ]
        }
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post(
                "/query",
                json={"query": "what about one-star", "conversation_context": conv},
            )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["metadata"]["is_followup"])
        self.assertEqual(r.json()["metadata"]["followup_type"], "scope")
        _args, call_kw = mock_p.answer.call_args
        self.assertEqual(call_kw.get("filter_overrides"), {"review_rating": 1})
        self.assertFalse(call_kw.get("reset_filters", False))
        effective_q = (_args[0] if _args else "") or ""
        self.assertIn("one-star", effective_q.lower())

    def test_explain_followup_forces_explain_and_transparency(self) -> None:
        base = _pipeline_return(explain=True)
        mock_p = MagicMock()
        mock_p.answer.return_value = base
        conv = {
            "turns": [
                {
                    "user_query_raw": "Summarize issues",
                    "resolved_query": "Summarize issues",
                    "query_family": "abstract_complaint_summary",
                    "filters": {},
                    "answer_summary": "…",
                    "chunk_ids": ["c1"],
                    "explain_used": False,
                }
            ]
        }
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post(
                "/query",
                json={"query": "why", "explain": False, "conversation_context": conv},
            )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["metadata"]["explain_used"])
        self.assertIsNotNone(data["explanation"])
        ct = (data["explanation"] or {}).get("conversation_transparency")
        self.assertIsInstance(ct, dict)
        self.assertEqual(ct.get("original_query"), "why")
        self.assertEqual(ct.get("resolved_query"), "Summarize issues")
        self.assertIn("prior_turn", ct)
        _a, call_kw = mock_p.answer.call_args
        self.assertTrue(call_kw.get("explain"))

    def test_explain_true_has_canonical_keys(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.return_value = _pipeline_return(explain=True)
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post(
                "/query",
                json={"query": "What is this?", "explain": True},
            )
        self.assertEqual(r.status_code, 200)
        ex = r.json()["explanation"]
        self.assertIsNotNone(ex)
        self.assertIn("evidence", ex)
        self.assertIn("reasoning_summary", ex)
        self.assertIn("confidence", ex)
        self.assertNotIn("answer", ex)
        _a, call_kw = mock_p.answer.call_args
        self.assertTrue(call_kw.get("explain"))

    def test_pipeline_value_error_unsupported_backend(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.side_effect = ValueError(
            "Unsupported LLM backend: 'foo'. Use 'openai' or 'ollama'."
        )
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post("/query", json={"query": "q", "llm_backend": "foo"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"]["code"], "invalid_request")

    def test_pipeline_connection_error(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.side_effect = ConnectionError("Ollama server not reachable")
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post("/query", json={"query": "q"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "backend_unavailable")

    def test_pipeline_unexpected_exception(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.side_effect = RuntimeError("boom")
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post("/query", json={"query": "q"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "backend_unavailable")

    def test_pipeline_generic_exception_internal(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.side_effect = KeyError("missing")
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post("/query", json={"query": "q"})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["error"]["code"], "internal_error")

    def test_openai_missing_key_value_error(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.side_effect = ValueError("OPENAI_API_KEY is not set.")
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post("/query", json={"query": "q", "llm_backend": "openai"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "backend_unavailable")

    def test_pipeline_io_error(self) -> None:
        mock_p = MagicMock()
        mock_p.answer.side_effect = FileNotFoundError("faiss.index")
        with TestClient(app) as client:
            app.state.pipeline = mock_p
            r = client.post("/query", json={"query": "q"})
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["error"]["code"], "pipeline_error")


if __name__ == "__main__":
    unittest.main()
