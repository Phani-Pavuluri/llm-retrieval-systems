from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.answer_trace import append_answer_trace
from src.prompt_builder import (
    build_answer_prompt,
    format_evidence_block,
    select_prompt_template_id,
)
from src.retrieval_request import RetrievalRequest


class TestSelectTemplate(unittest.TestCase):
    def test_family_over_task(self) -> None:
        r = RetrievalRequest(
            query_text="q",
            top_k=5,
            task_type="general_qa",
            query_family="value_complaint",
        )
        self.assertEqual(select_prompt_template_id(r), "family_value_complaint")

    def test_task_complaint_fallback(self) -> None:
        r = RetrievalRequest(query_text="complaints about defects", top_k=5, task_type="complaint_summary")
        self.assertEqual(select_prompt_template_id(r), "task_complaint_summary")

    def test_buyer_risk_family_template(self) -> None:
        r = RetrievalRequest(
            query_text="what should a buyer watch out for",
            top_k=5,
            task_type="complaint_summary",
            query_family="buyer_risk_issues",
        )
        self.assertEqual(select_prompt_template_id(r), "family_buyer_risk_issues")

    def test_symptom_extraction_family_template(self) -> None:
        r = RetrievalRequest(
            query_text="list symptoms like rash",
            top_k=5,
            task_type="general_qa",
            query_family="symptom_issue_extraction",
        )
        self.assertEqual(select_prompt_template_id(r), "family_symptom_issue_extraction")


class TestFormatEvidence(unittest.TestCase):
    def test_empty_frame(self) -> None:
        ctx, ids = format_evidence_block(pd.DataFrame())
        self.assertEqual(ids, [])
        self.assertIn("No review excerpts", ctx)

    def test_numbered_chunks(self) -> None:
        df = pd.DataFrame(
            {"chunk_id": ["c1", "c2"], "text": ["hello", "world"]}
        )
        ctx, ids = format_evidence_block(df)
        self.assertEqual(ids, ["c1", "c2"])
        self.assertIn("[Chunk 1 | chunk_id=c1]", ctx)

    def test_metadata_in_excerpt_headers(self) -> None:
        df = pd.DataFrame(
            {
                "chunk_id": ["B00ABCDEF_0_0"],
                "text": ["Great soap."],
                "asin": ["B00ABCDEF"],
                "brand": ["Acme"],
                "review_title": ["Love it"],
                "category": ["Beauty"],
                "sub_category": ["Soap"],
            }
        )
        ctx, _ids = format_evidence_block(df)
        self.assertIn("asin=B00ABCDEF", ctx)
        self.assertIn("brand=Acme", ctx)
        self.assertIn("review_title=Love it", ctx)
        self.assertIn("category=Beauty", ctx)
        self.assertIn("sub_category=Soap", ctx)


class TestBuildPrompt(unittest.TestCase):
    def test_grounding_instructions_present(self) -> None:
        r = RetrievalRequest(query_text="q", top_k=5, task_type="general_qa")
        df = pd.DataFrame({"chunk_id": ["a"], "text": ["only evidence"]})
        b = build_answer_prompt(r, "What is X?", df)
        self.assertIn("Insufficient evidence", b.prompt)
        self.assertIn("only evidence", b.prompt)
        self.assertTrue(b.chunk_ids)

    def test_causal_and_multi_chunk_rules_present(self) -> None:
        r = RetrievalRequest(query_text="q", top_k=5, task_type="general_qa")
        df = pd.DataFrame(
            {"chunk_id": ["a", "b"], "text": ["one", "two"]}
        )
        b = build_answer_prompt(r, "Q?", df)
        self.assertIn("Do NOT infer causes", b.prompt)
        self.assertIn("at least two different Chunk numbers", b.prompt)

    def test_rating_scope_scalar_one(self) -> None:
        r = RetrievalRequest(
            query_text="q",
            top_k=5,
            task_type="complaint_summary",
            filters={"review_rating": 1},
        )
        df = pd.DataFrame({"chunk_id": ["a"], "text": ["t"]})
        b = build_answer_prompt(r, "One-star issues?", df)
        self.assertIn("review_rating equal to 1", b.prompt)
        self.assertIn("ONLY include negative feedback", b.prompt)

    def test_output_style_hints_block(self) -> None:
        r = RetrievalRequest(query_text="q", top_k=5, task_type="general_qa")
        df = pd.DataFrame({"chunk_id": ["a"], "text": ["t"]})
        b = build_answer_prompt(
            r,
            "Same question",
            df,
            output_style_hints={"brevity": "short", "max_issues": 3},
        )
        self.assertIn("Output constraints", b.prompt)
        self.assertIn("concise", b.prompt.lower())
        self.assertIn("at most 3", b.prompt)

    def test_symptom_template_has_negation_rules(self) -> None:
        r = RetrievalRequest(
            query_text="q",
            top_k=5,
            query_family="symptom_issue_extraction",
            task_type="general_qa",
        )
        df = pd.DataFrame({"chunk_id": ["a", "b"], "text": ["x", "y"]})
        b = build_answer_prompt(r, "List rash if mentioned", df)
        self.assertIn("ONLY list symptoms", b.prompt)
        self.assertIn("other deodorants", b.prompt)

    def test_unknown_template_id_falls_back(self) -> None:
        r = RetrievalRequest(query_text="q", top_k=5, task_type="general_qa")
        df = pd.DataFrame({"chunk_id": ["a"], "text": ["t"]})
        b = build_answer_prompt(r, "Q?", df, template_id="not_a_real_template")
        self.assertEqual(b.template_id, "grounded_default")


class TestAnswerTrace(unittest.TestCase):
    def test_append_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.jsonl"
            append_answer_trace({"query": "q", "answer": "a"}, p)
            line = p.read_text(encoding="utf-8").strip()
            row = json.loads(line)
            self.assertEqual(row["query"], "q")
            self.assertIn("ts", row)


if __name__ == "__main__":
    unittest.main()
