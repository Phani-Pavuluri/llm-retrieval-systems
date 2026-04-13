from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


class TestAnswerEvalBucketsMatchJson(unittest.TestCase):
    def test_labeled_json_failure_buckets_match_constants(self) -> None:
        from src.answer_eval_constants import ANSWER_EVAL_FAILURE_BUCKET_SET

        p = _ROOT / "eval" / "answer_eval_labeled.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        fb = doc.get("failure_buckets")
        self.assertIsInstance(fb, list)
        self.assertEqual(frozenset(fb), ANSWER_EVAL_FAILURE_BUCKET_SET)


class TestSummarizeAnswerEval(unittest.TestCase):
    def test_aggregate_by_family(self) -> None:
        labeled = {
            "items": [
                {
                    "id": "a1",
                    "query_family": "fam_x",
                    "scores": {"grounded": 3, "correct": 2, "complete": 3},
                    "failure_bucket": "weak_synthesis",
                },
                {
                    "id": "a2",
                    "query_family": "fam_x",
                    "scores": {"grounded": 1, "correct": 1, "complete": 2},
                },
                {"id": "a3", "query_family": "fam_y", "scores": {"grounded": None, "correct": None, "complete": None}},
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            jf = td_path / "in.json"
            jf.write_text(json.dumps(labeled), encoding="utf-8")
            out = td_path / "summary.txt"
            outj = td_path / "summary.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run(
                [
                    sys.executable,
                    str(_ROOT / "scripts" / "summarize_answer_eval.py"),
                    "--labeled",
                    str(jf),
                    "--out",
                    str(out),
                    "--out-json",
                    str(outj),
                ],
                cwd=str(_ROOT),
                env=env,
                check=True,
            )
            data = json.loads(outj.read_text(encoding="utf-8"))
            self.assertEqual(data["labeled_count"], 2)
            self.assertIn("a3", data["pending_ids"])
            fx = data["by_query_family"]["fam_x"]
            self.assertEqual(fx["n"], 2)
            self.assertAlmostEqual(fx["mean_grounded"], 2.0)
            self.assertEqual(data["failure_bucket_counts"]["weak_synthesis"], 1)

    def test_unknown_failure_bucket_reported(self) -> None:
        labeled = {
            "failure_buckets": ["hallucination", "weak_synthesis"],
            "items": [
                {
                    "id": "x1",
                    "query_family": "fam",
                    "scores": {"grounded": 1, "correct": 1, "complete": 1},
                    "failure_bucket": "not_a_real_bucket",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            jf = td_path / "in.json"
            jf.write_text(json.dumps(labeled), encoding="utf-8")
            outj = td_path / "summary.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run(
                [
                    sys.executable,
                    str(_ROOT / "scripts" / "summarize_answer_eval.py"),
                    "--labeled",
                    str(jf),
                    "--out",
                    str(td_path / "s.txt"),
                    "--out-json",
                    str(outj),
                ],
                cwd=str(_ROOT),
                env=env,
                check=True,
            )
            data = json.loads(outj.read_text(encoding="utf-8"))
            self.assertEqual(data["unknown_failure_bucket_counts"]["not_a_real_bucket"], 1)
            self.assertEqual(data["failure_bucket_counts"], {})


if __name__ == "__main__":
    unittest.main()
