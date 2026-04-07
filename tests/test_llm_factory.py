"""Minimal tests for LLM factory and backend error messages."""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from src.llm import OllamaLLM, OpenAILLM, get_llm


class TestGetLLM(unittest.TestCase):
    def test_unsupported_backend_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_llm("azure")
        self.assertIn("Unsupported LLM backend", str(ctx.exception))
        self.assertIn("openai", str(ctx.exception))
        self.assertIn("ollama", str(ctx.exception))

    def test_openai_missing_api_key_raises_clear_message(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                get_llm("openai")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    @patch("src.llm.OpenAI")
    def test_openai_initializes_with_key(self, mock_openai: MagicMock) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            llm = get_llm("openai", model_name="gpt-4o-mini")
        self.assertIsInstance(llm, OpenAILLM)
        mock_openai.assert_called_once()

    @patch("src.llm.requests.post")
    def test_ollama_returns_response_field(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "hello from ollama"}
        mock_post.return_value = mock_resp

        llm = OllamaLLM(model_name="llama3")
        out = llm.generate("test prompt")
        self.assertEqual(out, "hello from ollama")
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["model"], "llama3")
        self.assertEqual(kwargs["json"]["prompt"], "test prompt")
        self.assertFalse(kwargs["json"]["stream"])

    @patch("src.llm.requests.post")
    def test_ollama_non_200_raises_clear_message(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal error"
        mock_resp.reason = "Server Error"
        mock_post.return_value = mock_resp

        llm = OllamaLLM(model_name="llama3")
        with self.assertRaises(RuntimeError) as ctx:
            llm.generate("x")
        self.assertIn("Ollama API error", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))

    @patch("src.llm.requests.post")
    def test_ollama_unreachable_raises_connection_error(
        self, mock_post: MagicMock
    ) -> None:
        import requests

        mock_post.side_effect = requests.ConnectionError("refused")

        llm = OllamaLLM(model_name="llama3")
        with self.assertRaises(ConnectionError) as ctx:
            llm.generate("x")
        self.assertIn("Ollama server not reachable", str(ctx.exception))
        self.assertIn("Is Ollama running", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
