from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests
from openai import OpenAI

OLLAMA_DEFAULT_BASE = "http://localhost:11434"
OLLAMA_GENERATE_PATH = "/api/generate"


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return model text for the given prompt."""


class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = OLLAMA_DEFAULT_BASE,
    ) -> None:
        self.model_name = model_name
        self._url = f"{base_url.rstrip('/')}{OLLAMA_GENERATE_PATH}"

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                self._url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=300,
            )
        except requests.RequestException as e:
            raise ConnectionError(
                f"Ollama server not reachable at {self._url}. "
                f"Is Ollama running? ({e})"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API error ({response.status_code}): "
                f"{response.text or response.reason}"
            )

        try:
            data = response.json()
        except ValueError as e:
            body = (response.text or "")[:500]
            raise RuntimeError(
                f"Ollama returned non-JSON response (HTTP {response.status_code}): {body}"
            ) from e

        if "response" not in data:
            raise RuntimeError(
                f"Ollama JSON missing 'response' field: {data!r}"
            )
        return data["response"]


def get_llm(backend: str, model_name: str | None = None) -> BaseLLM:
    from src import config

    name = (backend or "").strip().lower()
    if name == "openai":
        model = model_name if model_name is not None else config.OPENAI_MODEL
        return OpenAILLM(model_name=model)
    if name == "ollama":
        model = model_name if model_name is not None else config.OLLAMA_MODEL
        return OllamaLLM(model_name=model)
    raise ValueError(
        f"Unsupported LLM backend: {backend!r}. Use 'openai' or 'ollama'."
    )
