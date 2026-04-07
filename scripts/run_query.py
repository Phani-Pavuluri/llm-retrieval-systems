from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG query with OpenAI or Ollama LLM backend."
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "ollama"],
        default=None,
        help="LLM backend (default: config LLM_BACKEND)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the backend (default: OPENAI_MODEL or OLLAMA_MODEL from config)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Question to ask",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    args = parser.parse_args()

    from src.rag_pipeline import RAGPipeline

    backend = args.backend if args.backend is not None else config.LLM_BACKEND
    pipeline = RAGPipeline(llm_backend=backend, llm_model=args.model)

    query = args.query or (
        "What kinds of counterfeit or defective product complaints appear in these reviews?"
    )
    result = pipeline.answer(query, k=args.k)

    print("\nQUERY:")
    print(result["query"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED CHUNKS:")
    cols = ["asin", "review_rating", "brand", "category", "score", "text"]
    print(result["retrieved_chunks"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
