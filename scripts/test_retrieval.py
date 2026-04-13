import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.retrieval_request import RetrievalRequest
from src.retriever import Retriever


def main() -> None:
    retriever = Retriever()

    query = "counterfeit issues and defective product complaints"
    request = RetrievalRequest.from_raw(query, top_k=5)
    results = retriever.retrieve(request)

    print("\nQUERY:")
    print(query)
    print("\nRESULTS:")
    preferred = ["asin", "review_rating", "brand", "category", "score", "text"]
    cols = [c for c in preferred if c in results.columns]
    print(results[cols].to_string(index=False))


if __name__ == "__main__":
    main()
