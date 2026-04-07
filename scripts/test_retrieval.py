import sys
from pathlib import Path

# Project root (parent of `scripts/`) must be on path for `import src`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.retriever import Retriever


def main() -> None:
    retriever = Retriever()

    query = "counterfeit issues and defective product complaints"
    results = retriever.retrieve(query, k=5)

    print("\nQUERY:")
    print(query)
    print("\nRESULTS:")
    print(results[["asin", "review_rating", "brand", "category", "score", "text"]].to_string(index=False))


if __name__ == "__main__":
    main()