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