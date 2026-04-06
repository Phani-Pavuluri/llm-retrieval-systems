import pandas as pd

from src.config import PROCESSED_DATA_DIR, VECTOR_STORE_DIR
from src.embeddings import build_embedder_from_config
from src.vector_store import FaissVectorStore, save_metadata


def main() -> None:
    chunks_path = PROCESSED_DATA_DIR / "review_chunks.csv"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks_df = pd.read_csv(chunks_path)
    if chunks_df.empty:
        raise ValueError("Chunks file is empty.")

    print(f"Loaded {len(chunks_df)} chunks")

    texts = chunks_df["text"].fillna("").astype(str).tolist()

    embedder = build_embedder_from_config()
    embeddings = embedder.embed_texts(texts)

    print(f"Embeddings shape: {embeddings.shape}")

    store = FaissVectorStore(dimension=embeddings.shape[1])
    store.add(embeddings)

    index_path = VECTOR_STORE_DIR / "faiss.index"
    metadata_path = VECTOR_STORE_DIR / "chunk_metadata.csv"

    store.save(index_path)
    save_metadata(chunks_df, metadata_path)

    print(f"Saved index to: {index_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()