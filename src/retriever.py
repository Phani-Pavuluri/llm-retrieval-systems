import pandas as pd

from src.config import TOP_K, VECTOR_STORE_DIR
from src.embeddings import build_embedder_from_config
from src.vector_store import FaissVectorStore, load_metadata


class Retriever:
    def __init__(self) -> None:
        self.embedder = build_embedder_from_config()
        self.store = FaissVectorStore.load(VECTOR_STORE_DIR / "faiss.index")
        self.metadata = load_metadata(VECTOR_STORE_DIR / "chunk_metadata.csv")

    def retrieve(self, query: str, k: int = TOP_K) -> pd.DataFrame:
        query_embedding = self.embedder.embed_query(query)
        scores, indices = self.store.search(query_embedding, k=k)

        rows = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx].to_dict()
            row["score"] = float(score)
            rows.append(row)

        return pd.DataFrame(rows)