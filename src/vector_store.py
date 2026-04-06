from pathlib import Path
from typing import Tuple

import faiss
import numpy as np
import pandas as pd


class FaissVectorStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")
        scores, indices = self.index.search(query_embedding, k)
        return scores, indices

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

    @classmethod
    def load(cls, index_path: Path) -> "FaissVectorStore":
        index = faiss.read_index(str(index_path))
        store = cls(index.d)
        store.index = index
        return store


def save_metadata(metadata_df: pd.DataFrame, metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(metadata_path, index=False)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path)