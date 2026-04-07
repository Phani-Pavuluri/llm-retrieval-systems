import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from src.data_loader import load_reviews
from src.chunking import chunk_text
from src.config import PROCESSED_DATA_DIR


def main() -> None:
    df = load_reviews()

    records = []

    for row_idx, row in df.iterrows():
        chunks = chunk_text(row["review_text"])

        for chunk_idx, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"{row['asin']}_{row_idx}_{chunk_idx}",
                    "asin": row["asin"],
                    "review_rating": row["review_rating"],
                    "brand": row["brand"],
                    "category": row["category"],
                    "sub_category": row["sub_category"],
                    "review_title": row["review_title"],
                    "text": chunk,
                }
            )

    chunks_df = pd.DataFrame(records)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "review_chunks.csv"
    chunks_df.to_csv(output_path, index=False)

    print(f"Saved {len(chunks_df)} chunks to {output_path}")
    print(chunks_df.head())


if __name__ == "__main__":
    main()