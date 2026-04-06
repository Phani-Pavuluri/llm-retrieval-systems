import pandas as pd
from src.config import RAW_DATA_DIR, RAW_FILE_NAME, MIN_REVIEW_LENGTH


def load_reviews() -> pd.DataFrame:
    file_path = RAW_DATA_DIR / RAW_FILE_NAME
    df = pd.read_csv(file_path)

    required_cols = [
        "Asin",
        "Review Content",
        "Review Rating",
        "Review Title",
        "Brand",
        "Category",
        "Sub Category",
        "Product Description",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {df.columns.tolist()}"
        )

    df = df[required_cols].copy()

    df = df.dropna(subset=["Asin", "Review Content", "Review Rating"])
    df["Review Content"] = df["Review Content"].astype(str).str.strip()
    df = df[df["Review Content"].str.len() >= MIN_REVIEW_LENGTH].reset_index(drop=True)

    df = df.rename(
        columns={
            "Asin": "asin",
            "Review Content": "review_text",
            "Review Rating": "review_rating",
            "Review Title": "review_title",
            "Brand": "brand",
            "Category": "category",
            "Sub Category": "sub_category",
            "Product Description": "product_description",
        }
    )

    return df