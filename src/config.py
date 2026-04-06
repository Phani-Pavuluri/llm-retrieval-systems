from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load OPENAI_API_KEY (and other vars) from project-root `.env` when `python-dotenv` is installed.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

RAW_FILE_NAME = "amazon_com-product_reviews__20200101_20200331_sample.csv"  # update this
MIN_REVIEW_LENGTH = 50
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K = 5

# "sentence_transformers" | "openai"
EMBEDDING_BACKEND = "sentence_transformers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Used when EMBEDDING_BACKEND == "openai" (also set OPENAI_API_KEY).
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"