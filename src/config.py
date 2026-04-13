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

# LLM generation (RAG answer step): "openai" | "ollama"
LLM_BACKEND = "ollama"
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "llama3"

# Optional hybrid retrieval: lexical overlap on vector candidates, then weighted fuse.
HYBRID_RETRIEVAL = False
HYBRID_ALPHA = 0.7  # weight on semantic (FAISS) score after batch min-max norm
HYBRID_BETA = 0.3  # weight on keyword overlap score after batch min-max norm

# Phase 1 observability: set RETRIEVAL_TRACE_ENABLED True to append JSONL traces on each retrieve()
RETRIEVAL_TRACE_ENABLED = False
RETRIEVAL_TRACE_DIR = PROJECT_ROOT / "artifacts" / "retrieval_traces"

# Phase 4: append JSONL lines per RAG answer (prompt template, chunks, LLM, answer).
ANSWER_TRACE_ENABLED = False
ANSWER_TRACE_DIR = PROJECT_ROOT / "artifacts" / "answer_traces"

# Phase 2 candidate pool: k_fetch = min(ntotal, max(ceil(top_k * mult * pool_mult), ceil(min_cand * pool_mult)))
RETRIEVAL_OVERSAMPLE_MULTIPLIER = 5
RETRIEVAL_MIN_CANDIDATES = 20

# Phase 3 optional cross-encoder rerank (local, no API)
RERANK_ENABLED = False
# Default / baseline cross-encoder (experiments compare other models to this).
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Explicit alias for reports (kept equal to RERANK_MODEL unless you fork config).
RERANK_BASELINE_MODEL = RERANK_MODEL
# Suggested candidates for scripts/compare_reranker_models.py (override via CLI --models).
RERANK_MODEL_CANDIDATES = (
    RERANK_MODEL,
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-electra-base",
)
# Conservative pool for second-stage ranker (was 25)
RERANK_TOP_N = 12
RERANK_BATCH_SIZE = 16
# When True and request.use_rerank is None, apply_selective_rerank_policy() sets use_rerank.
RERANK_SELECTIVE = True
# If set, skip reranking when top-1 fused score is already this high (hybrid frames only).
RERANK_SKIP_IF_TOP_SCORE_AT_LEAST = 0.97
# Confidence guard uses hybrid-style frames (semantic_score column present).
RERANK_CONFIDENCE_REQUIRES_HYBRID_SCORES = True