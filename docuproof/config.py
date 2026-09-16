"""Settings and LLM provider presets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # models (all run locally on CPU)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    # chunking (counted in words; chunks never cross a page boundary)
    chunk_words: int = 180
    chunk_overlap_words: int = 40
    # retrieval
    pool: int = 20            # candidates taken from each retriever before fusion
    top_k: int = 5            # passages passed to the language model
    rrf_k: int = 60
    max_per_source: int = 3   # keep one paper from filling the whole context


# OpenAI-compatible endpoints (any other provider can be added the same way)
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.5-flash-lite",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "minimax/minimax-m3.1-flash:free",
    },
}

DEFAULT = Settings()

# Refusal gate: questions whose best reranker score is below this are refused without calling the language model.
# Chosen as (lowest score of any answerable question in the calibration set) - 1, so it only blocks clearly off-topic
# questions; borderline ones are left to the model, which can still answer INSUFFICIENT_EVIDENCE.
DEFAULT_GATE = -3.0
