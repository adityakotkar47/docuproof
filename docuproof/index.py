"""Dense (FAISS) and sparse (BM25) indexes over chunks, with on-disk persistence."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import DEFAULT, Settings
from .ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = frozenset("a an the of in on at to for and or is are was were be been by with as from that this it its which what who how".split())


class IndexNotBuiltError(FileNotFoundError):
    def __init__(self, folder: Path):
        super().__init__(
            f"Index at {folder} was not found. Create it with:\n"
            "uv run python scripts/fetch_corpus.py\n"
            "uv run python -m docuproof.cli build data/corpus index"
        )


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps codes such as 'e-4021' and 'gpt-3' as single tokens."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Embedder:
    """Thin wrapper around a sentence-transformers model (loaded on first use)."""

    def __init__(self, settings: Settings = DEFAULT):
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embed_model, device="cpu")
        return self._model

    def passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                            show_progress_bar=False), dtype="float32")

    def query(self, text: str) -> np.ndarray:
        return self.passages([self.settings.query_prefix + text])[0]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder):
        import faiss
        from rank_bm25 import BM25Okapi

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must align")
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.by_id = {c.id: i for i, c in enumerate(chunks)}
        self.faiss = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss.add(embeddings)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "HybridIndex":
        return cls(chunks, embedder.passages([c.text for c in chunks]), embedder)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "embeddings.npy", self.embeddings)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps({"id": c.id, "source": c.source, "page": c.page, "text": c.text}) + "\n")

    @classmethod
    def load(cls, folder: Path, embedder: Embedder) -> "HybridIndex":
        if not (folder / "chunks.jsonl").is_file() or not (folder / "embeddings.npy").is_file():
            raise IndexNotBuiltError(folder)
        rows = [json.loads(l) for l in (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        return cls([Chunk(**r) for r in rows], np.load(folder / "embeddings.npy"), embedder)

    # -- search ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores, idx = self.faiss.search(self.embedder.query(query)[None, :], min(k, len(self)))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def page_text(self, source: str, page: int) -> str:
        parts = [c.text for c in self.chunks if c.source == source and c.page == page]
        return " ".join(parts)
