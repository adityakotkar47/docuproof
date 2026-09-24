"""Retrieval modes: dense, BM25, hybrid (reciprocal rank fusion) and hybrid + cross-encoder rerank."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import DEFAULT, Settings
from .index import HybridIndex
from .ingest import Chunk


class Mode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Reranker:
    def __init__(self, settings: Settings = DEFAULT):
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.rerank_model, device="cpu", max_length=512)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts], show_progress_bar=False)]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # rank in each retriever that returned it


class Retriever:
    def __init__(self, index: HybridIndex, reranker: Reranker | None = None, settings: Settings = DEFAULT):
        self.index = index
        self.reranker = reranker
        self.settings = settings

    def search(self, query: str, mode: Mode = Mode.HYBRID_RERANK, k: int | None = None) -> list[Hit]:
        k = k or self.settings.top_k
        pool = max(self.settings.pool, k)
        dense = self.index.dense_search(query, pool) if mode != Mode.BM25 else []
        sparse = self.index.bm25_search(query, pool) if mode != Mode.DENSE else []
        ranks = {"dense": {i: r for r, (i, _) in enumerate(dense, 1)},
                 "bm25": {i: r for r, (i, _) in enumerate(sparse, 1)}}

        if mode == Mode.DENSE:
            ordered = dense
        elif mode == Mode.BM25:
            ordered = sparse
        else:
            ordered = rrf([[i for i, _ in dense], [i for i, _ in sparse]], self.settings.rrf_k)
            if mode == Mode.HYBRID_RERANK:
                if self.reranker is None:
                    raise ValueError("hybrid_rerank needs a reranker")
                cand = [i for i, _ in ordered[:pool]]
                scores = self.reranker.score(query, [self.index.chunks[i].text for i in cand])
                ordered = sorted(zip(cand, scores), key=lambda t: -t[1])

        hits: list[Hit] = []
        per_source: dict[str, int] = {}
        for i, score in ordered:
            chunk = self.index.chunks[i]
            if per_source.get(chunk.source, 0) >= self.settings.max_per_source:
                continue
            per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
            hits.append(Hit(chunk, float(score), {n: r[i] for n, r in ranks.items() if i in r}))
            if len(hits) == k:
                break
        return hits
