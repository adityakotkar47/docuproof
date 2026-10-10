"""Retrieval metrics, bootstrap confidence intervals and a paired exact test."""
from __future__ import annotations

import re
from dataclasses import dataclass
from math import comb

import numpy as np


def norm(text: str) -> str:
    """Lowercase, remove spaces and punctuation so a label survives line-break and hyphenation noise."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@dataclass(frozen=True)
class Target:
    source: str
    page: int
    phrase: str


def is_relevant(chunk, targets: list[Target]) -> bool:
    text = norm(chunk.text)
    return any(chunk.source == t.source and chunk.page == t.page and norm(t.phrase) in text for t in targets)


def first_rank(chunks: list, targets: list[Target]) -> int | None:
    for rank, c in enumerate(chunks, start=1):
        if is_relevant(c, targets):
            return rank
    return None


def target_recall(chunks: list, targets: list[Target]) -> float:
    """Share of the question's targets found anywhere in `chunks` (for multi-hop questions)."""
    if not targets:
        return 0.0
    return sum(is_relevant_any(chunks, t) for t in targets) / len(targets)


def is_relevant_any(chunks: list, target: Target) -> bool:
    return any(is_relevant(c, [target]) for c in chunks)


def hit_at(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([1.0 if r is not None and r <= k else 0.0 for r in ranks])


def mrr_at(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([1.0 / r if r is not None and r <= k else 0.0 for r in ranks])


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 2026, alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 2026) -> tuple[float, float, float]:
    """Mean of (b - a) with a 95% bootstrap interval over the same questions."""
    return bootstrap_ci(np.asarray(b, float) - np.asarray(a, float), n_boot, seed)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)
