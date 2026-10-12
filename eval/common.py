"""Shared helpers for the answer-level evaluations."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docuproof.metrics import Target, bootstrap_ci  # noqa: E402

CORRECT_SYSTEM = (
    "You grade an answer to a question against the evidence it should be based on. "
    "correct: the answer states what the evidence states (extra detail is fine). "
    "partial: it states some of it, or is vague. incorrect: it contradicts, misses, or invents the key fact. "
    'Reply as JSON: {"verdict": "correct|partial|incorrect", "reason": "short"}.'
)


def load_questions(path: str, kinds: set[str]) -> list[dict]:
    return [q for q in json.loads(Path(path).read_text()) if q["kind"] in kinds]


def pick(questions: list[dict], kind: str, n: int, seed: int = 2026) -> list[dict]:
    import random

    pool = [q for q in questions if q["kind"] == kind]
    return random.Random(seed).sample(pool, min(n, len(pool)))


def grade_correct(llm, question: str, evidence: list[str], answer: str) -> str:
    body = f"Question: {question}\n\nEvidence:\n" + "\n".join(f"- {e}" for e in evidence) + f"\n\nAnswer to grade:\n{answer}"
    raw = llm.chat([{"role": "system", "content": CORRECT_SYSTEM}, {"role": "user", "content": body}], json_mode=True, max_tokens=200)
    try:
        v = str(json.loads(raw).get("verdict", "")).lower()
    except ValueError:
        m = re.search(r"correct|partial|incorrect", raw.lower())
        v = m.group(0) if m else ""
    return v if v in {"correct", "partial", "incorrect"} else "incorrect"


def rate(values: list[float]) -> str:
    if not values:
        return "n/a"
    m, lo, hi = bootstrap_ci(__import__("numpy").array(values, dtype=float))
    return f"{m:.2f} [{lo:.2f}-{hi:.2f}] (n={len(values)})"


def targets(q: dict) -> list[Target]:
    return [Target(**t) for t in q["targets"]]
