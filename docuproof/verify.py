"""Claim auditing with two independent verifiers: an LLM judge and a local NLI model."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, Settings
from .ingest import split_sentences
from .rag import Answer, _CITE

LABELS = ("supported", "partial", "unsupported")

JUDGE_SYSTEM = (
    "You check whether claims are supported by the passages they cite. For each claim give one verdict: "
    "supported (the passages state it), partial (some of it is stated), or unsupported (not stated or contradicted). "
    "Judge only from the passages, not from your own knowledge. "
    'Reply as JSON: {"results": [{"id": 1, "verdict": "supported", "reason": "short reason"}]}.'
)


@dataclass
class Claim:
    text: str
    cites: list[int]


@dataclass
class ClaimAudit:
    claim: Claim
    llm_verdict: str | None
    llm_reason: str
    nli_entailment: float | None
    nli_verdict: str | None


def split_claims(answer_text: str) -> list[Claim]:
    claims = []
    for sent in split_sentences(answer_text):
        cites = sorted({int(n) for g in _CITE.findall(sent) for n in re.split(r"\s*,\s*", g)})
        text = re.sub(r"\s+", " ", _CITE.sub("", sent)).strip()
        if cites and len(re.findall(r"[A-Za-z]{3,}", text)) >= 3:
            claims.append(Claim(text, cites))
    return claims


def parse_verdicts(raw: str, n: int) -> dict[int, tuple[str, str]]:
    try:
        data = json.loads(raw)
        rows = data["results"] if isinstance(data, dict) else data
        out = {}
        for r in rows:
            v = str(r.get("verdict", "")).lower().strip()
            if int(r["id"]) in range(1, n + 1) and v in LABELS:
                out[int(r["id"])] = (v, str(r.get("reason", "")))
        return out
    except (ValueError, KeyError, TypeError, AttributeError):
        return {}


def llm_judge(llm, claims: list[Claim], answer: Answer) -> dict[int, tuple[str, str]]:
    if not claims:
        return {}
    blocks = []
    for i, c in enumerate(claims, 1):
        passages = "\n".join(f"  - {answer.hits[n - 1].chunk.text}" for n in c.cites if 1 <= n <= len(answer.hits))
        blocks.append(f"Claim {i}: {c.text}\nCited passages:\n{passages or '  (none)'}")
    raw = llm.chat([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": "\n\n".join(blocks)}],
                   json_mode=True, max_tokens=900)
    return parse_verdicts(raw, len(claims))


class NLIVerifier:
    """Entailment probability of a claim given the best 3-sentence window of each cited passage."""

    def __init__(self, settings: Settings = DEFAULT, threshold: float = 0.5):
        self.settings, self.threshold, self._model = settings, threshold, None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.nli_model, device="cpu", max_length=512)
        return self._model

    @property
    def entail_index(self) -> int:
        labels = {int(k): str(v).lower() for k, v in self.model.model.config.id2label.items()}
        return next(i for i, name in labels.items() if name.startswith("entail"))

    @staticmethod
    def windows(passage: str, size: int = 3) -> list[str]:
        sents = split_sentences(passage) or [passage]
        return [" ".join(sents[i:i + size]) for i in range(max(1, len(sents) - size + 1))]

    def entailment(self, claim: str, passages: list[str]) -> float:
        pairs = [(w, claim) for p in passages for w in self.windows(p)]
        if not pairs:
            return 0.0
        logits = np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype="float64")
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return float((e / e.sum(axis=1, keepdims=True))[:, self.entail_index].max())


def audit(answer: Answer, llm=None, nli: NLIVerifier | None = None) -> list[ClaimAudit]:
    claims = split_claims(answer.text)
    judged = llm_judge(llm, claims, answer) if llm is not None else {}
    out = []
    for i, c in enumerate(claims, 1):
        verdict, reason = judged.get(i, (None, ""))
        p = v = None
        if nli is not None:
            p = nli.entailment(c.text, [answer.hits[n - 1].chunk.text for n in c.cites if 1 <= n <= len(answer.hits)])
            v = "supported" if p >= nli.threshold else "unsupported"
        out.append(ClaimAudit(c, verdict, reason, p, v))
    return out


def agreement(audits: list[ClaimAudit]) -> dict:
    """How often the two verifiers agree on supported vs not supported."""
    both = [a for a in audits if a.llm_verdict and a.nli_verdict]
    if not both:
        return {"n": 0, "agree": 0.0}
    same = sum((a.llm_verdict == "supported") == (a.nli_verdict == "supported") for a in both)
    return {"n": len(both), "agree": same / len(both)}
