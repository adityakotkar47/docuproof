"""Single-shot grounded question answering with conversation memory and a refusal gate."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .retrieve import Hit, Mode, Retriever

REFUSAL = "INSUFFICIENT_EVIDENCE"

SYSTEM = (
    "You answer questions using ONLY the numbered sources provided. "
    "After every sentence that states a fact, cite the supporting source number in square brackets, "
    "for example [2] or [1][3]. Do not cite a source that does not support the sentence. "
    f"If the sources do not contain the answer, reply with exactly {REFUSAL} and nothing else. "
    "Do not use outside knowledge. Be concise."
)
REWRITE_SYSTEM = (
    "Rewrite the user's last question as a standalone question, using the conversation to resolve "
    "pronouns and references. Reply with the question only."
)
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    number: int
    source: str | None
    page: int | None
    valid: bool


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    citations: list[Citation]
    hits: list[Hit]
    query_used: str
    gated: bool = False          # refused by the retrieval-score gate without calling the LLM
    seconds: float = 0.0

    @property
    def n_invalid(self) -> int:
        return sum(not c.valid for c in self.citations)


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.source}, p.{h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def resolve_citations(text: str, hits: list[Hit]) -> list[Citation]:
    seen: list[int] = []
    for group in _CITE.findall(text):
        for n in re.split(r"\s*,\s*", group):
            if int(n) not in seen:
                seen.append(int(n))
    out = []
    for n in seen:
        ok = 1 <= n <= len(hits)
        out.append(Citation(n, hits[n - 1].chunk.source if ok else None, hits[n - 1].chunk.page if ok else None, ok))
    return out


@dataclass
class Conversation:
    max_turns: int = 6
    turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        self.turns = self.turns[-self.max_turns:]

    def as_messages(self) -> list[dict]:
        msgs: list[dict] = []
        for q, a in self.turns:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return msgs


class RAG:
    def __init__(self, retriever: Retriever, llm, mode: Mode = Mode.HYBRID_RERANK, gate: float | None = None):
        """gate: minimum top reranker score; below it the question is refused without an LLM call."""
        self.retriever, self.llm, self.mode, self.gate = retriever, llm, mode, gate

    def rewrite(self, question: str, conv: Conversation | None) -> str:
        if not conv or not conv.turns:
            return question
        msgs = [{"role": "system", "content": REWRITE_SYSTEM}, *conv.as_messages(),
                {"role": "user", "content": question}]
        return self.llm.chat(msgs, max_tokens=120) or question

    def ask(self, question: str, conv: Conversation | None = None) -> Answer:
        t0 = time.perf_counter()
        query = self.rewrite(question, conv)
        hits = self.retriever.search(query, self.mode)
        if self.gate is not None and self.mode == Mode.HYBRID_RERANK and (not hits or hits[0].score < self.gate):
            ans = Answer(question, REFUSAL, True, [], hits, query, gated=True, seconds=time.perf_counter() - t0)
        else:
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {query}"}]
            text = self.llm.chat(msgs)
            refused = REFUSAL in text
            ans = Answer(question, text, refused, [] if refused else resolve_citations(text, hits), hits, query,
                         seconds=time.perf_counter() - t0)
        if conv is not None:
            conv.add(question, ans.text)
        return ans
