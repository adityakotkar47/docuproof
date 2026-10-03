"""A small function-calling agent: the model decides when to search, read a page or stop."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .rag import REFUSAL
from .retrieve import Mode, Retriever

TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search the document collection and return the most relevant passages with file and page.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A focused search query"},
            "k": {"type": "integer", "description": "Number of passages (1-6)", "default": 4}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_page",
        "description": "Return the full text of one page of one document.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string"}, "page": {"type": "integer"}}, "required": ["source", "page"]}}},
    {"type": "function", "function": {
        "name": "list_sources",
        "description": "List the document names in the collection.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = (
    "You answer questions about a document collection by calling tools. Search before answering; for questions "
    "that combine several documents, search for each part separately. Use only what the tools return. "
    "Cite every fact as [source, p.N] using the file name and page from the tool results. "
    f"If the tools do not provide the answer, reply with exactly {REFUSAL}."
)
_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_SRC_PAGES = re.compile(r"([A-Za-z_][\w\-]*)\s*,\s*pp?\.\s*(\d+(?:\s*,\s*\d+)*)")


def parse_citations(text: str) -> list[tuple[str, int]]:
    """(source, page) pairs from citations such as [rag, p. 3], [rag, p. 3, 4] or [rag, p. 3, gan, p. 5]."""
    out: list[tuple[str, int]] = []
    for group in _BRACKET.findall(text):
        for source, pages in _SRC_PAGES.findall(group):
            for page in re.split(r"\s*,\s*", pages):
                if (source, int(page)) not in out:
                    out.append((source, int(page)))
    return out


@dataclass
class Step:
    tool: str
    arguments: dict
    result_chars: int


@dataclass
class AgentResult:
    question: str
    text: str
    refused: bool
    steps: list[Step] = field(default_factory=list)
    seen: set[tuple[str, int]] = field(default_factory=set)   # (source, page) pairs returned by tools
    cited: list[tuple[str, int, bool]] = field(default_factory=list)  # (source, page, valid)
    finished: bool = True


class Agent:
    def __init__(self, llm, retriever: Retriever, max_steps: int = 6, mode: Mode = Mode.HYBRID_RERANK):
        self.llm, self.retriever, self.max_steps, self.mode = llm, retriever, max_steps, mode

    # -- tools -------------------------------------------------------------
    def _search(self, query: str, k: int = 4, seen: set | None = None) -> str:
        hits = self.retriever.search(query, self.mode, k=max(1, min(int(k), 6)))
        rows = []
        for h in hits:
            if seen is not None:
                seen.add((h.chunk.source, h.chunk.page))
            rows.append(f"({h.chunk.source}, p.{h.chunk.page}) {h.chunk.text}")
        return "\n\n".join(rows) or "No results."

    def _read(self, source: str, page: int, seen: set | None = None) -> str:
        text = self.retriever.index.page_text(source, int(page))
        if text and seen is not None:
            seen.add((source, int(page)))
        return text[:3500] or "No such page."

    def _dispatch(self, name: str, args: dict, seen: set) -> str:
        try:
            if name == "search_documents":
                return self._search(args["query"], args.get("k", 4), seen)
            if name == "read_page":
                return self._read(args["source"], args["page"], seen)
            if name == "list_sources":
                return ", ".join(self.retriever.index.sources)
        except (KeyError, TypeError, ValueError) as exc:
            return f"Bad arguments: {exc}"
        return f"Unknown tool {name}"

    # -- loop --------------------------------------------------------------
    def run(self, question: str) -> AgentResult:
        messages: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        result = AgentResult(question, "", False)
        for _ in range(self.max_steps):
            msg = self.llm.chat_tools(messages, TOOLS)
            calls = msg.tool_calls or []
            if not calls:
                result.text = (msg.content or "").strip()
                break
            messages.append(msg.model_dump(exclude_none=True))   # keep provider extras (e.g. thought signatures)
            for call in calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = self._dispatch(call.function.name, args, result.seen)
                result.steps.append(Step(call.function.name, args, len(out)))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
        else:
            result.finished = False
            result.text = "(step limit reached)"
        result.refused = REFUSAL in result.text
        for src, page in parse_citations(result.text):
            result.cited.append((src, page, (src, page) in result.seen))
        return result
