"""Command line: build an index from a folder of documents, or ask a question.

    python -m docuproof.cli build data/corpus index/
    python -m docuproof.cli ask index/ "What is the generator trained to do in a GAN?"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import DEFAULT
from .index import Embedder, HybridIndex, IndexNotBuiltError
from .ingest import ingest_folder
from .llm import LLM
from .rag import RAG
from .retrieve import Mode, Reranker, Retriever
from .verify import NLIVerifier, audit


def load_index(folder: Path) -> HybridIndex:
    return HybridIndex.load(folder, Embedder(DEFAULT))


def build(args) -> None:
    t0 = time.perf_counter()
    chunks = ingest_folder(Path(args.docs))
    print(f"{len(chunks)} chunks from {len({c.source for c in chunks})} documents ({time.perf_counter() - t0:.0f}s)")
    index = HybridIndex.build(chunks, Embedder(DEFAULT))
    index.save(Path(args.out))
    print(f"saved index to {args.out} ({time.perf_counter() - t0:.0f}s total)")


def ask(args) -> None:
    try:
        index = load_index(Path(args.index))
    except IndexNotBuiltError as exc:
        raise SystemExit(str(exc))
    llm = LLM(args.provider, args.model)
    rag = RAG(Retriever(index, Reranker(DEFAULT)), llm, Mode(args.mode))
    answer = rag.ask(args.question)
    print(answer.text, "\n")
    for c in answer.citations:
        print(f"  [{c.number}] {c.source} p.{c.page}" if c.valid else f"  [{c.number}] INVALID")
    if args.audit:
        for a in audit(answer, llm, NLIVerifier(DEFAULT)):
            print(f"  claim: {a.claim.text[:80]!r} llm={a.llm_verdict} nli={a.nli_entailment and round(a.nli_entailment, 2)}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="docuproof")
    sub = ap.add_subparsers(required=True)
    b = sub.add_parser("build")
    b.add_argument("docs")
    b.add_argument("out")
    b.set_defaults(fn=build)
    a = sub.add_parser("ask")
    a.add_argument("index")
    a.add_argument("question")
    a.add_argument("--provider", default="gemini")
    a.add_argument("--model")
    a.add_argument("--mode", default=Mode.HYBRID_RERANK.value, choices=[m.value for m in Mode])
    a.add_argument("--audit", action="store_true")
    a.set_defaults(fn=ask)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
