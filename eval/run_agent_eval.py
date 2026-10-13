"""Compare the function-calling agent with single-shot RAG on multi-hop questions.

    python eval/run_agent_eval.py index eval/questions.json eval/results --gate -3.2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import grade_correct, load_questions, rate, targets

from docuproof.agent import Agent
from docuproof.config import DEFAULT
from docuproof.index import Embedder, HybridIndex
from docuproof.llm import LLM
from docuproof.metrics import is_relevant
from docuproof.rag import RAG
from docuproof.retrieve import Mode, Reranker, Retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qs = load_questions(args.questions, {"multi"})[: args.limit]
    index = HybridIndex.load(Path(args.index), Embedder(DEFAULT))
    retriever = Retriever(index, Reranker(DEFAULT))
    llm = LLM("gemini", args.model, min_interval=3.0, cache_dir=Path(".llm_cache"))
    rag, agent = RAG(retriever, llm, gate=args.gate), Agent(llm, retriever)

    rows = []
    for n, q in enumerate(qs, 1):
        ts = targets(q)
        evidence = [t.phrase for t in ts]
        one = rag.ask(q["question"])
        one_ok = "refused" if one.refused else grade_correct(llm, q["question"], evidence, one.text)
        one_gold = sum(any(is_relevant(h.chunk, [t]) for h in one.hits) for t in ts) / len(ts)
        ag = agent.run(q["question"])
        ag_ok = "refused" if ag.refused else grade_correct(llm, q["question"], evidence, ag.text)
        seen_pages = {(t.source, t.page) for t in ts}
        ag_gold = len(seen_pages & ag.seen) / len(seen_pages)
        rows.append({"id": q["id"], "question": q["question"], "single_correct": one_ok, "single_gold_recall": one_gold,
                     "agent_correct": ag_ok, "agent_gold_recall": ag_gold, "agent_steps": len(ag.steps),
                     "agent_tools": [s.tool for s in ag.steps], "agent_finished": ag.finished,
                     "agent_invalid_citations": sum(not c[2] for c in ag.cited), "agent_answer": ag.text, "single_answer": one.text})
        print(f"{n}/{len(qs)} single={one_ok} agent={ag_ok} steps={len(ag.steps)}", flush=True)
        (out / "agent_eval_rows.json").write_text(json.dumps(rows, indent=1))

    lines = ["# Agent vs single-shot RAG on multi-hop questions", "",
             f"Model: {args.model}. {len(rows)} multi-hop questions, one run. Correctness graded by an LLM against the evidence spans.", "",
             "| Measure | Single-shot RAG | Function-calling agent |", "|---|---|---|",
             f"| correct | {rate([r['single_correct'] == 'correct' for r in rows])} | {rate([r['agent_correct'] == 'correct' for r in rows])} |",
             f"| correct or partial | {rate([r['single_correct'] in ('correct', 'partial') for r in rows])} | {rate([r['agent_correct'] in ('correct', 'partial') for r in rows])} |",
             f"| gold passages reached | {rate([r['single_gold_recall'] for r in rows])} | {rate([r['agent_gold_recall'] for r in rows])} |",
             f"| mean tool calls | 0 (1 retrieval) | {sum(r['agent_steps'] for r in rows) / len(rows):.1f} |",
             f"| agent hit the step limit | - | {sum(not r['agent_finished'] for r in rows)}/{len(rows)} |",
             f"| agent citations to pages it never saw | - | {sum(r['agent_invalid_citations'] for r in rows)} |",
             "", f"LLM requests sent: {llm.calls} (cache hits: {llm.cache_hits}).",
             "Note: gold passages reached for the agent is measured at page level (the tool results it saw), for single-shot at chunk level."]
    (out / "agent_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
