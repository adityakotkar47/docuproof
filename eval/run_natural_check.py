"""Check the refusal gate on questions phrased the way a student would ask (short, vague, hand-written).

    python eval/run_natural_check.py index eval/natural_questions.json eval/results

The gate threshold is chosen on the generated questions; this set was written separately and is used only to test it.
An answerable question counts as answered if the system does not refuse; an unanswerable one counts as handled if it does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docuproof.config import DEFAULT, DEFAULT_GATE  # noqa: E402
from docuproof.index import Embedder, HybridIndex  # noqa: E402
from docuproof.llm import LLM  # noqa: E402
from docuproof.rag import RAG  # noqa: E402
from docuproof.retrieve import Mode, Reranker, Retriever  # noqa: E402

GATES = {"tuned to the generated questions": 1.32, "conservative (default)": DEFAULT_GATE}


def main(index_dir: str, questions_path: str, out_dir: str) -> None:
    qs = json.loads(Path(questions_path).read_text())
    retriever = Retriever(HybridIndex.load(Path(index_dir), Embedder(DEFAULT)), Reranker(DEFAULT))
    llm = LLM("gemini", "gemini-3.5-flash-lite", min_interval=3.0, cache_dir=Path(".llm_cache"))
    results = {}
    for name, gate in GATES.items():
        rag = RAG(retriever, llm, Mode.HYBRID_RERANK, gate=gate)
        rows = []
        for q in qs:
            a = rag.ask(q["question"])
            rows.append({"id": q["id"], "kind": q["kind"], "question": q["question"], "refused": a.refused, "gated": a.gated,
                         "top_score": round(a.hits[0].score, 2), "invalid_citations": a.n_invalid, "answer": a.text[:300]})
            print(name, q["id"], "refused" if a.refused else "answered", flush=True)
        results[name] = {"gate": gate, "rows": rows}
    answerable = [q for q in qs if q["kind"] == "answerable"]
    lines = ["# Refusal gate on natural-phrasing questions", "",
             f"{len(answerable)} answerable and {len(qs) - len(answerable)} unanswerable questions, hand-written and short. Not used to choose the gate.", "",
             "| Gate | Threshold | Answerable wrongly refused | Unanswerable refused (by gate / by model) | LLM calls skipped |", "|---|---|---|---|---|"]
    for name, res in results.items():
        a = [r for r in res["rows"] if r["kind"] == "answerable"]
        u = [r for r in res["rows"] if r["kind"] == "unanswerable"]
        lines.append(f"| {name} | {res['gate']:.2f} | {sum(r['refused'] for r in a)} of {len(a)} | "
                     f"{sum(r['refused'] for r in u)} of {len(u)} ({sum(r['gated'] for r in u)} / {sum(r['refused'] and not r['gated'] for r in u)}) | "
                     f"{sum(r['gated'] for r in res['rows'])} of {len(res['rows'])} |")
    lines += ["", f"LLM requests sent: {llm.calls} (cache hits: {llm.cache_hits})."]
    out = Path(out_dir)
    (out / "natural_check.md").write_text("\n".join(lines) + "\n")
    (out / "natural_check.json").write_text(json.dumps(results, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:4])
