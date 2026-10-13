"""Answer-level evaluation of the single-shot RAG pipeline.

    python eval/run_answer_eval.py index eval/questions.json eval/results --gate -3.0 --model gemini-3.5-flash-lite

Per question: answered or refused, correctness (LLM grader against the evidence spans), citation validity, whether a cited
passage is a gold passage, and a claim audit by two verifiers (LLM judge and a local NLI model).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import grade_correct, load_questions, pick, rate, targets

from docuproof.config import DEFAULT, DEFAULT_GATE
from docuproof.index import Embedder, HybridIndex
from docuproof.llm import LLM
from docuproof.metrics import is_relevant
from docuproof.rag import RAG
from docuproof.retrieve import Mode, Reranker, Retriever
from docuproof.verify import NLIVerifier, agreement, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--gate", type=float, default=DEFAULT_GATE)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--n-single", type=int, default=30)
    ap.add_argument("--n-multi", type=int, default=10)
    ap.add_argument("--n-unans", type=int, default=20)
    ap.add_argument("--no-nli", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qs = load_questions(args.questions, {"single", "multi", "unanswerable"})
    chosen = pick(qs, "single", args.n_single) + pick(qs, "multi", args.n_multi) + pick(qs, "unanswerable", args.n_unans)
    index = HybridIndex.load(Path(args.index), Embedder(DEFAULT))
    llm = LLM(args.provider, args.model, min_interval=3.0, cache_dir=Path(".llm_cache"))
    rag = RAG(Retriever(index, Reranker(DEFAULT)), llm, Mode.HYBRID_RERANK, gate=args.gate)
    nli = None if args.no_nli else NLIVerifier(DEFAULT)

    rows = []
    for n, q in enumerate(chosen, 1):
        ans = rag.ask(q["question"])
        row = {"id": q["id"], "kind": q["kind"], "question": q["question"], "answer": ans.text, "refused": ans.refused,
               "gated": ans.gated, "invalid_citations": ans.n_invalid, "n_citations": len(ans.citations), "seconds": round(ans.seconds, 2)}
        if q["kind"] != "unanswerable":
            ts = targets(q)
            row["cites_gold"] = any(is_relevant(ans.hits[c.number - 1].chunk, ts) for c in ans.citations if c.valid)
            row["gold_in_context"] = any(is_relevant(h.chunk, ts) for h in ans.hits)
            row["correct"] = "refused" if ans.refused else grade_correct(llm, q["question"], [t.phrase for t in ts], ans.text)
        if not ans.refused and ans.citations:
            audits = audit(ans, llm, nli)
            row["claims"] = [{"text": a.claim.text, "llm": a.llm_verdict, "nli": a.nli_entailment} for a in audits]
            row["agreement"] = agreement(audits)
        rows.append(row)
        print(f"{n}/{len(chosen)} {q['kind']:<12} refused={ans.refused} correct={row.get('correct')}", flush=True)
        (out / "answer_eval_rows.json").write_text(json.dumps(rows, indent=1))

    ans_rows = [r for r in rows if r["kind"] != "unanswerable"]
    un_rows = [r for r in rows if r["kind"] == "unanswerable"]
    lines = ["# Answer-level results", "", f"Model: {args.model}. Retrieval: hybrid + rerank, top {DEFAULT.top_k}. Gate: {args.gate}.",
             f"{len(ans_rows)} answerable ({sum(r['kind']=='single' for r in rows)} single-hop, {sum(r['kind']=='multi' for r in rows)} multi-hop), "
             f"{len(un_rows)} unanswerable. One run; 95% bootstrap intervals.", "",
             "| Measure | Value |", "|---|---|"]
    for kind in ("single", "multi"):
        sub = [r for r in ans_rows if r["kind"] == kind]
        if sub:
            c = Counter(r["correct"] for r in sub)
            lines += [f"| {kind}: correct | {rate([r['correct'] == 'correct' for r in sub])} |",
                      f"| {kind}: correct or partial | {rate([r['correct'] in ('correct', 'partial') for r in sub])} |",
                      f"| {kind}: wrongly refused | {c.get('refused', 0)}/{len(sub)} |",
                      f"| {kind}: gold passage in context | {rate([r['gold_in_context'] for r in sub])} |",
                      f"| {kind}: cites a gold passage | {rate([r['cites_gold'] for r in sub if not r['refused']])} |"]
    answered = [r for r in rows if not r["refused"]]
    lines += [f"| unanswerable: correctly refused | {rate([r['refused'] for r in un_rows])} |",
              f"| unanswerable: refused by gate (no LLM call) | {sum(r['gated'] for r in un_rows)}/{len(un_rows)} |",
              f"| answers with no invalid citation | {rate([r['invalid_citations'] == 0 for r in answered])} |"]
    claims = [c for r in rows for c in r.get("claims", [])]
    if claims:
        llm_ok = [c["llm"] == "supported" for c in claims if c["llm"]]
        nli_ok = [c["nli"] >= 0.5 for c in claims if c["nli"] is not None]
        lines += [f"| claims judged supported by LLM judge | {rate(llm_ok)} |", f"| claims judged supported by NLI model | {rate(nli_ok)} |"]
        both = [(c["llm"] == "supported", c["nli"] >= 0.5) for c in claims if c["llm"] and c["nli"] is not None]
        if both:
            agree = [a == b for a, b in both]
            lines.append(f"| verifier agreement (supported vs not) | {rate(agree)} |")
            lines += ["", "Verifier confusion (rows: LLM judge, columns: NLI):", "", "| | NLI supported | NLI not supported |", "|---|---|---|",
                      f"| LLM supported | {sum(a and b for a, b in both)} | {sum(a and not b for a, b in both)} |",
                      f"| LLM not supported | {sum((not a) and b for a, b in both)} | {sum((not a) and (not b) for a, b in both)} |"]
    lines += ["", f"LLM requests sent: {llm.calls} (cache hits: {llm.cache_hits})."]
    (out / "answer_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
