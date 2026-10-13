"""Collect every headline number into eval/results/summary.json (read by the README and the slides).

    python eval/summarize.py index eval/questions.json eval/results
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docuproof.metrics import bootstrap_ci  # noqa: E402


def ci(values) -> list | None:
    values = [float(v) for v in values]
    return [round(x, 4) for x in bootstrap_ci(np.array(values))] + [len(values)] if values else None


def main(index_dir: str, questions_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    chunks = [json.loads(l) for l in (Path(index_dir) / "chunks.jsonl").read_text().splitlines()]
    questions = json.loads(Path(questions_path).read_text())
    summary: dict = {
        "corpus": {"chunks": len(chunks), "papers": len({c["source"] for c in chunks}), "pages": len({(c["source"], c["page"]) for c in chunks})},
        "questions": dict(Counter(q["kind"] for q in questions)),
    }
    ret = out / "retrieval_results.json"
    if ret.exists():
        data = json.loads(ret.read_text())
        summary["retrieval"] = data["summary"]
        summary["gate"] = data["gate"]
        summary["scores"] = {"answerable_min": min(data["scores_answerable"]), "unanswerable_max": max(data["scores_unanswerable"])}
    nat = out / "natural_check.json"
    if nat.exists():
        summary["natural"] = {}
        for name, res in json.loads(nat.read_text()).items():
            a = [r for r in res["rows"] if r["kind"] == "answerable"]
            u = [r for r in res["rows"] if r["kind"] == "unanswerable"]
            summary["natural"][name] = {"gate": res["gate"], "answerable": len(a), "wrongly_refused": sum(r["refused"] for r in a),
                                        "unanswerable": len(u), "refused": sum(r["refused"] for r in u),
                                        "refused_by_gate": sum(r["gated"] for r in u)}
    rows_p = out / "answer_eval_rows.json"
    if rows_p.exists():
        rows = json.loads(rows_p.read_text())
        ans = [r for r in rows if r["kind"] != "unanswerable"]
        un = [r for r in rows if r["kind"] == "unanswerable"]
        a: dict = {"n_answerable": len(ans), "n_unanswerable": len(un)}
        for kind in ("single", "multi"):
            sub = [r for r in ans if r["kind"] == kind]
            if sub:
                a[kind] = {"n": len(sub), "correct": ci(r["correct"] == "correct" for r in sub),
                           "correct_or_partial": ci(r["correct"] in ("correct", "partial") for r in sub),
                           "wrongly_refused": sum(r["correct"] == "refused" for r in sub),
                           "gold_in_context": ci(r["gold_in_context"] for r in sub),
                           "cites_gold": ci(r["cites_gold"] for r in sub if not r["refused"])}
        a["unanswerable_refused"] = ci(r["refused"] for r in un)
        a["unanswerable_gated"] = sum(r["gated"] for r in un)
        answered = [r for r in rows if not r["refused"]]
        a["no_invalid_citation"] = ci(r["invalid_citations"] == 0 for r in answered)
        claims = [c for r in rows for c in r.get("claims", [])]
        a["claims"] = len(claims)
        a["llm_supported"] = ci(c["llm"] == "supported" for c in claims if c["llm"])
        a["nli_supported"] = ci(c["nli"] >= 0.5 for c in claims if c["nli"] is not None)
        both = [(c["llm"] == "supported", c["nli"] >= 0.5) for c in claims if c["llm"] and c["nli"] is not None]
        a["agreement"] = ci(x == y for x, y in both)
        a["confusion"] = {"both_supported": sum(x and y for x, y in both), "llm_only": sum(x and not y for x, y in both),
                          "nli_only": sum((not x) and y for x, y in both), "neither": sum((not x) and (not y) for x, y in both)}
        summary["answers"] = a
    ag_p = out / "agent_eval_rows.json"
    if ag_p.exists():
        rows = json.loads(ag_p.read_text())
        summary["agent"] = {
            "n": len(rows),
            "single_correct": ci(r["single_correct"] == "correct" for r in rows),
            "agent_correct": ci(r["agent_correct"] == "correct" for r in rows),
            "single_gold": ci(r["single_gold_recall"] for r in rows),
            "agent_gold": ci(r["agent_gold_recall"] for r in rows),
            "mean_tool_calls": round(sum(r["agent_steps"] for r in rows) / len(rows), 2),
            "hit_step_limit": sum(not r["agent_finished"] for r in rows),
            "invalid_citations": sum(r["agent_invalid_citations"] for r in rows),
        }
        refused = [r for r in rows if r["single_correct"] == "refused"]
        summary["auto"] = {  # single-shot first, escalate to the agent when it refuses
            "n": len(rows),
            "correct": sum(r["single_correct"] == "correct" or (r["single_correct"] == "refused" and r["agent_correct"] == "correct") for r in rows),
            "single_refused": len(refused),
            "agent_correct_on_refused": sum(r["agent_correct"] == "correct" for r in refused),
            "tool_calls_auto": sum(r["agent_steps"] for r in refused),
            "tool_calls_always_agent": sum(r["agent_steps"] for r in rows),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:3000])


if __name__ == "__main__":
    main(*sys.argv[1:4])
