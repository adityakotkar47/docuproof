"""Retrieval evaluation: dense vs BM25 vs hybrid vs hybrid + rerank, plus calibration of the refusal gate.

    python eval/run_retrieval_eval.py index eval/questions.json eval/results

Single-hop: Hit@k, MRR@k. Multi-hop: share of the question's evidence passages found in the top k (and whether all
were found). Unanswerable questions are used to pick the reranker-score threshold below which the system refuses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docuproof.config import DEFAULT  # noqa: E402
from docuproof.index import Embedder, HybridIndex  # noqa: E402
from docuproof.metrics import (Target, bootstrap_ci, first_rank, hit_at, mcnemar_exact, mrr_at,  # noqa: E402
                               paired_bootstrap, target_recall)
from docuproof.retrieve import Mode, Reranker, Retriever  # noqa: E402

K = 5
MODES = [Mode.DENSE, Mode.BM25, Mode.HYBRID, Mode.HYBRID_RERANK]


def fmt(ci: tuple[float, float, float]) -> str:
    return f"{ci[0]:.2f} [{ci[1]:.2f}-{ci[2]:.2f}]"


def calibrate_gate(answerable: list[float], unanswerable: list[float]) -> dict:
    """Two ways to pick the threshold on the top reranker score.

    youden:       maximises (answerable kept + unanswerable refused - 1). Looks best here but overfits: on short hand-written
                  questions it wrongly refused a third of answerable ones (see run_natural_check.py).
    conservative: lowest answerable score minus 1. Blocks only clearly off-topic questions and leaves borderline ones to the
                  model, which can still answer INSUFFICIENT_EVIDENCE. This is the default (DEFAULT_GATE).
    """
    best = {"threshold": None, "j": -1.0}
    for t in sorted(set(answerable + unanswerable)):
        kept = float(np.mean([s >= t for s in answerable]))
        refused = float(np.mean([s < t for s in unanswerable]))
        if kept + refused - 1 > best["j"]:
            best = {"threshold": float(t), "j": kept + refused - 1, "answerable_kept": kept, "unanswerable_refused": refused}
    cons = min(answerable) - 1.0
    best["conservative"] = {"threshold": float(cons), "answerable_kept": float(np.mean([s >= cons for s in answerable])),
                            "unanswerable_refused": float(np.mean([s < cons for s in unanswerable]))}
    return best


def main(index_dir: str, questions_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    questions = json.loads(Path(questions_path).read_text())
    index = HybridIndex.load(Path(index_dir), Embedder(DEFAULT))
    retriever = Retriever(index, Reranker(DEFAULT))
    single = [q for q in questions if q["kind"] == "single"]
    multi = [q for q in questions if q["kind"] == "multi"]
    unans = [q for q in questions if q["kind"] == "unanswerable"]
    targets = lambda q: [Target(**t) for t in q["targets"]]  # noqa: E731

    ranks: dict[str, list] = {}
    recalls: dict[str, np.ndarray] = {}
    allfound: dict[str, np.ndarray] = {}
    for mode in MODES:
        ranks[mode.value] = [first_rank([h.chunk for h in retriever.search(q["question"], mode, k=K)], targets(q)) for q in single]
        recs = [target_recall([h.chunk for h in retriever.search(q["question"], mode, k=K)], targets(q)) for q in multi]
        recalls[mode.value] = np.array(recs)
        allfound[mode.value] = np.array([1.0 if r == 1.0 else 0.0 for r in recs])
        print("done", mode.value, flush=True)

    lines = ["# Retrieval results", "",
             f"Corpus: {len(index)} chunks from {len(index.sources)} papers. Questions: {len(single)} single-hop, "
             f"{len(multi)} multi-hop, {len(unans)} unanswerable. Top-{K} passages. 95% bootstrap intervals (10,000 resamples).", "",
             "## Single-hop", "", "| Mode | Hit@1 | Hit@5 | MRR@5 |", "|---|---|---|---|"]
    summary = {"single": {}, "multi": {}}
    for m in MODES:
        r = ranks[m.value]
        row = {"hit1": bootstrap_ci(hit_at(r, 1)), "hit5": bootstrap_ci(hit_at(r, K)), "mrr5": bootstrap_ci(mrr_at(r, K))}
        summary["single"][m.value] = row
        lines.append(f"| {m.value} | {fmt(row['hit1'])} | {fmt(row['hit5'])} | {fmt(row['mrr5'])} |")

    lines += ["", "### Paired comparison on Hit@5 (same questions)", "",
              "| Baseline -> challenger | Difference [95% CI] | Only baseline hits / only challenger hits | McNemar p |", "|---|---|---|---|"]
    for a, b in [(Mode.DENSE, Mode.HYBRID), (Mode.BM25, Mode.HYBRID), (Mode.HYBRID, Mode.HYBRID_RERANK),
                 (Mode.DENSE, Mode.HYBRID_RERANK), (Mode.BM25, Mode.HYBRID_RERANK)]:
        ha, hb = hit_at(ranks[a.value], K), hit_at(ranks[b.value], K)
        oa, ob = int(((ha == 1) & (hb == 0)).sum()), int(((ha == 0) & (hb == 1)).sum())
        d = paired_bootstrap(ha, hb)
        lines.append(f"| {a.value} -> {b.value} | {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] | {oa} / {ob} | {mcnemar_exact(oa, ob):.3f} |")

    lines += ["", "## Multi-hop (two evidence passages from two papers)", "",
              f"| Mode | Evidence recall@{K} | Both passages found |", "|---|---|---|"]
    for m in MODES:
        r, a = bootstrap_ci(recalls[m.value]), bootstrap_ci(allfound[m.value])
        summary["multi"][m.value] = {"recall": r, "all": a}
        lines.append(f"| {m.value} | {fmt(r)} | {fmt(a)} |")

    # refusal gate calibration on the reranker's top score
    top = lambda qs: [retriever.search(q["question"], Mode.HYBRID_RERANK, k=1)[0].score for q in qs]  # noqa: E731
    s_ans, s_un = top(single + multi), top(unans)
    gate = calibrate_gate(s_ans, s_un)
    c = gate["conservative"]
    lines += ["", "## Refusal gate (top reranker score)", "",
              f"Answerable top score: median {np.median(s_ans):.2f}, lowest {min(s_ans):.2f}; unanswerable: median {np.median(s_un):.2f}, highest {max(s_un):.2f}.",
              f"Conservative threshold (lowest answerable score - 1) = {c['threshold']:.2f}: keeps {c['answerable_kept']:.0%} of answerable questions and "
              f"refuses {c['unanswerable_refused']:.0%} of unanswerable ones before any model call. The model handles the rest.",
              f"For comparison, the threshold that maximises the same two numbers is {gate['threshold']:.2f} (keeps {gate['answerable_kept']:.0%}, refuses "
              f"{gate['unanswerable_refused']:.0%}); it overfits, see eval/results/natural_check.md."]
    (out / "retrieval_results.md").write_text("\n".join(lines) + "\n")
    (out / "retrieval_results.json").write_text(json.dumps({"summary": summary, "gate": gate, "scores_answerable": s_ans, "scores_unanswerable": s_un}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:4])
