# Retrieval results

Corpus: 3489 chunks from 25 papers. Questions: 74 single-hop, 28 multi-hop, 20 unanswerable. Top-5 passages. 95% bootstrap intervals (10,000 resamples).

## Single-hop

| Mode | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|
| dense | 0.50 [0.39-0.62] | 0.73 [0.64-0.82] | 0.60 [0.50-0.70] |
| bm25 | 0.84 [0.76-0.92] | 0.89 [0.82-0.96] | 0.86 [0.79-0.93] |
| hybrid | 0.68 [0.57-0.78] | 0.89 [0.81-0.96] | 0.77 [0.68-0.85] |
| hybrid_rerank | 0.80 [0.70-0.88] | 0.91 [0.84-0.96] | 0.85 [0.77-0.92] |

### Paired comparison on Hit@5 (same questions)

| Baseline -> challenger | Difference [95% CI] | Only baseline hits / only challenger hits | McNemar p |
|---|---|---|---|
| dense -> hybrid | +0.16 [+0.08, +0.26] | 1 / 13 | 0.002 |
| bm25 -> hybrid | +0.00 [-0.05, +0.05] | 2 / 2 | 1.000 |
| hybrid -> hybrid_rerank | +0.01 [-0.03, +0.07] | 1 / 2 | 1.000 |
| dense -> hybrid_rerank | +0.18 [+0.08, +0.27] | 1 / 14 | 0.001 |
| bm25 -> hybrid_rerank | +0.01 [-0.03, +0.07] | 1 / 2 | 1.000 |

## Multi-hop (two evidence passages from two papers)

| Mode | Evidence recall@5 | Both passages found |
|---|---|---|
| dense | 0.54 [0.38-0.68] | 0.36 [0.18-0.54] |
| bm25 | 0.48 [0.36-0.61] | 0.21 [0.07-0.39] |
| hybrid | 0.62 [0.48-0.77] | 0.46 [0.29-0.64] |
| hybrid_rerank | 0.66 [0.54-0.79] | 0.43 [0.25-0.61] |

## Refusal gate (top reranker score)

Answerable top score: median 5.14, lowest -1.97; unanswerable: median -8.09, highest -0.19.
Conservative threshold (lowest answerable score - 1) = -2.97: keeps 100% of answerable questions and refuses 90% of unanswerable ones before any model call. The model handles the rest.
For comparison, the threshold that maximises the same two numbers is 1.32 (keeps 98%, refuses 100%); it overfits, see eval/results/natural_check.md.
