# Answer-level results

Model: gemini-3.5-flash-lite. Retrieval: hybrid + rerank, top 5. Gate: -3.0.
40 answerable (30 single-hop, 10 multi-hop), 20 unanswerable. One run; 95% bootstrap intervals.

| Measure | Value |
|---|---|
| single: correct | 0.90 [0.77-1.00] (n=30) |
| single: correct or partial | 0.90 [0.77-1.00] (n=30) |
| single: wrongly refused | 1/30 |
| single: gold passage in context | 0.93 [0.83-1.00] (n=30) |
| single: cites a gold passage | 0.93 [0.83-1.00] (n=29) |
| multi: correct | 0.50 [0.20-0.80] (n=10) |
| multi: correct or partial | 0.50 [0.20-0.80] (n=10) |
| multi: wrongly refused | 4/10 |
| multi: gold passage in context | 0.90 [0.70-1.00] (n=10) |
| multi: cites a gold passage | 0.83 [0.50-1.00] (n=6) |
| unanswerable: correctly refused | 1.00 [1.00-1.00] (n=20) |
| unanswerable: refused by gate (no LLM call) | 18/20 |
| answers with no invalid citation | 1.00 [1.00-1.00] (n=35) |
| claims judged supported by LLM judge | 0.98 [0.93-1.00] (n=43) |
| claims judged supported by NLI model | 0.74 [0.60-0.86] (n=43) |
| verifier agreement (supported vs not) | 0.77 [0.63-0.88] (n=43) |

Verifier confusion (rows: LLM judge, columns: NLI):

| | NLI supported | NLI not supported |
|---|---|---|
| LLM supported | 32 | 10 |
| LLM not supported | 0 | 1 |

LLM requests sent: 3 (cache hits: 109).
