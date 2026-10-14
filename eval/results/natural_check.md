# Refusal gate on natural-phrasing questions

20 answerable and 6 unanswerable questions, hand-written and short. Not used to choose the gate.

| Gate | Threshold | Answerable wrongly refused | Unanswerable refused (by gate / by model) | LLM calls skipped |
|---|---|---|---|---|
| tuned to the generated questions | 1.32 | 6 of 20 | 6 of 6 (5 / 1) | 11 of 26 |
| conservative (default) | -3.00 | 2 of 20 | 6 of 6 (4 / 2) | 4 of 26 |

LLM requests sent: 24 (cache hits: 15).
