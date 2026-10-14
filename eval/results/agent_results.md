# Agent vs single-shot RAG on multi-hop questions

Model: gemini-3.5-flash-lite. 20 multi-hop questions, one run. Correctness graded by an LLM against the evidence spans.

| Measure | Single-shot RAG | Function-calling agent |
|---|---|---|
| correct | 0.35 [0.15-0.55] (n=20) | 0.80 [0.60-0.95] (n=20) |
| correct or partial | 0.35 [0.15-0.55] (n=20) | 0.90 [0.75-1.00] (n=20) |
| gold passages reached | 0.60 [0.45-0.75] (n=20) | 0.70 [0.55-0.85] (n=20) |
| mean tool calls | 0 (1 retrieval) | 2.5 |
| agent hit the step limit | - | 1/20 |
| agent citations to pages it never saw | - | 0 |

LLM requests sent: 96 (cache hits: 28).
Note: gold passages reached for the agent is measured at page level (the tool results it saw), for single-shot at chunk level.
