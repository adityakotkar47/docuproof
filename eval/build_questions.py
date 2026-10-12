"""Build the evaluation question set from the indexed corpus.

single      one passage answers the question
multi       two passages from two different papers are both needed
unanswerable  nothing in the corpus answers it (hand-written)

An LLM writes each question and copies the evidence span from the passage; a question is kept only if the span is
found verbatim in the passage it came from. The question writer is a different model from the one that answers.

    python eval/build_questions.py index eval/questions.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docuproof.config import DEFAULT  # noqa: E402
from docuproof.index import Embedder, HybridIndex  # noqa: E402
from docuproof.llm import LLM  # noqa: E402
from docuproof.metrics import norm  # noqa: E402

SEED = 2026
PER_PAPER = 3
BATCH = 5

PAIRS = [("gan", "vae"), ("ddpm", "latent_diffusion"), ("rag", "self_rag"), ("react", "toolformer"),
         ("bert", "gpt3"), ("lora", "instructgpt"), ("ragas", "llm_as_judge"),
         ("hallucination_survey", "correctness_not_faithfulness"), ("dpr", "sbert"), ("chain_of_thought", "react"),
         ("lost_in_the_middle", "rag"), ("clip", "latent_diffusion"), ("bge", "sbert"), ("llama2", "instructgpt"),
         ("foundation_model_risks", "hallucination_survey")]

UNANSWERABLE = [
    "What is the best recipe for a vegetarian lasagna?", "Who won the 2022 FIFA World Cup?",
    "How do I deploy a Kubernetes pod with a persistent volume?", "What is the boiling point of ethanol in Celsius?",
    "Which programming language was used to write the Linux kernel?", "What is the capital of Australia?",
    "How many players are on a cricket team?", "What is the current price of Bitcoin?",
    "What is the context window of Gemini 1.5 Pro?", "How much did it cost to train Gemini Ultra?",
    "Who founded DeepMind?", "What GPU memory is needed to fine-tune a 70B model with QLoRA on one card?",
    "What topics does an undergraduate operating systems course cover?", "How does the Mamba state space model work?",
    "What is the license of the Mistral 7B weights?", "What was the training dataset size of Stable Diffusion 3?",
    "How do I set up OAuth2 login in Django?", "What is the Fermi paradox?",
    "Which Indian state has the largest area?", "How is mRNA vaccine technology manufactured?",
]

SINGLE_SYSTEM = (
    "For each numbered passage write ONE question that the passage alone answers, and copy the evidence. "
    "Rules: the question must be self-contained and must not mention 'the passage', 'the text' or 'the paper'; "
    "ask about a specific fact, definition, method detail or number; the evidence must be an EXACT contiguous span "
    "of 6 to 25 words copied character for character from the passage. "
    'Reply as JSON: {"items": [{"id": 1, "question": "...", "evidence": "..."}]}.'
)
MULTI_SYSTEM = (
    "You are given passage A and passage B from two different papers. Write ONE question that needs facts from BOTH "
    "(for example a comparison or a combination), is self-contained, and does not mention 'passage', 'text' or 'paper'. "
    "Copy one EXACT contiguous span of 6 to 25 words from each passage as evidence. "
    'Reply as JSON: {"question": "...", "evidence_a": "...", "evidence_b": "..."}.'
)


def usable(chunk) -> bool:
    letters = sum(ch.isalpha() for ch in chunk.text)
    return chunk.n_words >= 110 and letters / max(1, len(chunk.text)) > 0.78 and chunk.text.count(". ") >= 3


def target(chunk, phrase: str) -> dict | None:
    return {"source": chunk.source, "page": chunk.page, "phrase": phrase.strip()} if norm(phrase) in norm(chunk.text) else None


def loads(raw: str):
    try:
        return json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


def main(index_dir: str, out: str) -> None:
    rng = random.Random(SEED)
    index = HybridIndex.load(Path(index_dir), Embedder(DEFAULT))
    llm = LLM("gemini", "gemini-3.1-flash-lite", min_interval=3.5, cache_dir=Path(".llm_cache"))
    questions: list[dict] = []

    # ---- single hop -------------------------------------------------------
    picks = []
    for src in index.sources:
        pool = [i for i, c in enumerate(index.chunks) if c.source == src and usable(c)]
        picks += rng.sample(pool, min(PER_PAPER, len(pool)))
    rng.shuffle(picks)
    for b in range(0, len(picks), BATCH):
        batch = picks[b:b + BATCH]
        body = "\n\n".join(f"Passage {n}:\n{index.chunks[i].text}" for n, i in enumerate(batch, 1))
        data = loads(llm.chat([{"role": "system", "content": SINGLE_SYSTEM}, {"role": "user", "content": body}],
                              json_mode=True, max_tokens=1500))
        for item in data.get("items", []):
            n = int(item.get("id", 0))
            if 1 <= n <= len(batch):
                t = target(index.chunks[batch[n - 1]], str(item.get("evidence", "")))
                if t and item.get("question"):
                    questions.append({"kind": "single", "question": item["question"].strip(), "targets": [t]})
    n_single = len(questions)

    # ---- multi hop: nearest chunk in the paired paper -----------------------
    for pa, pb in PAIRS:
        a_ids = [i for i, c in enumerate(index.chunks) if c.source == pa and usable(c)]
        b_ids = [i for i, c in enumerate(index.chunks) if c.source == pb and usable(c)]
        if not a_ids or not b_ids:
            continue
        for ia in rng.sample(a_ids, min(2, len(a_ids))):
            sims = index.embeddings[b_ids] @ index.embeddings[ia]
            ib = b_ids[int(sims.argmax())]
            a, b = index.chunks[ia], index.chunks[ib]
            data = loads(llm.chat([{"role": "system", "content": MULTI_SYSTEM},
                                   {"role": "user", "content": f"Passage A:\n{a.text}\n\nPassage B:\n{b.text}"}],
                                  json_mode=True, max_tokens=700))
            ta, tb = target(a, str(data.get("evidence_a", ""))), target(b, str(data.get("evidence_b", "")))
            if ta and tb and data.get("question"):
                questions.append({"kind": "multi", "question": data["question"].strip(), "targets": [ta, tb]})
    n_multi = len(questions) - n_single

    for q in UNANSWERABLE:
        questions.append({"kind": "unanswerable", "question": q, "targets": []})
    for i, q in enumerate(questions, 1):
        q["id"] = f"q{i:03d}"
    Path(out).write_text(json.dumps(questions, indent=1, ensure_ascii=False))
    print(f"single {n_single}, multi {n_multi}, unanswerable {len(UNANSWERABLE)} -> {out} (LLM calls: {llm.calls}, cached: {llm.cache_hits})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
