"""Gradio interface: ask a question, run the agent, or compare retrieval modes.

    GEMINI_API_KEY=... python app.py            # index folder defaults to ./index
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from docuproof.agent import Agent
from docuproof.assistant import Assistant
from docuproof.cli import load_index
from docuproof.config import DEFAULT, DEFAULT_GATE
from docuproof.index import IndexNotBuiltError
from docuproof.llm import LLM, LLMError
from docuproof.rag import RAG, Conversation
from docuproof.retrieve import Mode, Reranker, Retriever
from docuproof.verify import NLIVerifier, audit

INDEX_DIR = Path(os.getenv("DOCUPROOF_INDEX", "index"))
GATE = float(os.getenv("DOCUPROOF_GATE", DEFAULT_GATE))
MODES = [m.value for m in Mode]

try:
    llm = LLM(os.getenv("DOCUPROOF_PROVIDER", "gemini"), os.getenv("DOCUPROOF_MODEL"), min_interval=1.0)
    index = load_index(INDEX_DIR)
except (IndexNotBuiltError, LLMError) as exc:
    raise SystemExit(str(exc))
retriever = Retriever(index, Reranker(DEFAULT))
nli = NLIVerifier(DEFAULT)
agent = Agent(llm, retriever)


def ask(question: str, mode: str, do_audit: bool, escalate: bool, conv: Conversation):
    if not question.strip():
        return "Type a question first.", [], "", conv
    rag = RAG(retriever, llm, Mode(mode), gate=GATE)
    if escalate:
        out = Assistant(rag, agent).ask(question, conv)
        ans = out.single
    else:
        out, ans = None, rag.ask(question, conv)
    note = f"*Searched for:* {ans.query_used}\n\n" if ans.query_used != question else ""
    rows = [[i, h.chunk.source, h.chunk.page, round(h.score, 3), h.chunk.text[:220] + "..."] for i, h in enumerate(ans.hits, 1)]

    if out is not None and out.used_agent:   # the first search was not enough, so the agent answered
        res = out.agent
        trace = "\n".join(f"{i}. `{s.tool}` {s.arguments}" for i, s in enumerate(res.steps, 1)) or "(no tool calls)"
        cites = "\n".join(f"- {s} p.{p}: {'seen in tool results' if ok else 'NOT seen (invalid)'}" for s, p, ok in res.cited)
        text = note + ("**I don't know based on the documents.**" if res.refused else res.text)
        report = ("*The first search was not enough, so the agent answered. Claim audit applies to single-shot answers.*\n\n"
                  f"**Tool calls**\n\n{trace}\n\n**Citations**\n\n{cites or '-'}")
        return text, [], report, conv     # the first search's passages are not what the agent used, so hide them

    text = note + ("**I don't know based on the documents.**" if ans.refused else ans.text)
    report = ""
    if do_audit and not ans.refused:
        parts = []
        for a in audit(ans, llm, nli):
            nli_txt = "-" if a.nli_entailment is None else f"{a.nli_entailment:.2f}"
            parts.append(f"- {a.claim.text}\n  - LLM judge: **{a.llm_verdict}** ({a.llm_reason})\n  - NLI entailment: **{nli_txt}**")
        bad = [c for c in ans.citations if not c.valid]
        report = f"Citations: {len(ans.citations)} ({len(bad)} invalid)\n\n" + ("\n".join(parts) or "No cited claims to audit.")
    return text, rows, report, conv


def run_agent(question: str):
    if not question.strip():
        return "Type a question first.", ""
    res = agent.run(question)
    trace = "\n".join(f"{i}. `{s.tool}` {s.arguments}" for i, s in enumerate(res.steps, 1)) or "(no tool calls)"
    cites = "\n".join(f"- {s} p.{p}: {'seen in tool results' if ok else 'NOT seen (invalid)'}" for s, p, ok in res.cited)
    return ("**I don't know based on the documents.**" if res.refused else res.text), f"**Tool calls**\n\n{trace}\n\n**Citations**\n\n{cites or '-'}"


def compare(question: str):
    rows = []
    for mode in Mode:
        for rank, h in enumerate(retriever.search(question, mode), 1):
            rows.append([mode.value, rank, h.chunk.source, h.chunk.page, round(h.score, 3)])
    return rows


with gr.Blocks(title="DocuProof") as demo:
    gr.Markdown(f"# DocuProof\nGrounded answers with citations and claim auditing over {len(index)} passages "
                f"from {len(index.sources)} documents.")
    conv = gr.State(Conversation())
    with gr.Tab("Ask"):
        q = gr.Textbox(label="Question", lines=2)
        with gr.Row():
            mode = gr.Dropdown(MODES, value=Mode.HYBRID_RERANK.value, label="Retrieval mode")
            do_audit = gr.Checkbox(value=True, label="Audit claims (LLM judge + NLI)")
            escalate = gr.Checkbox(value=True, label="Auto: use the agent if the first search is not enough")
        with gr.Row():
            go = gr.Button("Ask", variant="primary")
            new = gr.Button("New conversation")
        answer = gr.Markdown(label="Answer")
        sources = gr.Dataframe(headers=["#", "Document", "Page", "Score", "Passage"], label="Retrieved passages", wrap=True)
        report = gr.Markdown(label="Claim audit")
        go.click(ask, [q, mode, do_audit, escalate, conv], [answer, sources, report, conv])
        q.submit(ask, [q, mode, do_audit, escalate, conv], [answer, sources, report, conv])
        new.click(lambda: (Conversation(), "", [], ""), None, [conv, answer, sources, report])
    with gr.Tab("Agent (tool calling)"):
        aq = gr.Textbox(label="Question (works well for questions that combine several documents)", lines=2)
        ago = gr.Button("Run agent", variant="primary")
        aans = gr.Markdown()
        atrace = gr.Markdown()
        ago.click(run_agent, aq, [aans, atrace])
    with gr.Tab("Compare retrieval modes"):
        cq = gr.Textbox(label="Query", lines=1)
        cgo = gr.Button("Compare", variant="primary")
        ctab = gr.Dataframe(headers=["Mode", "Rank", "Document", "Page", "Score"])
        cgo.click(compare, cq, ctab)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
                theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate", font=[gr.themes.Font(f) for f in ("Segoe UI", "Calibri", "system-ui", "sans-serif")]))
