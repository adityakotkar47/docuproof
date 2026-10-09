from docuproof.config import Settings
from docuproof.rag import RAG, REFUSAL, Conversation, resolve_citations
from docuproof.retrieve import Hit, Mode, Retriever, rrf

from conftest import CHUNKS, FakeLLM, FakeReranker


def test_rrf_scores_and_order():
    fused = dict(rrf([[1, 2, 3], [3, 1]], k=60))
    assert abs(fused[1] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(fused[3] - (1 / 63 + 1 / 61)) < 1e-12
    assert [d for d, _ in rrf([[1, 2, 3], [3, 1]], k=60)][0] == 1


def test_rrf_unseen_in_one_list_still_ranked():
    assert {d for d, _ in rrf([[1], [2]])} == {1, 2}


def test_all_modes_return_k_hits(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=3, pool=5))
    for mode in Mode:
        assert len(r.search("retrieval generator passages", mode)) == 3


def test_rerank_mode_needs_reranker(index):
    try:
        Retriever(index, None).search("x", Mode.HYBRID_RERANK)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_max_per_source_cap(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=5, pool=10, max_per_source=1))
    sources = [h.chunk.source for h in r.search("error code retrieval passages", Mode.HYBRID)]
    assert len(sources) == len(set(sources))


def test_hit_records_ranks_in_each_retriever(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=3, pool=5))
    hit = r.search("E-4021", Mode.HYBRID)[0]
    assert hit.chunk.id == "c:3:1" and "bm25" in hit.ranks


def _hits(n):
    return [Hit(CHUNKS[i], 1.0) for i in range(n)]


def test_resolve_citations_valid_invalid_and_grouped():
    cites = resolve_citations("A fact [1]. Another [2, 3]. Wrong [9].", _hits(3))
    assert [(c.number, c.valid) for c in cites] == [(1, True), (2, True), (3, True), (9, False)]
    assert cites[0].source == "gan" and cites[3].source is None


def test_ask_returns_cited_answer(index):
    rag = RAG(Retriever(index, FakeReranker(), Settings(top_k=2, pool=5)), FakeLLM(["Fact [1]."]))
    ans = rag.ask("adversarial generator")
    assert not ans.refused and ans.citations and ans.citations[0].valid


def test_ask_detects_model_refusal(index):
    rag = RAG(Retriever(index, FakeReranker(), Settings(top_k=2, pool=5)), FakeLLM([REFUSAL]))
    ans = rag.ask("who is the president")
    assert ans.refused and ans.citations == []


def test_gate_refuses_without_calling_llm(index):
    llm = FakeLLM([])
    rag = RAG(Retriever(index, FakeReranker(), Settings(top_k=2, pool=5)), llm, gate=999.0)
    ans = rag.ask("adversarial generator")
    assert ans.refused and ans.gated and llm.seen == []


def test_memory_rewrites_follow_up_with_history(index):
    llm = FakeLLM(["First answer [1].", "What is the discriminator in a GAN?", "Second answer [1]."])
    rag = RAG(Retriever(index, FakeReranker(), Settings(top_k=2, pool=5)), llm)
    conv = Conversation()
    rag.ask("What is a GAN generator?", conv)
    ans = rag.ask("And the other part?", conv)
    assert ans.query_used == "What is the discriminator in a GAN?"
    assert any("What is a GAN generator?" in m["content"] for m in llm.seen[1])


def test_conversation_keeps_only_recent_turns():
    conv = Conversation(max_turns=2)
    for i in range(5):
        conv.add(f"q{i}", f"a{i}")
    assert [q for q, _ in conv.turns] == ["q3", "q4"]


# --- auto mode -------------------------------------------------------------
def _assistant(index, replies):
    from docuproof.agent import Agent
    from docuproof.assistant import Assistant

    llm = FakeLLM(replies)
    retriever = Retriever(index, FakeReranker(), Settings(top_k=2, pool=5))
    return Assistant(RAG(retriever, llm, gate=-5.0), Agent(llm, retriever)), llm


def test_auto_keeps_single_shot_answer_when_it_succeeds(index):
    assistant, llm = _assistant(index, ["Fact [1]."])
    out = assistant.ask("adversarial generator")
    assert not out.used_agent and not out.refused and out.text == "Fact [1]."


def test_auto_escalates_to_agent_when_single_shot_refuses(index):
    from conftest import final_message, tool_call

    assistant, llm = _assistant(index, [REFUSAL, tool_call("search_documents", {"query": "adversarial"}),
                                        final_message("The generator plays a game [gan, p. 1].")])
    out = assistant.ask("adversarial generator")
    assert out.used_agent and not out.refused and out.agent.cited == [("gan", 1, True)]


def test_auto_does_not_escalate_gated_questions(index):
    from docuproof.agent import Agent
    from docuproof.assistant import Assistant

    llm = FakeLLM([])
    retriever = Retriever(index, FakeReranker(), Settings(top_k=2, pool=5))
    out = Assistant(RAG(retriever, llm, gate=999.0), Agent(llm, retriever)).ask("anything")
    assert out.refused and not out.used_agent and llm.seen == []


def test_auto_reports_refusal_when_agent_also_refuses(index):
    from conftest import final_message

    assistant, _ = _assistant(index, [REFUSAL, final_message(REFUSAL)])
    out = assistant.ask("adversarial generator")
    assert out.used_agent and out.refused


def test_auto_updates_memory_with_the_answer_shown(index):
    from conftest import final_message

    assistant, _ = _assistant(index, [REFUSAL, final_message("Agent answer [gan, p. 1].")])
    conv = Conversation()
    assistant.ask("adversarial generator", conv)
    assert conv.turns == [("adversarial generator", "Agent answer [gan, p. 1].")]
