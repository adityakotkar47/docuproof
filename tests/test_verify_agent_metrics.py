import numpy as np

from docuproof.agent import Agent, parse_citations
from docuproof.config import Settings
from docuproof.metrics import (Target, bootstrap_ci, first_rank, hit_at, mcnemar_exact, mrr_at, norm,
                               paired_bootstrap, target_recall)
from docuproof.rag import Answer
from docuproof.retrieve import Hit, Retriever
from docuproof.verify import ClaimAudit, Claim, agreement, audit, parse_verdicts, split_claims

from conftest import CHUNKS, FakeLLM, FakeReranker, final_message, tool_call


# --- claims and verdicts -------------------------------------------------
def test_split_claims_extracts_cited_sentences_only():
    claims = split_claims("GANs use two networks [1]. This sentence has no citation. Diffusion adds noise [2][3].")
    assert [(c.text, c.cites) for c in claims] == [
        ("GANs use two networks .", [1]), ("Diffusion adds noise .", [2, 3])]


def test_parse_verdicts_accepts_valid_and_drops_bad_rows():
    raw = '{"results": [{"id": 1, "verdict": "Supported", "reason": "x"}, {"id": 2, "verdict": "maybe"}, {"id": 9, "verdict": "supported"}]}'
    assert parse_verdicts(raw, 2) == {1: ("supported", "x")}


def test_parse_verdicts_survives_garbage():
    assert parse_verdicts("not json", 3) == {}
    assert parse_verdicts('{"results": 5}', 3) == {}


class FakeNLI:
    threshold = 0.5

    def entailment(self, claim, passages):
        return 0.9 if "adversarial" in " ".join(passages) else 0.1


def _answer(text):
    return Answer("q", text, False, [], [Hit(CHUNKS[0], 1.0), Hit(CHUNKS[1], 1.0)], "q")


def test_audit_combines_both_verifiers():
    llm = FakeLLM(['{"results": [{"id": 1, "verdict": "supported", "reason": "ok"}, {"id": 2, "verdict": "unsupported", "reason": "no"}]}'])
    out = audit(_answer("The generator is trained adversarially [1]. The VAE uses a transformer [2]."), llm, FakeNLI())
    assert [a.llm_verdict for a in out] == ["supported", "unsupported"]
    assert [a.nli_verdict for a in out] == ["supported", "unsupported"]
    assert agreement(out) == {"n": 2, "agree": 1.0}


def test_agreement_counts_disagreements():
    claim = Claim("x y z", [1])
    audits = [ClaimAudit(claim, "supported", "", 0.9, "supported"), ClaimAudit(claim, "supported", "", 0.1, "unsupported")]
    assert agreement(audits)["agree"] == 0.5
    assert agreement([])["n"] == 0


# --- metrics --------------------------------------------------------------
def test_norm_ignores_case_space_and_punctuation():
    assert norm("Error-Code  E 4021!") == "errorcodee4021"


def test_first_rank_and_hit_mrr():
    targets = [Target("rag", 3, "returned no passages")]
    assert first_rank(CHUNKS, targets) == 4
    assert first_rank(CHUNKS, [Target("rag", 99, "returned no passages")]) is None
    ranks = [1, 3, None, 6]
    assert list(hit_at(ranks, 4)) == [1, 1, 0, 0]
    assert np.allclose(mrr_at(ranks, 4), [1, 1 / 3, 0, 0])


def test_target_recall_for_multi_hop():
    ts = [Target("gan", 1, "adversarial game"), Target("ddpm", 4, "reverse a gradual noising")]
    assert target_recall(CHUNKS[:1], ts) == 0.5
    assert target_recall(CHUNKS, ts) == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    vals = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    mean, lo, hi = bootstrap_ci(vals, n_boot=2000)
    assert lo <= mean <= hi and bootstrap_ci(vals, n_boot=2000) == (mean, lo, hi)


def test_paired_bootstrap_detects_a_clear_gain():
    a, b = np.zeros(40), np.ones(40)
    mean, lo, hi = paired_bootstrap(a, b, n_boot=1000)
    assert mean == 1.0 and lo == 1.0


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 5) - 0.0625) < 1e-9
    assert mcnemar_exact(3, 3) == 1.0


# --- agent -----------------------------------------------------------------
def test_agent_calls_tool_then_answers_with_valid_citation(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=3, pool=5))
    llm = FakeLLM([tool_call("search_documents", {"query": "adversarial generator"}),
                   final_message("The generator plays a game [gan, p.1].")])
    res = Agent(llm, r).run("What does the GAN generator do?")
    assert [s.tool for s in res.steps] == ["search_documents"]
    assert res.cited == [("gan", 1, True)] and not res.refused and res.finished


def test_agent_flags_citation_to_page_it_never_saw(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=3, pool=5))
    llm = FakeLLM([tool_call("list_sources", {}), final_message("Claim [vae, p.2].")])
    res = Agent(llm, r).run("q")
    assert res.seen == set() and res.cited == [("vae", 2, False)]


def test_agent_read_page_and_list_sources(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=3, pool=5))
    llm = FakeLLM([tool_call("list_sources", {}), tool_call("read_page", {"source": "rag", "page": 3}, "c2"),
                   final_message("Done [rag, p.3].")])
    res = Agent(llm, r).run("q")
    assert [s.tool for s in res.steps] == ["list_sources", "read_page"]
    assert res.cited == [("rag", 3, True)]


def test_agent_handles_bad_tool_arguments_without_crashing(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=3, pool=5))
    llm = FakeLLM([tool_call("read_page", {"source": "rag"}), final_message("INSUFFICIENT_EVIDENCE")])
    res = Agent(llm, r).run("q")
    assert res.refused and res.steps[0].tool == "read_page"


def test_agent_stops_at_step_limit(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=3, pool=5))
    llm = FakeLLM([tool_call("list_sources", {}, f"c{i}") for i in range(3)])
    res = Agent(llm, r, max_steps=3).run("q")
    assert not res.finished and len(res.steps) == 3


def test_parse_citations_handles_single_grouped_and_multi_source_forms():
    assert parse_citations("A [rag, p. 3]. B [rag, p.4].") == [("rag", 3), ("rag", 4)]
    assert parse_citations("C [toolformer, p. 2, 10].") == [("toolformer", 2), ("toolformer", 10)]
    assert parse_citations("D [toolformer, p. 10, react, p. 7].") == [("toolformer", 10), ("react", 7)]
    assert parse_citations("E [self_rag, pp. 4, 5] and [self_rag, p. 4].") == [("self_rag", 4), ("self_rag", 5)]
    assert parse_citations("no citations here [1] and [see above]") == []


def test_agent_flags_the_unseen_page_inside_a_grouped_citation(index):
    r = Retriever(index, FakeReranker(), Settings(top_k=3, pool=5))
    llm = FakeLLM([tool_call("read_page", {"source": "rag", "page": 3}), final_message("X [rag, p. 3, vae, p. 2].")])
    res = Agent(llm, r).run("q")
    assert res.cited == [("rag", 3, True), ("vae", 2, False)]
