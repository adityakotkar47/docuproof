import pytest

from docuproof.config import Settings
from docuproof.index import HybridIndex, IndexNotBuiltError, tokenize
from docuproof import ingest
from docuproof.ingest import Page, chunk_page, clean_text, looks_like_references, strip_running_lines

from conftest import CHUNKS, FakeEmbedder


def test_clean_text_joins_hyphenated_breaks_and_collapses_space():
    assert clean_text("retrie-\nval   works\n\nwell") == "retrieval works well"


def test_strip_running_lines_removes_repeated_headers_only():
    pages = [f"Conference 2024\nunique body line {chr(97 + i)}x{chr(107 + i)}\n{i}" for i in range(8)]
    cleaned = strip_running_lines(pages)
    assert all("Conference" not in p for p in cleaned)
    assert all(f"unique body line {chr(97 + i)}x{chr(107 + i)}" in p for i, p in enumerate(cleaned))


def test_strip_running_lines_leaves_short_documents_alone():
    pages = ["Header\nbody a", "Header\nbody b"]
    assert strip_running_lines(pages) == pages


def test_references_page_detected():
    refs = " ".join(f"Smith J. ({1990 + i}) Title of paper {i}." for i in range(20))
    assert looks_like_references(refs)
    assert not looks_like_references("This is an ordinary paragraph about neural networks and training.")


def test_chunking_respects_size_overlap_and_page():
    sentences = [f"Sentence number {i} has exactly eight plain words in it." for i in range(60)]
    page = Page("doc", 7, " ".join(sentences))
    settings = Settings(chunk_words=50, chunk_overlap_words=16)
    chunks = chunk_page(page, settings)
    assert len(chunks) > 3
    assert all(c.page == 7 and c.source == "doc" for c in chunks)
    assert all(c.n_words <= 50 + 10 for c in chunks)
    # consecutive chunks share trailing/leading sentences
    assert chunks[0].text.split(".")[-2].strip() in chunks[1].text


def test_chunk_ids_are_unique():
    page = Page("doc", 1, " ".join(f"Sentence {i} is here and long enough to count." for i in range(40)))
    ids = [c.id for c in chunk_page(page, Settings(chunk_words=30, chunk_overlap_words=10))]
    assert len(ids) == len(set(ids))


def test_tokenize_keeps_codes_and_drops_stopwords():
    toks = tokenize("What does error code E-4021 mean in GPT-3?")
    assert "e-4021" in toks and "gpt-3" in toks
    assert "the" not in tokenize("the retriever")


def test_bm25_finds_exact_code(index):
    top = index.bm25_search("E-4021", 3)
    assert top and index.chunks[top[0][0]].id == "c:3:1"


def test_dense_search_prefers_similar_text(index):
    top = index.dense_search("adversarial generator discriminator", 1)
    assert index.chunks[top[0][0]].source == "gan"


def test_save_and_load_roundtrip(tmp_path, index):
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, FakeEmbedder())
    assert [c.id for c in loaded.chunks] == [c.id for c in CHUNKS]
    assert loaded.dense_search("diffusion noising", 1)[0][0] == index.dense_search("diffusion noising", 1)[0][0]


def test_load_missing_index_reports_build_commands(tmp_path):
    with pytest.raises(IndexNotBuiltError) as exc_info:
        HybridIndex.load(tmp_path, FakeEmbedder())
    message = str(exc_info.value)
    assert f"Index at {tmp_path} was not found" in message
    assert "uv run python scripts/fetch_corpus.py" in message
    assert "uv run python -m docuproof.cli build data/corpus index" in message


def test_page_text_concatenates_chunks_of_a_page(index):
    text = index.page_text("rag", 3)
    assert "Retrieval-augmented" in text and "E-4021" in text


def test_clean_text_expands_ligatures():
    assert clean_text("e\ufb03cient \ufb01ne-tuning of \ufb02ow") == "efficient fine-tuning of flow"


def test_extract_pdf_pages_falls_back_to_pypdf_without_poppler(monkeypatch, tmp_path):
    from pypdf import PdfWriter

    pdf = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(width=100, height=100)
    w.write(pdf)
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    assert ingest.extract_pdf_pages(pdf) == [""]


def test_extract_pdf_pages_splits_pdftotext_output_on_form_feed(monkeypatch, tmp_path):
    class Proc:
        stdout = "page one text\fpage two text\f".encode()

    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/usr/bin/pdftotext")
    monkeypatch.setattr(ingest.subprocess, "run", lambda *a, **k: Proc())
    assert ingest.extract_pdf_pages(tmp_path / "x.pdf") == ["page one text", "page two text"]
