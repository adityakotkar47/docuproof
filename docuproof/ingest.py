"""Read documents into pages and split pages into overlapping sentence-based chunks."""
from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT, Settings

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class Page:
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def clean_text(raw: str) -> str:
    """Join hyphenated line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFKC", raw).replace("\x00", " ")   # NFKC expands ligatures such as the fi glyph
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_running_lines(page_texts: list[str], min_share: float = 0.35) -> list[str]:
    """Remove short lines (page headers, footers, arXiv stamps) repeated across many pages."""
    if len(page_texts) < 4:
        return page_texts
    norm = lambda line: re.sub(r"\d+", "#", line.strip().lower())  # noqa: E731
    counts: Counter[str] = Counter()
    for text in page_texts:
        counts.update({norm(l) for l in text.splitlines() if 0 < len(l.strip()) <= 90})
    limit = max(3, int(min_share * len(page_texts)))
    repeated = {line for line, n in counts.items() if n >= limit}
    return ["\n".join(l for l in text.splitlines() if norm(l) not in repeated) for text in page_texts]


def looks_like_references(text: str) -> bool:
    """Bibliography pages are dense in years and add little to question answering."""
    words = max(1, len(text.split()))
    return len(_YEAR.findall(text)) * 100 / words > 6


def extract_pdf_pages(path: Path) -> list[str]:
    """Page texts. Prefers poppler's pdftotext (better word spacing); falls back to pypdf."""
    if shutil.which("pdftotext"):
        proc = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, timeout=300)
        text = proc.stdout.decode("utf-8", errors="replace")
        if text.strip():
            pages = text.split("\f")
            return pages[:-1] if pages and not pages[-1].strip() else pages
    from pypdf import PdfReader

    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def load_pdf(path: Path) -> list[Page]:
    raw = extract_pdf_pages(path)
    pages: list[Page] = []
    for i, text in enumerate(strip_running_lines(raw), start=1):
        text = clean_text(text)
        if len(text) >= 200 and not looks_like_references(text):
            pages.append(Page(path.stem, i, text))
    return pages


def load_text(path: Path) -> list[Page]:
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return [Page(path.stem, 1, text)] if text else []


def load_document(path: Path) -> list[Page]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"unsupported file type: {path.name}")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_page(page: Page, settings: Settings = DEFAULT) -> list[Chunk]:
    """Greedy sentence packing to ~chunk_words, carrying the last sentences into the next chunk."""
    sentences = split_sentences(page.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0

    def flush() -> None:
        chunks.append(Chunk(f"{page.source}:{page.page}:{len(chunks)}", page.source, page.page, " ".join(current)))

    for sent in sentences:
        n = len(sent.split())
        if current and count + n > settings.chunk_words:
            flush()
            # overlap: keep trailing sentences up to chunk_overlap_words
            keep: list[str] = []
            kept = 0
            for s in reversed(current):
                w = len(s.split())
                if kept + w > settings.chunk_overlap_words:
                    break
                keep.insert(0, s)
                kept += w
            current, count = keep, kept
        current.append(sent)
        count += n
    if current and (not chunks or " ".join(current) != chunks[-1].text):
        flush()
    return chunks


def ingest_folder(folder: Path, settings: Settings = DEFAULT) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        for page in load_document(path):
            chunks.extend(chunk_page(page, settings))
    return chunks
