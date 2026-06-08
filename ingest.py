"""Document ingestion and chunking for The Unofficial Guide.

Loads the synthetic Rate My Professor-style review files from documents/,
cleans them, and splits them into self-contained, review-level chunks with
source metadata. See planning.md (Chunking Strategy) for the rationale.

Run directly to inspect the output:
    python ingest.py
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# A long single review gets split into windows of this many characters with
# OVERLAP characters shared between neighbours, so a fact that spans the split
# is still retrievable from at least one chunk. Most reviews are well under
# MAX_CHUNK_CHARS and stay intact as a single chunk.
MAX_CHUNK_CHARS = 600
OVERLAP_CHARS = 80
MIN_CHUNK_CHARS = 20

# Parses a header line like:
# "Professor Daniel Rivera — Computer Science (CSCI 135, Introduction...) — CUNY Hunter College"
HEADER_RE = re.compile(
    r"Professor\s+(?P<professor>.+?)\s+[—-]\s+.*?\((?P<course>[A-Z]{2,6}\s?\d{2,4})[,)]"
)


@dataclass
class Chunk:
    text: str
    source: str
    professor: str
    course: str
    chunk_index: int

    def to_metadata(self) -> dict:
        return {
            "source": self.source,
            "professor": self.professor,
            "course": self.course,
            "chunk_index": self.chunk_index,
        }


@dataclass
class Document:
    source: str
    professor: str
    course: str
    raw_text: str
    reviews: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Strip markup artifacts and normalize whitespace."""
    text = html.unescape(text)            # &amp; -> &, &#39; -> '
    text = re.sub(r"<[^>]+>", "", text)   # any stray HTML tags
    text = text.replace("\u00a0", " ")    # non-breaking spaces
    # collapse runs of spaces/tabs but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_header(raw_text: str, source: str) -> tuple[str, str]:
    """Extract professor name and course code from the file's first lines."""
    first_line = raw_text.splitlines()[0] if raw_text else ""
    match = HEADER_RE.search(first_line)
    if match:
        professor = match.group("professor").strip()
        course = re.sub(r"\s+", " ", match.group("course")).strip()
        return professor, course
    # fallback: derive from filename like "rivera_csci135.txt"
    stem = Path(source).stem
    return stem, "unknown"


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[Document]:
    """Load every .txt file in documents/ and split it into review blocks."""
    documents: list[Document] = []
    for path in sorted(documents_dir.glob("*.txt")):
        raw = clean_text(path.read_text(encoding="utf-8"))
        if not raw:
            continue
        professor, course = parse_header(raw, path.name)

        # Drop the header line and the synthetic-data NOTE line, then split the
        # rest into individual reviews on blank lines.
        body_lines = raw.splitlines()
        body = "\n".join(
            line for line in body_lines[1:] if not line.strip().startswith("NOTE:")
        )
        blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
        documents.append(
            Document(
                source=path.name,
                professor=professor,
                course=course,
                raw_text=raw,
                reviews=blocks,
            )
        )
    return documents


def _split_long(text: str) -> list[str]:
    """Window a too-long review into overlapping pieces on sentence-ish bounds."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        window = text[start:end]
        # try to break on the last sentence boundary inside the window
        if end < len(text):
            boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if boundary > MAX_CHUNK_CHARS // 2:
                end = start + boundary + 1
                window = text[start:end]
        pieces.append(window.strip())
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return [p for p in pieces if p]


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Turn each review into one (or, if long, a few overlapping) chunk(s).

    Every chunk is prefixed with the professor + course so it is self-contained
    for both semantic retrieval and source attribution.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        idx = 0
        prefix = f"[Professor {doc.professor}, {doc.course}] "
        for review in doc.reviews:
            for piece in _split_long(review):
                body = piece.strip()
                if len(body) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(
                    Chunk(
                        text=prefix + body,
                        source=doc.source,
                        professor=doc.professor,
                        course=doc.course,
                        chunk_index=idx,
                    )
                )
                idx += 1
    return chunks


def build_chunks() -> list[Chunk]:
    """Public entry point used by the embedding step."""
    return chunk_documents(load_documents())


if __name__ == "__main__":
    docs = load_documents()
    chunks = build_chunks()
    print(f"Loaded {len(docs)} documents from {DOCUMENTS_DIR}")
    print(f"Produced {len(chunks)} chunks total\n")

    lengths = [len(c.text) for c in chunks]
    if lengths:
        print(
            f"Chunk length chars -> min {min(lengths)}, "
            f"max {max(lengths)}, avg {sum(lengths) // len(lengths)}\n"
        )

    print("=== 5 sample chunks ===")
    step = max(1, len(chunks) // 5)
    for c in chunks[::step][:5]:
        print(f"\n[source: {c.source} | chunk {c.chunk_index}]")
        print(c.text)
