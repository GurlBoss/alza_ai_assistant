"""Splits a document into overlapping chunks, keeping section headings."""

import re

CHUNK_WORDS = 400
OVERLAP_WORDS = 50

HEADING = re.compile(r"^#{1,6}\s+\S")


def split_sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) pairs; markdown headings start a new section."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if HEADING.match(line):
            sections.append((line.lstrip("#").strip(), []))
        else:
            sections[-1][1].append(line)
    return [
        (heading, "\n".join(lines).strip())
        for heading, lines in sections
        if heading or "\n".join(lines).strip()
    ]


def split_words(
    text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS
) -> list[str]:
    """Windows of `size` words, each starting `size - overlap` words after the last."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return windows


def chunk_document(text: str) -> list[str]:
    """Chunks of one document. Each chunk starts with its section heading, so it
    still makes sense on its own."""
    chunks = []
    for heading, body in split_sections(text):
        for window in split_words(body) or [""]:
            chunk = f"{heading}\n{window}".strip() if heading else window
            if chunk:
                chunks.append(chunk)
    return chunks
