"""Utilities for loading resume documents from disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Document:
    """Simple container for resume content and metadata."""

    content: str
    metadata: dict


def _iter_text_files(directory: Path) -> Iterable[Path]:
    """Yield text-like files from ``directory`` sorted by name."""

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            yield path


def _split_markdown_sections(text: str) -> Sequence[tuple[str, str, str]]:
    """Split a Markdown document into sections by ``##`` headers.

    Returns a list of ``(parent_title, section_title, body)`` tuples.
    The top-level ``#`` header is captured as ``parent_title`` and carried
    through to every section beneath it.  Bullet points (``-`` / ``*``)
    stay attached to their nearest ``##`` header.
    """

    sections: list[tuple[str, str, list[str]]] = []
    parent_title = "Overview"
    current_title = "Overview"
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Top-level header — sets parent_title but does NOT start a new chunk
        if line.startswith("# ") and not line.startswith("## "):
            parent_title = line.lstrip("# ").strip() or parent_title
            continue

        # Second-level header — flush current chunk, start a new one
        if line.startswith("## "):
            if current_lines:
                sections.append((parent_title, current_title, current_lines))
                current_lines = []
            current_title = line.lstrip("# ").strip() or current_title
            continue

        # Everything else (bullets, text, blank lines) stays with current section
        current_lines.append(line)

    # Flush last section
    if current_lines:
        sections.append((parent_title, current_title, current_lines))

    normalised: list[tuple[str, str, str]] = []
    for p_title, s_title, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            normalised.append((p_title, s_title, body))

    # If no ## headers were found, treat the whole document as one chunk
    if not normalised and text.strip():
        normalised.append((parent_title, "Overview", text.strip()))

    return normalised


def load_resume_documents(directory: Path) -> List[Document]:
    """Load and split resume documents from ``directory``.

    Parameters
    ----------
    directory:
        Directory containing Markdown or plain-text resume files.

    Returns
    -------
    list[Document]
        Parsed documents, each capturing a logical resume section.
    """

    directory = directory.expanduser().resolve()
    if not directory.exists():
        return []

    documents: list[Document] = []
    for path in _iter_text_files(directory):
        text = path.read_text(encoding="utf-8")
        for idx, (parent_title, title, body) in enumerate(
            _split_markdown_sections(text), start=1
        ):
            documents.append(
                Document(
                    content=body,
                    metadata={
                        "source": str(path.relative_to(directory)),
                        "title": title,
                        "parent_title": parent_title,
                        "chunk_index": idx,
                    },
                )
            )
    return documents
