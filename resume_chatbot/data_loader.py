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


def _split_markdown_sections(text: str) -> Sequence[tuple[str, str]]:
    """Split a Markdown document into ``(title, body)`` sections."""

    sections: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            if current_lines:
                sections.append((current_title, current_lines))
                current_lines = []
            current_title = line.lstrip("# ").strip() or current_title
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    normalised_sections: list[tuple[str, str]] = []
    for title, lines in sections:
        body = "\n".join(line for line in lines).strip()
        if body:
            normalised_sections.append((title, body))
    if not normalised_sections and text.strip():
        normalised_sections.append(("Overview", text.strip()))
    return normalised_sections


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
        for idx, (title, body) in enumerate(_split_markdown_sections(text), start=1):
            documents.append(
                Document(
                    content=body,
                    metadata={
                        "source": str(path.relative_to(directory)),
                        "title": title,
                        "chunk": idx,
                    },
                )
            )
    return documents
