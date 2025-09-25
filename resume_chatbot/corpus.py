"""Helpers for assembling the resume corpus used by retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .data_loader import Document, load_resume_documents
from .knowledge_graph import load_knowledge_graph_documents


def load_resume_corpus(directory: Path) -> List[Document]:
    """Load resume sections plus optional knowledge graph context from ``directory``."""

    documents = load_resume_documents(directory)
    documents.extend(load_knowledge_graph_documents(directory))
    return documents


__all__ = ["load_resume_corpus"]
