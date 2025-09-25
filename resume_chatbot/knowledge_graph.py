"""Utilities for loading and formatting resume knowledge graphs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .data_loader import Document


def _ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text[-1] in ".!?":
        return text
    return f"{text}."


def _format_timeframe(start: Optional[str], end: Optional[str]) -> Optional[str]:
    if start and end:
        return f"{start} to {end}"
    if start:
        return f"starting {start}"
    if end:
        return f"until {end}"
    return None


def _format_fact(
    subject: str,
    relation: str,
    target: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    location: Optional[str] = None,
    description: Optional[str] = None,
    highlights: Optional[Iterable[str]] = None,
) -> str:
    relation_text = relation.replace("_", " ").strip() or "related to"
    sentences: List[str] = [f"{subject} {relation_text} {target}."]

    timeframe = _format_timeframe(start, end)
    if timeframe:
        sentences.append(f"The timeframe was {timeframe}.")
    if location:
        sentences.append(f"This took place in {location}.")
    if description:
        sentences.append(_ensure_sentence(description))
    if highlights:
        filtered = [highlight.strip() for highlight in highlights if highlight and highlight.strip()]
        if filtered:
            sentences.append(_ensure_sentence("Highlights include " + "; ".join(filtered)))

    return " ".join(_ensure_sentence(sentence) for sentence in sentences if sentence)


def load_knowledge_graph_documents(
    directory: Path,
    *,
    filename: str = "knowledge_graph.json",
) -> List[Document]:
    """Load knowledge graph triples and convert them into text documents."""

    directory = directory.expanduser().resolve()
    path = directory / filename
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    nodes_data: List[dict] = data.get("nodes", [])  # type: ignore[assignment]
    edges_data: List[dict] = data.get("edges", [])  # type: ignore[assignment]

    nodes: Dict[str, dict] = {}
    for node in nodes_data:
        node_id = node.get("id")
        if not node_id:
            continue
        nodes[str(node_id)] = node

    grouped: Dict[str, List[str]] = defaultdict(list)
    for edge in edges_data:
        source_id = edge.get("source")
        target_id = edge.get("target")
        if not source_id or not target_id:
            continue

        subject = nodes.get(str(source_id), {}).get("label", str(source_id))
        target = nodes.get(str(target_id), {}).get("label", str(target_id))
        relation = str(edge.get("relation", "related_to"))
        fact = _format_fact(
            subject,
            relation,
            target,
            start=edge.get("start"),
            end=edge.get("end"),
            location=edge.get("location"),
            description=edge.get("description"),
            highlights=edge.get("highlights"),
        )
        if fact:
            grouped[subject].append(fact)

    documents: List[Document] = []
    if not grouped:
        return documents

    relative_source = str(path.relative_to(directory))
    for index, (subject, facts) in enumerate(sorted(grouped.items()), start=1):
        content = " ".join(facts).strip()
        if not content:
            continue
        documents.append(
            Document(
                content=content,
                metadata={
                    "source": relative_source,
                    "title": subject,
                    "chunk": index,
                },
            )
        )
    return documents


__all__ = ["load_knowledge_graph_documents"]
