import json
from pathlib import Path

from resume_chatbot.knowledge_graph import load_knowledge_graph_documents


def test_load_knowledge_graph_documents(tmp_path: Path) -> None:
    graph = {
        "title": "Test Graph",
        "nodes": [
            {"id": "person", "label": "Alex Example"},
            {"id": "company", "label": "Example Corp"},
        ],
        "edges": [
            {
                "source": "person",
                "target": "company",
                "relation": "worked_at",
                "start": "2023",
                "end": "2024",
                "location": "Remote",
                "description": "Senior data scientist building production pipelines.",
                "highlights": ["Implemented retrievers", "Managed 5-person team"],
            }
        ],
    }
    (tmp_path / "knowledge_graph.json").write_text(json.dumps(graph), encoding="utf-8")

    documents = load_knowledge_graph_documents(tmp_path)

    assert len(documents) == 1
    document = documents[0]
    assert document.metadata["source"] == "knowledge_graph.json"
    assert "Alex Example" in document.metadata["title"]
    content = document.content.lower()
    assert "worked at" in content
    assert "highlights include" in content


def test_load_knowledge_graph_documents_missing_file(tmp_path: Path) -> None:
    documents = load_knowledge_graph_documents(tmp_path)
    assert documents == []
