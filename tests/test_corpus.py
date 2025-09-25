from pathlib import Path

from resume_chatbot.corpus import load_resume_corpus


def test_load_resume_corpus_combines_sources() -> None:
    documents = load_resume_corpus(Path("data/resume"))
    assert documents, "Expected resume documents to be loaded."

    sources = {document.metadata.get("source") for document in documents}
    assert "resume_example.md" in sources
    assert "knowledge_graph.json" in sources
