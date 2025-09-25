from pathlib import Path

import pytest

from resume_chatbot.data_loader import Document
from resume_chatbot.retriever import ResumeRetriever


def test_retriever_returns_high_score_for_matching_document():
    docs = [
        Document(
            content="Alex is a machine learning engineer specialising in retrieval augmented generation.",
            metadata={"source": "resume.md", "title": "Summary"},
        ),
        Document(
            content="Alex volunteers at the local AI meetup.",
            metadata={"source": "resume.md", "title": "Volunteering"},
        ),
    ]
    retriever = ResumeRetriever(docs)

    results = retriever.retrieve("What kind of engineer is Alex?", top_k=1)

    assert results
    assert "machine learning" in results[0].document.content


def test_retriever_handles_zero_matches():
    docs = [
        Document(
            content="The candidate enjoys mentoring.",
            metadata={"source": "resume.md", "title": "Community"},
        )
    ]
    retriever = ResumeRetriever(docs)

    results = retriever.retrieve("What certifications does Alex hold?", top_k=3)

    assert results == []


def test_retriever_requires_documents():
    with pytest.raises(ValueError):
        ResumeRetriever([])
