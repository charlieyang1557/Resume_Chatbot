from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_loader import CorpusRecord
from prompt_templates import PromptBuilder
from resume_chatbot.qa_service import (
    MockLanguageModel,
    ResumeQAApplication,
    ResumeQAService,
    SimpleSearchEngine,
)


@pytest.fixture()
def sample_records() -> list[CorpusRecord]:
    return [
        CorpusRecord(
            id="exp-1",
            source="resume",
            section="Experience",
            date_range="2022-2024",
            skills=["Python", "SQL", "Machine Learning"],
            text="Charlie built anomaly detection systems at PTC Onshape using Prophet and Isolation Forest.",
        ),
        CorpusRecord(
            id="exp-2",
            source="resume",
            section="Experience",
            date_range="2021-2022",
            skills=["Product Analytics", "Dashboards"],
            text="At Pinecone Charlie delivered Book of Business dashboards and churn analysis using Python.",
        ),
    ]


def test_simple_search_engine_returns_results(sample_records: list[CorpusRecord]) -> None:
    engine = SimpleSearchEngine()
    results = engine.search("What did Charlie do at Pinecone?", sample_records, top_k=2)
    assert results
    assert any(result.record.id == "exp-2" for result in results)


def test_resume_qa_service_answers_question(sample_records: list[CorpusRecord]) -> None:
    service = ResumeQAService(
        records=sample_records,
        search_engine=SimpleSearchEngine(),
        prompt_builder=PromptBuilder(),
        language_model=MockLanguageModel(),
    )
    result = service.answer_question("Tell me about Charlie's work at PTC Onshape.")
    assert "Onshape" in result.answer
    assert result.sources
    assert result.prompt_length > 0
    assert result.response_time >= 0


def test_resume_qa_application_initializes(tmp_path: Path, sample_records: list[CorpusRecord]) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for record in sample_records:
            fh.write(json.dumps(record.to_dict()) + "\n")

    app = ResumeQAApplication(corpus_path)
    with pytest.raises(RuntimeError):
        app.answer_question("What skills does Charlie have?")

    app.initialize()
    assert app.ready
    stats = app.stats()
    assert stats.total_records == len(sample_records)


def test_charlie_and_yutian_alias_are_equivalent() -> None:
    alias_record = CorpusRecord(
        id="summary-1",
        source="resume",
        section="Summary",
        date_range="",
        skills=["SQL", "Python"],
        text="Yutian Yang—data scientist and analytics engineer focused on SQL and Python projects.",
    )

    engine = SimpleSearchEngine()
    results = engine.search("Who is Charlie?", [alias_record], top_k=1)
    assert results and results[0].record.id == alias_record.id

    service = ResumeQAService(
        records=[alias_record],
        search_engine=engine,
        prompt_builder=PromptBuilder(),
        language_model=MockLanguageModel(),
    )
    answer = service.answer_question("Who is Charlie?")
    assert "I don't have" not in answer.answer
    assert "Yutian Yang" in answer.answer

