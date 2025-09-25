"""Object-oriented services for the FastAPI resume chatbot."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import re

from data_loader import CorpusLoader, CorpusRecord
from prompt_templates import PromptBuilder, PromptTemplates


class SearchEngine(Protocol):
    """Protocol describing the minimal search interface required."""

    def search(
        self, query: str, records: Sequence[CorpusRecord], top_k: int
    ) -> Sequence[SearchResult]:
        """Return the ``top_k`` most relevant records for ``query``."""


class LanguageModel(Protocol):
    """Protocol used by the service to generate answers."""

    def generate(self, question: str, context: Sequence[SearchResult]) -> str:
        """Return an answer for ``question`` using ``context`` records."""


@dataclass(frozen=True)
class SourceInfo:
    """Structured metadata about a retrieved source."""

    id: str
    source: str
    section: str
    date_range: str
    skills: List[str]
    relevance_score: float
    url: Optional[str] = None

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "SourceInfo":
        record = result.record
        return cls(
            id=record.id,
            source=record.source,
            section=record.section,
            date_range=record.date_range,
            skills=list(record.skills),
            relevance_score=result.score,
            url=record.url,
        )


@dataclass(frozen=True)
class SearchResult:
    """Minimal search result representation used by the service."""

    record: CorpusRecord
    score: float


@dataclass(frozen=True)
class AnswerResult:
    """Result of answering a question against the resume corpus."""

    answer: str
    sources: List[SourceInfo]
    validation: Dict[str, Any]
    prompt_length: int
    response_time: float


@dataclass(frozen=True)
class StatsResult:
    """Summary statistics about the loaded corpus."""

    total_records: int
    sources: Dict[str, int]
    top_skills: List[tuple[str, int]]
    total_characters: int
    avg_chars_per_record: float


class SimpleSearchEngine:
    """Lightweight keyword-based search used when FAISS is unavailable."""

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "what",
        "with",
    }

    def _tokenize(self, text: str) -> List[str]:
        return [
            token
            for token in re.findall(r"\b\w+\b", text.lower())
            if token not in self._STOPWORDS
        ]

    def _section_weight(self, section: str) -> float:
        section_lower = section.lower()
        weight = 0.0
        if "experience" in section_lower:
            weight += 0.25
        if "project" in section_lower:
            weight += 0.15
        if "education" in section_lower:
            weight += 0.1
        return weight

    def _skill_overlap(self, query_tokens: Iterable[str], skills: Sequence[str]) -> float:
        if not skills:
            return 0.0
        skill_tokens = {token for skill in skills for token in self._tokenize(skill)}
        if not skill_tokens:
            return 0.0
        matches = len(set(query_tokens).intersection(skill_tokens))
        return matches * 0.15

    def _keyword_overlap(self, query_counter: Counter[str], text_tokens: List[str]) -> float:
        if not query_counter:
            return 0.0
        text_counter = Counter(text_tokens)
        overlap = 0
        for token, count in query_counter.items():
            if token in text_counter:
                overlap += min(count, text_counter[token])
        return overlap / max(sum(query_counter.values()), 1)

    def search(
        self, query: str, records: Sequence[CorpusRecord], top_k: int = 3
    ) -> List[SearchResult]:
        query_tokens = self._tokenize(query)
        query_counter = Counter(query_tokens)

        scored_results: List[SearchResult] = []
        for record in records:
            text_tokens = self._tokenize(record.text)
            score = self._keyword_overlap(query_counter, text_tokens)

            if score <= 0 and not query_tokens:
                continue

            if query.lower() in record.text.lower():
                score += 0.4

            score += self._skill_overlap(query_tokens, record.skills)
            score += self._section_weight(record.section)

            if score > 0:
                scored_results.append(SearchResult(record=record, score=score))

        scored_results.sort(key=lambda item: item.score, reverse=True)
        return scored_results[:top_k]


class MockLanguageModel:
    """Deterministic language model used for local testing."""

    def __init__(self, *, max_bullets: int = 3) -> None:
        self._max_bullets = max(1, max_bullets)

    @staticmethod
    def _focus_sentence(sentence: str) -> str:
        """Trim boilerplate headings and keep the most action-oriented fragment."""

        focus_keywords = [
            "designed",
            "built",
            "developed",
            "conducted",
            "created",
            "implemented",
            "improved",
            "analyzed",
            "led",
        ]
        lower = sentence.lower()
        for keyword in focus_keywords:
            idx = lower.find(keyword)
            if idx > 0:
                focused = sentence[idx:].lstrip()
                if focused and focused[0].islower():
                    focused = focused[0].upper() + focused[1:]
                return focused
            if idx == 0:
                return sentence
        return sentence

    def _extract_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        cleaned: List[str] = []
        for sentence in sentences:
            candidate = sentence.strip()
            if candidate:
                candidate = candidate.replace("###", " ").replace("##", " ")
                candidate = re.sub(r"\s+", " ", candidate)
                candidate = re.sub(r"^[#*\-\s]+", "", candidate)
                candidate = re.sub(r"^\d+[.)]?\s+", "", candidate)
                candidate = self._focus_sentence(candidate)
                cleaned.append(candidate)
        return cleaned

    def _select_relevant_sentences(
        self, question: str, context: Sequence[SearchResult]
    ) -> List[Tuple[str, str]]:
        if not context:
            return []

        keywords = [
            token
            for token in re.findall(r"\b\w+\b", question.lower())
            if len(token) > 3
        ]
        results: List[Tuple[str, str]] = []
        seen: set[str] = set()
        for item in context:
            sentences = self._extract_sentences(item.record.text)
            if not sentences:
                continue

            matched: List[str] = []
            if keywords:
                for sentence in sentences:
                    lower = sentence.lower()
                    if any(keyword in lower for keyword in keywords):
                        matched.append(sentence)
            if not matched:
                matched = sentences[:1]

            citation = f"[{item.record.source}:{item.record.section}]"
            for sentence in matched:
                normalized = sentence.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                results.append((sentence, citation))
            if len(results) >= self._max_bullets:
                break

        return results[: self._max_bullets]

    def generate(self, question: str, context: Sequence[SearchResult]) -> str:
        if not context:
            return "I don't have that information in Charlie's records."

        highlights = self._select_relevant_sentences(question, context)
        if not highlights:
            first = context[0]
            citation = f"[{first.record.source}:{first.record.section}]"
            return (
                "I found relevant information, but the context is limited. "
                f"Please review the source directly {citation}"
            )

        bullet_lines = [f"- {sentence} {citation}" for sentence, citation in highlights]
        response_parts = [
            "Here is what Charlie's records highlight:",
            *bullet_lines,
            "Let me know if you need deeper detail from any specific project.",
        ]
        return "\n".join(response_parts)


class ResumeQAService:
    """High-level orchestration for answering resume questions."""

    def __init__(
        self,
        *,
        records: Sequence[CorpusRecord],
        search_engine: SearchEngine,
        prompt_builder: PromptBuilder,
        language_model: LanguageModel,
        templates: Optional[PromptTemplates] = None,
    ) -> None:
        if not records:
            raise ValueError("ResumeQAService requires at least one corpus record")

        self._records = list(records)
        self._search_engine = search_engine
        self._templates = templates or PromptTemplates()
        self._language_model = language_model
        self._base_config = replace(prompt_builder.config)

    def _builder_for_template(self, template: str) -> PromptBuilder:
        template = (template or "").lower().strip()
        if template == "hiring_manager":
            system_prompt = self._templates.hiring_manager_prompt()
        elif template == "technical":
            system_prompt = self._templates.technical_prompt()
        elif template == "general":
            system_prompt = self._templates.general_prompt()
        else:
            system_prompt = self._templates.recruiter_prompt()

        config = replace(self._base_config, system_prompt=system_prompt)
        return PromptBuilder(config)

    def answer_question(
        self, question: str, *, top_k: int = 3, template: str = "recruiter"
    ) -> AnswerResult:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        start = time.perf_counter()
        results = list(self._search_engine.search(question, self._records, top_k))
        builder = self._builder_for_template(template)
        prompt = builder.build_prompt(question, results)
        answer = self._language_model.generate(question, results)
        validation = builder.validate_response(answer, has_context=bool(results))
        elapsed = time.perf_counter() - start

        sources = [SourceInfo.from_search_result(result) for result in results]
        return AnswerResult(
            answer=answer,
            sources=sources,
            validation=validation,
            prompt_length=len(prompt),
            response_time=elapsed,
        )

    def stats(self) -> StatsResult:
        sources: Dict[str, int] = {}
        skills_count: Dict[str, int] = {}
        total_chars = 0

        for record in self._records:
            sources[record.source] = sources.get(record.source, 0) + 1
            for skill in record.skills:
                skills_count[skill] = skills_count.get(skill, 0) + 1
            total_chars += len(record.text)

        total_records = len(self._records)
        avg_chars = total_chars / total_records if total_records else 0.0
        top_skills = sorted(
            skills_count.items(), key=lambda item: item[1], reverse=True
        )[:10]

        return StatsResult(
            total_records=total_records,
            sources=sources,
            top_skills=top_skills,
            total_characters=total_chars,
            avg_chars_per_record=avg_chars,
        )

    @property
    def records(self) -> Sequence[CorpusRecord]:
        return tuple(self._records)


class ResumeQAApplication:
    """Coordinates loading of the corpus and exposes the OOP service."""

    def __init__(
        self,
        corpus_path: Path,
        *,
        search_engine: Optional[SearchEngine] = None,
        language_model: Optional[LanguageModel] = None,
    ) -> None:
        self._corpus_path = Path(corpus_path)
        self._search_engine = search_engine or SimpleSearchEngine()
        self._language_model = language_model or MockLanguageModel()
        self._service: Optional[ResumeQAService] = None

    def initialize(self) -> None:
        loader = CorpusLoader(self._corpus_path)
        records = loader.load()
        if not records:
            raise RuntimeError(f"No records loaded from corpus: {self._corpus_path}")

        builder = PromptBuilder()
        self._service = ResumeQAService(
            records=records,
            search_engine=self._search_engine,
            prompt_builder=builder,
            language_model=self._language_model,
        )

    def ensure_ready(self) -> ResumeQAService:
        if self._service is None:
            raise RuntimeError("ResumeQAApplication has not been initialized")
        return self._service

    def answer_question(
        self, question: str, *, top_k: int = 3, template: str = "recruiter"
    ) -> AnswerResult:
        service = self.ensure_ready()
        return service.answer_question(question, top_k=top_k, template=template)

    def stats(self) -> StatsResult:
        service = self.ensure_ready()
        return service.stats()

    @property
    def records_loaded(self) -> int:
        if self._service is None:
            return 0
        return len(self._service.records)

    @property
    def ready(self) -> bool:
        return self._service is not None

    @property
    def corpus_path(self) -> Path:
        return self._corpus_path


__all__ = [
    "AnswerResult",
    "LanguageModel",
    "MockLanguageModel",
    "ResumeQAApplication",
    "ResumeQAService",
    "SearchResult",
    "SearchEngine",
    "SimpleSearchEngine",
    "SourceInfo",
    "StatsResult",
]
