"""FastAPI application entry-point for the resume chatbot API."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - FastAPI isn't available in all environments
    FASTAPI_AVAILABLE = False

from resume_chatbot.qa_service import AnswerResult, ResumeQAApplication, SourceInfo, StatsResult


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    """Incoming request model for the ``/ask`` endpoint."""

    question: str = Field(
        ..., min_length=1, max_length=500, description="Question about the candidate"
    )
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of context chunks to retrieve"
    )
    template: str = Field(
        default="recruiter",
        description="Prompt template: recruiter, hiring_manager, technical, general",
    )


class SourceResponse(BaseModel):
    id: str
    source: str
    section: str
    date_range: str
    skills: List[str]
    relevance_score: float
    url: Optional[str] = None


class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceResponse] = Field(
        ..., description="Source documents used"
    )
    response_time: float = Field(..., description="Response time in seconds")
    validation: Dict[str, Any] = Field(
        ..., description="Response validation results"
    )
    prompt_length: int = Field(..., description="Length of generated prompt")


class StatsResponse(BaseModel):
    total_records: int
    sources: Dict[str, int]
    top_skills: List[tuple[str, int]]
    total_characters: int
    avg_chars_per_record: float


def _build_source_response(source: SourceInfo) -> SourceResponse:
    return SourceResponse(
        id=source.id,
        source=source.source,
        section=source.section,
        date_range=source.date_range,
        skills=source.skills,
        relevance_score=source.relevance_score,
        url=source.url,
    )


def _build_stats_response(stats: StatsResult) -> StatsResponse:
    return StatsResponse(
        total_records=stats.total_records,
        sources=stats.sources,
        top_skills=stats.top_skills,
        total_characters=stats.total_characters,
        avg_chars_per_record=stats.avg_chars_per_record,
    )


def _build_question_response(result: AnswerResult) -> QuestionResponse:
    return QuestionResponse(
        answer=result.answer,
        sources=[_build_source_response(source) for source in result.sources],
        response_time=result.response_time,
        validation=result.validation,
        prompt_length=result.prompt_length,
    )


def create_app(corpus_path: str = "corpus_original.jsonl") -> FastAPI:
    """Create a FastAPI application backed by :class:`ResumeQAApplication`."""

    if not FASTAPI_AVAILABLE:  # pragma: no cover - avoids import errors during tests
        raise ImportError(
            "FastAPI not available. Install with: pip install fastapi uvicorn"
        )

    qa_app = ResumeQAApplication(Path(corpus_path))

    app = FastAPI(
        title="RAG Resume Q&A API",
        description="Retrieval-Augmented Generation system for answering questions about resumes",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        try:
            logger.info("🚀 Initializing RAG Resume Q&A API...")
            qa_app.initialize()
            logger.info("✅ API initialized with %s records", qa_app.records_loaded)
        except Exception as exc:  # pragma: no cover - logging only
            logger.error("❌ Failed to initialize API: %s", exc)

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        return {
            "status": "healthy" if qa_app.ready else "not_initialized",
            "records_loaded": qa_app.records_loaded,
            "timestamp": time.time(),
        }

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats() -> StatsResponse:
        if not qa_app.ready:
            raise HTTPException(status_code=503, detail="API not initialized")
        return _build_stats_response(qa_app.stats())

    @app.post("/ask", response_model=QuestionResponse)
    async def ask_question(request: QuestionRequest) -> QuestionResponse:
        if not qa_app.ready:
            raise HTTPException(status_code=503, detail="API not initialized")

        try:
            result = qa_app.answer_question(
                request.question, top_k=request.top_k, template=request.template
            )
            return _build_question_response(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - ensures HTTP error in production
            logger.error("Error processing question: %s", exc)
            raise HTTPException(status_code=500, detail="Error processing question")

    @app.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "message": "RAG Resume Q&A API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats",
            "ask": "/ask",
        }

    return app


__all__ = ["create_app"]
