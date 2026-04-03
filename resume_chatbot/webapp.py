"""FastAPI-powered web interface for the resume chatbot."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fastapi import Request

from .chatbot import ResumeChatbot
from .corpus import load_resume_corpus
from .llm import SimpleLLM, create_llm
from .rate_limit import RateLimiter
from .retriever import ResumeRetriever

STARTER_QUESTIONS = [
    "How did Charlie reduce onboarding time by 30%?",
    "Tell me about the anomaly detection work at PTC",
    "What was Charlie's impact at Pinecone?",
]


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chatbot.")


class Source(BaseModel):
    source: str
    title: str | None = None
    score: float
    content: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


class ConfigResponse(BaseModel):
    hero_title: str
    hero_subtitle: str
    starter_questions: List[str]


def _build_chatbot(resume_directory: Path, llm_backend: str) -> ResumeChatbot:
    documents = load_resume_corpus(resume_directory)
    if not documents:
        raise RuntimeError(
            "No resume documents found. Add Markdown/text files to the resume directory."
        )

    retriever: Any
    llm = create_llm(llm_backend)

    # Try to build hybrid retriever, fall back to TF-IDF
    try:
        from .hybrid_retriever import BM25Retriever, DenseRetriever, HybridRetriever

        alpha = float(os.getenv("RETRIEVER_ALPHA", "0.5"))
        sparse = BM25Retriever(documents)
        dense = DenseRetriever(documents)
        retriever = HybridRetriever(sparse, dense, alpha=alpha)
    except ImportError:
        retriever = ResumeRetriever(documents)

    # Try to build query rewriter (requires Claude API key)
    query_rewriter = None
    if llm_backend == "claude" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            from .query_rewriter import QueryRewriter

            query_rewriter = QueryRewriter()
        except ImportError:
            pass

    return ResumeChatbot(
        retriever=retriever, llm=llm, query_rewriter=query_rewriter
    )


def _sources_from_result(result: Any) -> List[Dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for item in result.documents:
        metadata = item.document.metadata
        sources.append({
            "source": metadata.get("source", "resume"),
            "title": metadata.get("title"),
            "score": round(item.score, 4),
            "content": item.document.content[:300],
        })
    return sources


def create_app(
    *,
    resume_directory: Path = Path("data/resume"),
    llm_backend: str | None = None,
    top_k: int = 5,
) -> FastAPI:
    """Create a FastAPI app backed by the resume chatbot."""

    backend = llm_backend or os.getenv("LLM_BACKEND", "simple")
    chatbot = _build_chatbot(resume_directory, backend)
    app = FastAPI(title="Resume Chatbot", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.chatbot = chatbot
    app.state.top_k = top_k
    app.state.limiter = RateLimiter(
        per_ip_limit=int(os.getenv("RATE_LIMIT_PER_IP", "20")),
        daily_limit=int(os.getenv("RATE_LIMIT_DAILY", "100")),
    )

    # -----------------------------------------------------------------------
    # API routes
    # -----------------------------------------------------------------------

    def _check_rate_limit(request: Request) -> None:
        ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
        ip = ip.split(",")[0].strip()
        error = app.state.limiter.check(ip)
        if error:
            raise HTTPException(status_code=429, detail=error)

    @app.get("/api/health")
    async def health():
        limiter: RateLimiter = app.state.limiter
        return {
            "status": "ok",
            "daily_usage": f"{limiter.daily_count}/{limiter.daily_limit}",
        }

    @app.get("/api/config", response_model=ConfigResponse)
    async def config():
        return ConfigResponse(
            hero_title="Charlie (Yutian) Yang",
            hero_subtitle="Data Scientist & ML Engineer",
            starter_questions=STARTER_QUESTIONS,
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, raw_request: Request) -> ChatResponse:
        _check_rate_limit(raw_request)
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        result = app.state.chatbot.ask(message, top_k=app.state.top_k)
        sources = _sources_from_result(result)
        return ChatResponse(
            answer=result.answer,
            sources=[Source(**s) for s in sources],
        )

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest, raw_request: Request):
        _check_rate_limit(raw_request)
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        def event_generator():
            for token, documents in app.state.chatbot.ask_stream(
                message, top_k=app.state.top_k
            ):
                if token is not None:
                    payload = json.dumps({"type": "token", "content": token})
                    yield f"data: {payload}\n\n"
                if documents is not None:
                    sources = []
                    for item in documents:
                        metadata = item.document.metadata
                        sources.append({
                            "source": metadata.get("source", "resume"),
                            "title": metadata.get("title"),
                            "score": round(item.score, 4),
                            "content": item.document.content[:300],
                        })
                    payload = json.dumps({"type": "sources", "sources": sources})
                    yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_generator(), media_type="text/event-stream"
        )

    # -----------------------------------------------------------------------
    # Static files (React frontend build) — must be last
    # -----------------------------------------------------------------------

    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount(
            "/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend"
        )

    return app


app = create_app()

__all__ = ["create_app", "app"]
