"""Resume Chatbot package with lazy imports to avoid heavy dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ResumeChatbot",
    "ChatResult",
    "Document",
    "load_resume_documents",
    "load_resume_corpus",
    "load_knowledge_graph_documents",
    "ResumeRetriever",
    "RetrievedDocument",
    "BaseLLM",
    "SimpleLLM",
    "OpenAIChatLLM",
    "create_llm",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation helper
    if name in {"DEFAULT_SYSTEM_PROMPT", "ResumeChatbot", "ChatResult"}:
        module = import_module(".chatbot", __name__)
    elif name in {"Document", "load_resume_documents"}:
        module = import_module(".data_loader", __name__)
    elif name == "load_resume_corpus":
        module = import_module(".corpus", __name__)
    elif name == "load_knowledge_graph_documents":
        module = import_module(".knowledge_graph", __name__)
    elif name in {"ResumeRetriever", "RetrievedDocument"}:
        module = import_module(".retriever", __name__)
    elif name in {"BaseLLM", "SimpleLLM", "OpenAIChatLLM", "create_llm"}:
        module = import_module(".llm", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(module, name)
