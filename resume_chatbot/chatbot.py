"""High-level orchestration of retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Protocol, Sequence, Tuple, Union

from .llm import BaseLLM
from .retriever import RetrievedDocument

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about Charlie Yang's resume. "
    "Use only the provided resume context. Cite specific metrics and outcomes when "
    "available (e.g., '15% improvement', '10% churn reduction'). If the context "
    "doesn't contain the answer, say so clearly. Use he/him pronouns consistently."
)


class Retriever(Protocol):
    """Any object with a ``.retrieve(query, top_k)`` method."""

    def retrieve(self, query: str, top_k: int = ...) -> List[RetrievedDocument]: ...


class QueryRewriterProtocol(Protocol):
    """Any object with a ``.rewrite(query, history)`` method."""

    def rewrite(
        self,
        query: str,
        history: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> str: ...


@dataclass(frozen=True)
class ChatResult:
    """Result of a chat interaction."""

    answer: str
    documents: List[RetrievedDocument]


class ResumeChatbot:
    """Combine a retriever and an LLM to answer resume questions."""

    def __init__(
        self,
        retriever: Any,
        llm: BaseLLM,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_limit: int = 5,
        query_rewriter: Optional[Any] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._system_prompt = system_prompt
        self._history_limit = max(0, history_limit)
        self._history: list[tuple[str, str]] = []
        self._query_rewriter = query_rewriter

    @staticmethod
    def _format_context(retrieved_documents: Sequence[RetrievedDocument]) -> str:
        sections: list[str] = []
        for item in retrieved_documents:
            metadata = item.document.metadata
            title = metadata.get("title")
            source = metadata.get("source", "resume")
            header = f"[{source}]"
            if title:
                header = f"[{source} :: {title}]"
            sections.append(f"{header}\n{item.document.content}")
        return "\n\n".join(sections)

    def _rewrite_query(self, question: str) -> str:
        if self._query_rewriter is None:
            return question
        return self._query_rewriter.rewrite(question, history=tuple(self._history))

    def _append_history(self, question: str, answer: str) -> None:
        if self._history_limit:
            self._history.append((question, answer))
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]

    def ask(self, question: str, *, top_k: int = 3) -> ChatResult:
        """Answer ``question`` using the resume corpus."""

        rewritten = self._rewrite_query(question)
        retrieved = self._retriever.retrieve(rewritten, top_k=top_k)
        context = self._format_context(retrieved)
        answer = self._llm.generate(
            question,
            context,
            system_prompt=self._system_prompt,
            chat_history=tuple(self._history),
        )
        self._append_history(question, answer)
        return ChatResult(answer=answer, documents=list(retrieved))

    def ask_stream(
        self, question: str, *, top_k: int = 3
    ) -> Generator[Union[tuple[str, None], tuple[None, List[RetrievedDocument]]], None, None]:
        """Stream answer tokens, then yield the source documents at the end.

        Yields ``(token, None)`` for each text chunk, then ``(None, documents)``
        as the final item.  Falls back to non-streaming if the LLM doesn't
        support ``stream()``.
        """

        rewritten = self._rewrite_query(question)
        retrieved = self._retriever.retrieve(rewritten, top_k=top_k)
        context = self._format_context(retrieved)

        if hasattr(self._llm, "stream"):
            full_answer: list[str] = []
            for token in self._llm.stream(
                question,
                context,
                system_prompt=self._system_prompt,
                chat_history=tuple(self._history),
            ):
                full_answer.append(token)
                yield (token, None)
            self._append_history(question, "".join(full_answer))
        else:
            answer = self._llm.generate(
                question,
                context,
                system_prompt=self._system_prompt,
                chat_history=tuple(self._history),
            )
            yield (answer, None)
            self._append_history(question, answer)

        yield (None, list(retrieved))

    def reset_history(self) -> None:
        """Clear the stored chat history."""
        self._history.clear()

    @property
    def history(self) -> Tuple[tuple[str, str], ...]:
        """Return the stored chat history as ``(question, answer)`` pairs."""
        return tuple(self._history)
