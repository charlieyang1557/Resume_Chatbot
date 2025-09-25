"""High-level orchestration of retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .llm import BaseLLM
from .retriever import RetrievedDocument, ResumeRetriever

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a single candidate's resume. "
    "Use only the provided resume context. If the answer is not in the context, say you don't know. "
    "Always use the correct pronouns as specified in the resume context. If pronouns are specified (e.g., he/him/his), use them consistently throughout your response."
)


@dataclass(frozen=True)
class ChatResult:
    """Result of a chat interaction."""

    answer: str
    documents: List[RetrievedDocument]


class ResumeChatbot:
    """Combine a retriever and an LLM to answer resume questions."""

    def __init__(
        self,
        retriever: ResumeRetriever,
        llm: BaseLLM,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_limit: int = 5,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._system_prompt = system_prompt
        self._history_limit = max(0, history_limit)
        self._history: list[tuple[str, str]] = []

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

    def ask(self, question: str, *, top_k: int = 3) -> ChatResult:
        """Answer ``question`` using the resume corpus."""

        retrieved = self._retriever.retrieve(question, top_k=top_k)
        context = self._format_context(retrieved)
        answer = self._llm.generate(
            question,
            context,
            system_prompt=self._system_prompt,
            chat_history=tuple(self._history),
        )
        if self._history_limit:
            self._history.append((question, answer))
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]
        return ChatResult(answer=answer, documents=list(retrieved))

    def reset_history(self) -> None:
        """Clear the stored chat history."""

        self._history.clear()

    @property
    def history(self) -> Tuple[tuple[str, str], ...]:
        """Return the stored chat history as ``(question, answer)`` pairs."""

        return tuple(self._history)
