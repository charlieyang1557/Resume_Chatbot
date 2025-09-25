"""Language model abstractions for the resume chatbot."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple


ChatHistory = Sequence[Tuple[str, str]]


class BaseLLM(ABC):
    """Interface for language model backends."""

    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str] = None,
        chat_history: Optional[ChatHistory] = None,
    ) -> str:
        """Return an assistant response for ``question`` given ``context``."""


class SimpleLLM(BaseLLM):
    """Deterministic fallback LLM that summarises context snippets."""

    def __init__(self, fallback_response: str | None = None, max_sentences: int = 3):
        self.fallback_response = (
            fallback_response
            or "I could not find that information in the resume you provided."
        )
        self.max_sentences = max(1, max_sentences)

    def _summarise(self, context: str) -> str:
        sentences: list[str] = []
        pattern = re.compile(r"[^.!?]+[.!?]")
        for paragraph in context.split("\n\n"):
            for match in pattern.findall(paragraph.strip()):
                candidate = match.strip()
                if candidate:
                    sentences.append(candidate)
        if not sentences:
            fallback = context.replace("\n", " ").strip()
            if fallback:
                sentences.append(fallback)
        if not sentences:
            return ""
        summary = " ".join(sentences[: self.max_sentences]).strip()
        return summary

    def generate(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str] = None,
        chat_history: Optional[ChatHistory] = None,
    ) -> str:
        if not context.strip():
            return self.fallback_response
        summary = self._summarise(context)
        if not summary:
            return self.fallback_response
        response = (
            f"{summary}\n\nQuestion: {question.strip()}\n"
            "This answer is based on the provided resume sections."
        )
        return response


class OpenAIChatLLM(BaseLLM):
    """Wrapper around the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The 'openai' package is required for OpenAIChatLLM. Install resume-chatbot with the 'openai' extra."
            ) from exc

        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if self._client.api_key is None:
            raise ValueError("OPENAI_API_KEY is not set.")
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self._temperature = temperature
        self._max_tokens = max_tokens

    @staticmethod
    def _format_messages(
        question: str,
        context: str,
        *,
        system_prompt: Optional[str],
        chat_history: Optional[ChatHistory],
    ) -> List[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_history:
            for user_turn, assistant_turn in chat_history:
                messages.append({"role": "user", "content": user_turn})
                messages.append({"role": "assistant", "content": assistant_turn})
        user_prompt = (
            "You are helping someone ask questions about a resume. "
            "Use only the supplied resume context to answer succinctly. "
            "If the context is insufficient, say you don't know.\n\n"
            f"Resume context:\n{context}\n\nQuestion: {question}"
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str] = None,
        chat_history: Optional[ChatHistory] = None,
    ) -> str:
        messages = self._format_messages(
            question,
            context,
            system_prompt=system_prompt,
            chat_history=chat_history,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        choice = response.choices[0]
        return (choice.message.content or "").strip()


def create_llm(backend: str) -> BaseLLM:
    """Factory function that creates an LLM backend by name."""

    backend = backend.lower().strip()
    if backend == "simple":
        return SimpleLLM()
    if backend == "openai":
        return OpenAIChatLLM()
    raise ValueError("Unsupported LLM backend. Choose 'simple' or 'openai'.")
