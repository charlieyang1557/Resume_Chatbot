"""Language model abstractions for the resume chatbot."""

from __future__ import annotations

import os
import re
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple


try:  # pragma: no cover - import guard exercised indirectly
    import requests  # type: ignore[import]
except ImportError:  # pragma: no cover - exercised when dependency missing
    requests = None  # type: ignore[assignment]


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


class OllamaLLM(BaseLLM):
    """Wrapper around the Ollama API for local LLM inference."""

    def __init__(
        self,
        *,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        if requests is None:  # pragma: no cover - simple guard
            raise ImportError(
                "The 'requests' package is required for OllamaLLM. Install resume-chatbot with the 'ollama' extra."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def _format_prompt(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str],
        chat_history: Optional[ChatHistory],
    ) -> str:
        """Format the prompt for Ollama."""
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        
        # Add chat history if provided
        if chat_history:
            for user_turn, assistant_turn in chat_history:
                prompt_parts.append(f"Human: {user_turn}")
                prompt_parts.append(f"Assistant: {assistant_turn}")
        
        # Add the main context and question
        main_prompt = (
            "You are helping someone ask questions about a resume. "
            "Use only the supplied resume context to answer succinctly and professionally. "
            "Provide complete sentences and well-structured responses. "
            "IMPORTANT: Always use the correct pronouns as specified in the resume context. "
            "If pronouns are specified (e.g., he/him/his), use them consistently throughout your response. "
            "If the context is insufficient, say you don't know.\n\n"
            f"Resume context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        prompt_parts.append(main_prompt)
        
        return "\n\n".join(prompt_parts)

    def generate(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str] = None,
        chat_history: Optional[ChatHistory] = None,
    ) -> str:
        prompt = self._format_prompt(
            question,
            context,
            system_prompt=system_prompt,
            chat_history=chat_history,
        )
        
        try:
            assert requests is not None  # for type-checkers
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                    }
                },
                timeout=30,
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
        except json.JSONDecodeError:
            return "Error parsing response from Ollama"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


def create_llm(backend: str) -> BaseLLM:
    """Factory function that creates an LLM backend by name."""

    backend = backend.lower().strip()
    if backend == "simple":
        return SimpleLLM()
    if backend == "openai":
        return OpenAIChatLLM()
    if backend == "ollama":
        return OllamaLLM()
    raise ValueError("Unsupported LLM backend. Choose 'simple', 'openai', or 'ollama'.")
