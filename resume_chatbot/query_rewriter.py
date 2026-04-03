"""Lightweight query rewriting via Claude to improve retrieval quality.

Before retrieval, the user's raw query is rewritten to be self-contained
and specific to Charlie Yang's resume.  For example:

    "his work" → "Charlie Yang's professional work experience at Pinecone and PTC"
    "what ML stuff" → "What machine learning models and techniques has Charlie Yang used?"
"""

from __future__ import annotations

import os
import re
from typing import Optional, Sequence, Tuple

ChatHistory = Sequence[Tuple[str, str]]

_REWRITE_SYSTEM = (
    "You are a query rewriter for a resume chatbot about Charlie (Yutian) Yang. "
    "Rewrite the user's question to be self-contained, specific, and optimised "
    "for searching resume content. Resolve pronouns (he/his/him → Charlie Yang). "
    "Include relevant proper nouns from the conversation history if applicable. "
    "Output ONLY the rewritten question — no explanation, no quotes."
)

_NAME_PATTERN = re.compile(r"\bcharlie\b|\byang\b", re.IGNORECASE)


def _is_already_specific(query: str) -> bool:
    """Return True if the query is short and already mentions Charlie/Yang."""
    words = query.split()
    return bool(_NAME_PATTERN.search(query)) and len(words) < 8


class QueryRewriter:
    """Rewrite user queries for better retrieval using a lightweight Claude call."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for QueryRewriter."
            ) from exc

        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    def rewrite(
        self,
        query: str,
        history: Optional[ChatHistory] = None,
    ) -> str:
        """Rewrite ``query`` to be retrieval-friendly.

        Skips rewriting if the query already contains "Charlie" or "Yang"
        and is under 8 words (already specific enough).  Otherwise, always
        rewrites — long rambling queries benefit most from rewriting.
        """
        if _is_already_specific(query):
            return query

        messages: list[dict[str, str]] = []
        if history:
            for user_turn, assistant_turn in history:
                messages.append({"role": "user", "content": user_turn})
                messages.append({"role": "assistant", "content": assistant_turn})
        messages.append({"role": "user", "content": query})

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=150,
                temperature=0.0,
                system=_REWRITE_SYSTEM,
                messages=messages,  # type: ignore[arg-type]
            )
            rewritten = response.content[0].text.strip()
            return rewritten if rewritten else query
        except Exception:
            return query
