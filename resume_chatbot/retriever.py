"""Lightweight TF-IDF retriever for resume documents."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .data_loader import Document

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")


def _tokenise(text: str) -> list[str]:
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
    return [token for token in tokens if token not in _STOPWORDS]


def _compute_idf(tokenised_documents: Sequence[Sequence[str]]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for tokens in tokenised_documents:
        for token in set(tokens):
            df[token] += 1
    num_documents = len(tokenised_documents)
    return {
        token: math.log((1 + num_documents) / (1 + count)) + 1
        for token, count in df.items()
    }


@dataclass(frozen=True)
class RetrievedDocument:
    """A retrieved document and its similarity score."""

    document: Document
    score: float


class ResumeRetriever:
    """Perform simple TF-IDF retrieval over resume documents."""

    def __init__(self, documents: Iterable[Document]):
        self._documents: list[Document] = list(documents)
        if not self._documents:
            raise ValueError("ResumeRetriever requires at least one document")

        self._tokenised_docs: list[list[str]] = [
            _tokenise(document.content) for document in self._documents
        ]
        self._idf = _compute_idf(self._tokenised_docs)
        self._document_vectors: list[tuple[dict[str, float], float]] = [
            self._build_vector(tokens) for tokens in self._tokenised_docs
        ]

    def _build_vector(self, tokens: Sequence[str]) -> tuple[dict[str, float], float]:
        tf = Counter(tokens)
        vector: dict[str, float] = {}
        norm = 0.0
        length = float(len(tokens)) or 1.0
        for token, count in tf.items():
            if token not in self._idf:
                continue
            weight = (count / length) * self._idf[token]
            if weight <= 0:
                continue
            vector[token] = weight
            norm += weight * weight
        return vector, math.sqrt(norm) if norm else 0.0

    def _vectorise_query(self, query: str) -> tuple[dict[str, float], float]:
        tokens = _tokenise(query)
        tf = Counter(tokens)
        vector: dict[str, float] = {}
        norm = 0.0
        length = float(len(tokens)) or 1.0
        for token, count in tf.items():
            idf = self._idf.get(token)
            if idf is None:
                continue
            weight = (count / length) * idf
            vector[token] = weight
            norm += weight * weight
        return vector, math.sqrt(norm) if norm else 0.0

    @staticmethod
    def _cosine_similarity(
        vector_a: dict[str, float],
        norm_a: float,
        vector_b: dict[str, float],
        norm_b: float,
    ) -> float:
        if norm_a == 0 or norm_b == 0:
            return 0.0
        shared = set(vector_a) & set(vector_b)
        dot_product = sum(vector_a[token] * vector_b[token] for token in shared)
        return dot_product / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """Return the ``top_k`` most relevant documents for ``query``."""

        query_vector, query_norm = self._vectorise_query(query)
        scored: list[RetrievedDocument] = []
        for document, (doc_vector, doc_norm) in zip(
            self._documents, self._document_vectors
        ):
            score = self._cosine_similarity(query_vector, query_norm, doc_vector, doc_norm)
            if score <= 0:
                continue
            scored.append(RetrievedDocument(document=document, score=float(score)))
        scored.sort(key=lambda item: item.score, reverse=True)
        if top_k <= 0:
            return scored
        return scored[:top_k]
