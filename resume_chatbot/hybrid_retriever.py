"""Hybrid retrieval combining BM25 (sparse) and dense embeddings via Reciprocal Rank Fusion.

Architecture
------------
* **BM25Retriever** – keyword-based sparse retrieval using ``rank_bm25``.
* **DenseRetriever** – semantic retrieval using ``fastembed`` embeddings + FAISS.
* **HybridRetriever** – fuses both result lists with Reciprocal Rank Fusion (RRF).

The ``alpha`` parameter (0-1) controls the weighting between sparse and dense:

    score = alpha * rrf_dense + (1 - alpha) * rrf_sparse

* ``alpha = 0.0``  →  pure BM25 (keyword matching only).  Best when queries use
  exact terms that appear verbatim in the resume (e.g. "BigQuery", "Prophet").
* ``alpha = 1.0``  →  pure dense (semantic matching only).  Best when queries are
  conceptual or use synonyms (e.g. "machine learning work" matching "Random Forests").
* ``alpha = 0.5``  →  balanced.  Generally the safest default — captures both exact
  keyword hits and semantic similarity.

RRF is rank-based rather than score-based, which naturally handles the different
score distributions of BM25 (unbounded) vs cosine similarity (0-1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .data_loader import Document
from .retriever import RetrievedDocument

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_PATTERN.findall(text)]


# ---------------------------------------------------------------------------
# BM25 sparse retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Sparse retriever using Okapi BM25 from ``rank_bm25``."""

    def __init__(self, documents: Sequence[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._documents = list(documents)
        tokenized = [_tokenize(doc.content) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievedDocument] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(
                RetrievedDocument(document=self._documents[idx], score=score)
            )
        return results


# ---------------------------------------------------------------------------
# Dense (semantic) retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Semantic retriever using ``fastembed`` embeddings and FAISS.

    Uses ``fastembed`` instead of ``sentence-transformers`` to avoid the
    heavy ``torch`` dependency (~2 GB).  The ``BAAI/bge-small-en-v1.5``
    model produces 384-dim embeddings and runs on CPU via ONNX.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        import faiss
        from fastembed import TextEmbedding

        self._documents = list(documents)
        self._model = TextEmbedding(model_name=model_name)

        texts = [doc.content for doc in self._documents]
        embeddings = np.array(list(self._model.embed(texts)), dtype=np.float32)

        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        query_vec = np.array(
            list(self._model.embed([query])), dtype=np.float32
        )
        import faiss

        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, min(top_k, len(self._documents)))
        results: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                RetrievedDocument(
                    document=self._documents[int(idx)], score=float(score)
                )
            )
        return results


# ---------------------------------------------------------------------------
# Hybrid retriever with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

_RRF_K = 60  # Standard RRF constant


def _rrf_scores(
    results: Sequence[RetrievedDocument],
) -> dict[int, float]:
    """Compute RRF scores keyed by ``id(document)``."""
    return {
        id(item.document): 1.0 / (_RRF_K + rank)
        for rank, item in enumerate(results, start=1)
    }


class HybridRetriever:
    """Fuse sparse (BM25) and dense retrieval results with Reciprocal Rank Fusion.

    Parameters
    ----------
    sparse:
        A retriever with a ``.retrieve(query, top_k)`` method (typically BM25).
    dense:
        A retriever with a ``.retrieve(query, top_k)`` method (typically FAISS).
    alpha:
        Weight for dense scores (0 = pure sparse, 1 = pure dense, 0.5 = balanced).
        See module docstring for guidance on choosing a value.
    """

    def __init__(
        self,
        sparse: BM25Retriever,
        dense: DenseRetriever,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._sparse = sparse
        self._dense = dense
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {value}")
        self._alpha = value

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        # Fetch more candidates than needed so RRF has enough to fuse
        fetch_k = top_k * 3

        sparse_results = self._sparse.retrieve(query, top_k=fetch_k)
        dense_results = self._dense.retrieve(query, top_k=fetch_k)

        sparse_rrf = _rrf_scores(sparse_results)
        dense_rrf = _rrf_scores(dense_results)

        # Collect all unique documents
        doc_map: dict[int, Document] = {}
        for item in sparse_results + dense_results:
            doc_map[id(item.document)] = item.document

        # Fuse scores
        fused: list[RetrievedDocument] = []
        for doc_id, document in doc_map.items():
            score = (
                self._alpha * dense_rrf.get(doc_id, 0.0)
                + (1 - self._alpha) * sparse_rrf.get(doc_id, 0.0)
            )
            fused.append(RetrievedDocument(document=document, score=score))

        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]
