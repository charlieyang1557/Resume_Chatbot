"""Tests for hybrid retrieval: BM25, Dense (mocked), RRF fusion."""

from __future__ import annotations

import pytest

from resume_chatbot.data_loader import Document
from resume_chatbot.retriever import RetrievedDocument
from resume_chatbot.hybrid_retriever import _rrf_scores, _RRF_K, HybridRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOC_A = Document(content="Pinecone vector database internship", metadata={"title": "Pinecone"})
DOC_B = Document(content="PTC anomaly detection Prophet", metadata={"title": "PTC"})
DOC_C = Document(content="UC Davis statistics education", metadata={"title": "Education"})


class FakeRetriever:
    """Deterministic retriever returning pre-set results."""

    def __init__(self, results: list[RetrievedDocument]) -> None:
        self._results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# RRF math
# ---------------------------------------------------------------------------

def test_rrf_scores_basic():
    results = [
        RetrievedDocument(document=DOC_A, score=10.0),
        RetrievedDocument(document=DOC_B, score=5.0),
    ]
    scores = _rrf_scores(results)

    assert scores[id(DOC_A)] == pytest.approx(1.0 / (_RRF_K + 1))
    assert scores[id(DOC_B)] == pytest.approx(1.0 / (_RRF_K + 2))


def test_rrf_rank1_always_highest():
    results = [
        RetrievedDocument(document=DOC_A, score=1.0),
        RetrievedDocument(document=DOC_B, score=0.5),
        RetrievedDocument(document=DOC_C, score=0.1),
    ]
    scores = _rrf_scores(results)
    assert scores[id(DOC_A)] > scores[id(DOC_B)] > scores[id(DOC_C)]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

def test_hybrid_alpha_validation():
    sparse = FakeRetriever([])
    dense = FakeRetriever([])

    with pytest.raises(ValueError, match="alpha must be in"):
        HybridRetriever(sparse, dense, alpha=1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="alpha must be in"):
        HybridRetriever(sparse, dense, alpha=-0.1)  # type: ignore[arg-type]


def test_hybrid_pure_sparse():
    sparse_results = [
        RetrievedDocument(document=DOC_A, score=10.0),
        RetrievedDocument(document=DOC_B, score=5.0),
    ]
    dense_results = [
        RetrievedDocument(document=DOC_C, score=0.9),
    ]

    hybrid = HybridRetriever(
        sparse=FakeRetriever(sparse_results),  # type: ignore[arg-type]
        dense=FakeRetriever(dense_results),  # type: ignore[arg-type]
        alpha=0.0,
    )
    results = hybrid.retrieve("test", top_k=3)

    # With alpha=0.0, dense contributes nothing — sparse ranking dominates
    assert results[0].document is DOC_A
    assert results[1].document is DOC_B


def test_hybrid_pure_dense():
    sparse_results = [
        RetrievedDocument(document=DOC_C, score=5.0),
    ]
    dense_results = [
        RetrievedDocument(document=DOC_A, score=0.95),
        RetrievedDocument(document=DOC_B, score=0.80),
    ]

    hybrid = HybridRetriever(
        sparse=FakeRetriever(sparse_results),  # type: ignore[arg-type]
        dense=FakeRetriever(dense_results),  # type: ignore[arg-type]
        alpha=1.0,
    )
    results = hybrid.retrieve("test", top_k=3)

    # With alpha=1.0, sparse contributes nothing — dense ranking dominates
    assert results[0].document is DOC_A
    assert results[1].document is DOC_B


def test_hybrid_balanced_fuses_both():
    sparse_results = [
        RetrievedDocument(document=DOC_B, score=10.0),  # B is rank 1 sparse
        RetrievedDocument(document=DOC_A, score=5.0),   # A is rank 2 sparse
    ]
    dense_results = [
        RetrievedDocument(document=DOC_A, score=0.95),  # A is rank 1 dense
        RetrievedDocument(document=DOC_B, score=0.80),  # B is rank 2 dense
    ]

    hybrid = HybridRetriever(
        sparse=FakeRetriever(sparse_results),  # type: ignore[arg-type]
        dense=FakeRetriever(dense_results),  # type: ignore[arg-type]
        alpha=0.5,
    )
    results = hybrid.retrieve("test", top_k=2)

    # Both A and B are rank 1 in one list and rank 2 in the other
    # With alpha=0.5, they should have equal fused scores
    assert len(results) == 2
    assert results[0].score == pytest.approx(results[1].score)


def test_hybrid_top_k_limits_output():
    results_list = [
        RetrievedDocument(document=DOC_A, score=10.0),
        RetrievedDocument(document=DOC_B, score=5.0),
        RetrievedDocument(document=DOC_C, score=1.0),
    ]
    hybrid = HybridRetriever(
        sparse=FakeRetriever(results_list),  # type: ignore[arg-type]
        dense=FakeRetriever(results_list),  # type: ignore[arg-type]
        alpha=0.5,
    )
    results = hybrid.retrieve("test", top_k=1)
    assert len(results) == 1


def test_hybrid_alpha_setter():
    hybrid = HybridRetriever(
        sparse=FakeRetriever([]),  # type: ignore[arg-type]
        dense=FakeRetriever([]),  # type: ignore[arg-type]
        alpha=0.5,
    )
    hybrid.alpha = 0.8
    assert hybrid.alpha == 0.8

    with pytest.raises(ValueError):
        hybrid.alpha = 2.0
