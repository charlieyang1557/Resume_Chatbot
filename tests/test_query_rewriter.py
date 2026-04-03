"""Tests for query rewriter with mocked Claude calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from resume_chatbot.query_rewriter import QueryRewriter, _is_already_specific


# ---------------------------------------------------------------------------
# Skip heuristic
# ---------------------------------------------------------------------------

def test_already_specific_short_with_name():
    assert _is_already_specific("Charlie's skills?") is True
    assert _is_already_specific("Yang education") is True


def test_not_specific_no_name():
    assert _is_already_specific("What ML models has he built?") is False


def test_not_specific_long_with_name():
    # Contains "Charlie" but >8 words — should rewrite
    assert _is_already_specific(
        "Tell me everything about what Charlie did during his internship at Pinecone"
    ) is False


def test_not_specific_short_no_name():
    assert _is_already_specific("his work") is False


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rewriter():
    with patch("resume_chatbot.query_rewriter.QueryRewriter.__init__", return_value=None):
        rewriter = QueryRewriter.__new__(QueryRewriter)
        rewriter._client = MagicMock()
        rewriter._model = "claude-sonnet-4-20250514"
        yield rewriter


def test_rewrite_calls_claude(mock_rewriter: QueryRewriter):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="What was Charlie Yang's impact at Pinecone?")]
    mock_rewriter._client.messages.create.return_value = mock_response

    result = mock_rewriter.rewrite("his impact at that vector db company")

    assert result == "What was Charlie Yang's impact at Pinecone?"
    mock_rewriter._client.messages.create.assert_called_once()


def test_rewrite_skips_specific_query(mock_rewriter: QueryRewriter):
    result = mock_rewriter.rewrite("Charlie's skills?")

    assert result == "Charlie's skills?"
    mock_rewriter._client.messages.create.assert_not_called()


def test_rewrite_includes_history(mock_rewriter: QueryRewriter):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="What tools did Charlie use at Pinecone?")]
    mock_rewriter._client.messages.create.return_value = mock_response

    result = mock_rewriter.rewrite(
        "What tools?",
        history=[("Tell me about Pinecone", "Charlie interned at Pinecone...")],
    )

    messages = mock_rewriter._client.messages.create.call_args.kwargs["messages"]
    assert len(messages) == 3  # 2 history + 1 current


def test_rewrite_fallback_on_error(mock_rewriter: QueryRewriter):
    mock_rewriter._client.messages.create.side_effect = Exception("API error")

    result = mock_rewriter.rewrite("his work experience")

    assert result == "his work experience"


def test_rewrite_fallback_on_empty_response(mock_rewriter: QueryRewriter):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="")]
    mock_rewriter._client.messages.create.return_value = mock_response

    result = mock_rewriter.rewrite("his work")

    assert result == "his work"
