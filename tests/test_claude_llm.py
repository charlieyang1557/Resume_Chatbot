"""Tests for ClaudeLLM with mocked Anthropic SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from resume_chatbot.llm import ClaudeLLM, create_llm


@pytest.fixture
def mock_anthropic():
    with patch("resume_chatbot.llm.ClaudeLLM.__init__", return_value=None) as mock_init:
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm._client = MagicMock()
        llm._model = "claude-sonnet-4-20250514"
        llm._temperature = 0.1
        llm._max_tokens = 1024
        yield llm


def test_generate_returns_text(mock_anthropic: ClaudeLLM):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Charlie worked at Pinecone.")]
    mock_anthropic._client.messages.create.return_value = mock_response

    result = mock_anthropic.generate(
        "Where did Charlie work?",
        "Charlie Yang interned at Pinecone in 2024.",
        system_prompt="Answer from resume context only.",
    )

    assert result == "Charlie worked at Pinecone."
    mock_anthropic._client.messages.create.assert_called_once()
    call_kwargs = mock_anthropic._client.messages.create.call_args
    assert call_kwargs.kwargs["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs.kwargs["system"] == "Answer from resume context only."


def test_generate_with_chat_history(mock_anthropic: ClaudeLLM):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="He used BigQuery and DBT.")]
    mock_anthropic._client.messages.create.return_value = mock_response

    result = mock_anthropic.generate(
        "What tools did he use?",
        "Charlie used BigQuery, DBT, and Sigma at Pinecone.",
        chat_history=[("Where did Charlie work?", "Pinecone")],
    )

    assert result == "He used BigQuery and DBT."
    messages = mock_anthropic._client.messages.create.call_args.kwargs["messages"]
    # history (2 msgs) + current question (1 msg) = 3
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_stream_yields_tokens(mock_anthropic: ClaudeLLM):
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["Charlie ", "worked ", "at Pinecone."])
    mock_anthropic._client.messages.stream.return_value = mock_stream

    tokens = list(mock_anthropic.stream(
        "Where did Charlie work?",
        "Pinecone internship 2024.",
    ))

    assert tokens == ["Charlie ", "worked ", "at Pinecone."]


def test_generate_no_system_prompt(mock_anthropic: ClaudeLLM):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Answer")]
    mock_anthropic._client.messages.create.return_value = mock_response

    mock_anthropic.generate("question", "context")

    call_kwargs = mock_anthropic._client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == ""


def test_create_llm_claude():
    with patch("resume_chatbot.llm.ClaudeLLM") as MockClaude:
        MockClaude.return_value = MagicMock()
        llm = create_llm("claude")
        MockClaude.assert_called_once()


def test_create_llm_simple():
    llm = create_llm("simple")
    from resume_chatbot.llm import SimpleLLM
    assert isinstance(llm, SimpleLLM)
