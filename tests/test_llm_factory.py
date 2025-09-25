import pytest

from resume_chatbot.llm import SimpleLLM, create_llm


def test_create_llm_simple_backend() -> None:
    llm = create_llm("simple")
    assert isinstance(llm, SimpleLLM)


def test_create_llm_invalid_backend() -> None:
    with pytest.raises(ValueError):
        create_llm("unsupported")
