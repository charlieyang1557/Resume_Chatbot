from resume_chatbot.chatbot import DEFAULT_SYSTEM_PROMPT, ChatResult, ResumeChatbot
from resume_chatbot.data_loader import Document
from resume_chatbot.llm import SimpleLLM
from resume_chatbot.retriever import ResumeRetriever


def build_chatbot(documents: list[Document]) -> ResumeChatbot:
    retriever = ResumeRetriever(documents)
    llm = SimpleLLM()
    return ResumeChatbot(retriever=retriever, llm=llm, system_prompt=DEFAULT_SYSTEM_PROMPT)


def test_chatbot_returns_answer_from_context():
    documents = [
        Document(
            content="Alex has over 7 years of experience working with Python and large language models.",
            metadata={"source": "resume.md", "title": "Summary"},
        )
    ]
    chatbot = build_chatbot(documents)

    response = chatbot.ask("How much Python experience does Alex have?", top_k=1)

    assert isinstance(response, ChatResult)
    assert "7 years" in response.answer
    assert response.documents[0].document.metadata["title"] == "Summary"


def test_chatbot_uses_fallback_when_no_context():
    documents = [
        Document(
            content="Alex is based in Seattle.",
            metadata={"source": "resume.md", "title": "Contact"},
        )
    ]
    chatbot = build_chatbot(documents)

    response = chatbot.ask("What is Alex's favourite programming language?", top_k=1)

    assert "could not" in response.answer.lower()
    assert response.documents == []


def test_chat_history_is_capped():
    documents = [
        Document(
            content="Alex mentors engineers.",
            metadata={"source": "resume.md", "title": "Community"},
        )
    ]
    chatbot = ResumeChatbot(
        retriever=ResumeRetriever(documents),
        llm=SimpleLLM(),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        history_limit=2,
    )

    chatbot.ask("Tell me about Alex's mentoring.")
    chatbot.ask("Does Alex enjoy teaching?")
    chatbot.ask("What leadership has Alex demonstrated?")

    assert len(chatbot.history) == 2
