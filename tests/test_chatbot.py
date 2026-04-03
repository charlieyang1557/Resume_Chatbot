from resume_chatbot.chatbot import DEFAULT_SYSTEM_PROMPT, ChatResult, ResumeChatbot
from resume_chatbot.data_loader import Document
from resume_chatbot.llm import SimpleLLM
from resume_chatbot.retriever import ResumeRetriever


def build_chatbot(documents: list[Document], **kwargs) -> ResumeChatbot:
    retriever = ResumeRetriever(documents)
    llm = SimpleLLM()
    return ResumeChatbot(
        retriever=retriever, llm=llm, system_prompt=DEFAULT_SYSTEM_PROMPT, **kwargs
    )


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
    chatbot = build_chatbot(documents, history_limit=2)

    chatbot.ask("Tell me about Alex's mentoring.")
    chatbot.ask("Does Alex enjoy teaching?")
    chatbot.ask("What leadership has Alex demonstrated?")

    assert len(chatbot.history) == 2


# ---------------------------------------------------------------------------
# Query rewriter integration
# ---------------------------------------------------------------------------

class FakeRewriter:
    def __init__(self, rewritten: str):
        self._rewritten = rewritten
        self.called_with = None

    def rewrite(self, query, history=None):
        self.called_with = query
        return self._rewritten


def test_query_rewriter_is_called():
    documents = [
        Document(
            content="Charlie Yang interned at Pinecone in New York.",
            metadata={"source": "resume.md", "title": "Pinecone"},
        )
    ]
    rewriter = FakeRewriter("Charlie Yang Pinecone internship experience")
    chatbot = build_chatbot(documents, query_rewriter=rewriter)

    chatbot.ask("his work at the vector db company")

    assert rewriter.called_with == "his work at the vector db company"


def test_no_rewriter_passes_query_through():
    documents = [
        Document(
            content="Charlie built dashboards.",
            metadata={"source": "resume.md", "title": "Work"},
        )
    ]
    chatbot = build_chatbot(documents)
    response = chatbot.ask("dashboards", top_k=1)
    assert response.documents  # should find the document


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class FakeStreamingLLM(SimpleLLM):
    """SimpleLLM that also supports stream()."""

    def stream(self, question, context, *, system_prompt=None, chat_history=None):
        answer = self.generate(
            question, context, system_prompt=system_prompt, chat_history=chat_history
        )
        for word in answer.split():
            yield word + " "


def test_ask_stream_yields_tokens_then_documents():
    documents = [
        Document(
            content="Charlie has a Master of Science in Statistics from UC Davis.",
            metadata={"source": "resume.md", "title": "Education"},
        )
    ]
    retriever = ResumeRetriever(documents)
    llm = FakeStreamingLLM()
    chatbot = ResumeChatbot(retriever=retriever, llm=llm)

    events = list(chatbot.ask_stream("Master Statistics UC Davis", top_k=1))

    # Last event should be (None, documents)
    token_events = [e for e in events if e[0] is not None]
    doc_events = [e for e in events if e[1] is not None]

    assert len(token_events) >= 1
    assert len(doc_events) == 1
    assert len(doc_events[0][1]) >= 1


def test_ask_stream_fallback_no_stream_method():
    documents = [
        Document(
            content="Charlie used Python and SQL.",
            metadata={"source": "resume.md", "title": "Skills"},
        )
    ]
    chatbot = build_chatbot(documents)

    events = list(chatbot.ask_stream("skills", top_k=1))

    # SimpleLLM has no stream() — should still produce token + doc events
    token_events = [e for e in events if e[0] is not None]
    doc_events = [e for e in events if e[1] is not None]
    assert len(token_events) >= 1
    assert len(doc_events) == 1
