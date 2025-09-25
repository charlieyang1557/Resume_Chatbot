from pathlib import Path

from fastapi.testclient import TestClient

from resume_chatbot.webapp import create_app


def test_webapp_index_serves_html() -> None:
    app = create_app(resume_directory=Path("data/resume"))
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Resume Chatbot</title>" in response.text


def test_webapp_chat_endpoint_returns_sources() -> None:
    app = create_app(resume_directory=Path("data/resume"), top_k=2)
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "What programming languages are listed?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert isinstance(payload["sources"], list)
    assert payload["sources"], "Expected at least one source to be returned."
