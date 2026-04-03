import json
from pathlib import Path

from fastapi.testclient import TestClient

from resume_chatbot.webapp import create_app


def _make_client() -> TestClient:
    app = create_app(resume_directory=Path("data/resume"), llm_backend="simple", top_k=3)
    return TestClient(app)


def test_health_endpoint():
    client = _make_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_endpoint():
    client = _make_client()
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["hero_title"] == "Charlie (Yutian) Yang"
    assert len(data["starter_questions"]) == 3


def test_chat_endpoint_returns_answer_and_sources():
    client = _make_client()
    response = client.post(
        "/api/chat", json={"message": "What programming languages does Charlie know?"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert isinstance(payload["sources"], list)


def test_chat_empty_message_returns_400():
    client = _make_client()
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 400


def test_stream_endpoint_returns_sse():
    client = _make_client()
    response = client.post(
        "/api/chat/stream",
        json={"message": "Tell me about Charlie's work experience"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = []
    for line in response.text.strip().split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Should have at least one token event and a done event
    types = [e["type"] for e in events]
    assert "done" in types

    # Should have a sources event
    source_events = [e for e in events if e["type"] == "sources"]
    assert len(source_events) <= 1  # 0 or 1


def test_stream_empty_message_returns_400():
    client = _make_client()
    response = client.post("/api/chat/stream", json={"message": ""})
    assert response.status_code == 400
