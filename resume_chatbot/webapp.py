"""FastAPI-powered web interface for the resume chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .chatbot import ResumeChatbot
from .corpus import load_resume_corpus
from .llm import create_llm
from .retriever import ResumeRetriever


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the chatbot.")


class Source(BaseModel):
    source: str
    title: str | None = None
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


def _build_chatbot(resume_directory: Path, llm_backend: str) -> ResumeChatbot:
    documents = load_resume_corpus(resume_directory)
    if not documents:
        raise RuntimeError(
            "No resume documents found. Add Markdown/text files to the resume directory."
        )
    retriever = ResumeRetriever(documents)
    llm = create_llm(llm_backend)
    return ResumeChatbot(retriever=retriever, llm=llm)


def _render_index_html() -> str:
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Resume Chatbot</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-border: #334155;
      --text: #e2e8f0;
      --accent: #38bdf8;
      --accent-contrast: #0f172a;
      --user-bg: #38bdf81f;
      --assistant-bg: #22c55e20;
      font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 32px 16px;
    }

    .app-shell {
      width: min(960px, 100%);
      background: rgba(15, 23, 42, 0.8);
      backdrop-filter: blur(18px);
      border: 1px solid var(--panel-border);
      border-radius: 20px;
      box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.6);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    header {
      padding: 24px 32px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    header h1 {
      margin: 0;
      font-size: clamp(1.8rem, 2vw + 1rem, 2.2rem);
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    header h1 span {
      background: linear-gradient(120deg, #38bdf8, #22d3ee);
      -webkit-background-clip: text;
      color: transparent;
    }

    header p {
      margin: 0;
      color: #94a3b8;
      max-width: 680px;
      line-height: 1.5;
    }

    .chat-window {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 24px 32px;
      overflow-y: auto;
      scroll-behavior: smooth;
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.8) 100%);
    }

    .bubble {
      padding: 16px 20px;
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      line-height: 1.6;
      backdrop-filter: blur(6px);
      box-shadow: 0 10px 30px -15px rgba(148, 163, 184, 0.4);
      animation: fadeIn 0.3s ease;
      max-width: 85%;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .bubble.user {
      align-self: flex-end;
      background: var(--user-bg);
      border-color: rgba(56, 189, 248, 0.35);
    }

    .bubble.assistant {
      align-self: flex-start;
      background: var(--assistant-bg);
      border-color: rgba(34, 197, 94, 0.35);
    }

    .sources {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 0.9rem;
      color: #cbd5f5;
    }

    .sources span {
      opacity: 0.8;
    }

    form {
      display: flex;
      gap: 12px;
      padding: 24px 32px;
      border-top: 1px solid var(--panel-border);
      background: rgba(15, 23, 42, 0.85);
    }

    textarea {
      flex: 1;
      resize: none;
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 16px;
      color: var(--text);
      font-size: 1rem;
      line-height: 1.5;
      min-height: 72px;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    textarea:focus {
      outline: none;
      border-color: rgba(56, 189, 248, 0.6);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
    }

    button {
      padding: 0 24px;
      border-radius: 16px;
      border: none;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      background: linear-gradient(120deg, #38bdf8, #22d3ee);
      color: var(--accent-contrast);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px -12px rgba(56, 189, 248, 0.6);
    }

    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      box-shadow: none;
    }

    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(6px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 600px) {
      header, .chat-window, form {
        padding: 20px;
      }

      .bubble {
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class=\"app-shell\">
    <header>
      <h1>Resume <span>Chatbot</span></h1>
      <p>Ask questions about the resume and receive grounded answers powered by retrieval-augmented generation.</p>
    </header>
    <main class=\"chat-window\" id=\"chat-window\">
      <div class=\"bubble assistant\">
        <strong>Assistant</strong>
        <span>Hi! I'm ready to answer questions about this resume. What would you like to know?</span>
      </div>
    </main>
    <form id=\"chat-form\">
      <textarea id=\"chat-input\" placeholder=\"Ask about experience, skills, education...\" required></textarea>
      <button type=\"submit\">Send</button>
    </form>
  </div>

  <script>
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const chatWindow = document.getElementById('chat-window');

    function appendBubble(role, content, sources = []) {
      const bubble = document.createElement('div');
      bubble.className = `bubble ${role}`;

      const name = document.createElement('strong');
      name.textContent = role === 'user' ? 'You' : 'Assistant';
      bubble.appendChild(name);

      const message = document.createElement('span');
      message.innerText = content;
      bubble.appendChild(message);

      if (sources.length > 0) {
        const list = document.createElement('div');
        list.className = 'sources';
        const heading = document.createElement('span');
        heading.textContent = 'Sources';
        list.appendChild(heading);

        sources.forEach((source) => {
          const item = document.createElement('span');
          const title = source.title ? ` :: ${source.title}` : '';
          item.textContent = `${source.source}${title} (score: ${source.score.toFixed(3)})`;
          list.appendChild(item);
        });
        bubble.appendChild(list);
      }

      chatWindow.appendChild(bubble);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) {
        return;
      }

      appendBubble('user', message);
      input.value = '';

      const button = form.querySelector('button');
      button.disabled = true;

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Request failed');
        }

        const data = await response.json();
        appendBubble('assistant', data.answer, data.sources || []);
      } catch (error) {
        appendBubble('assistant', error.message || 'Something went wrong.');
      } finally {
        button.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>"""


def create_app(
    *,
    resume_directory: Path = Path("data/resume"),
    llm_backend: str = "ollama",
    top_k: int = 3,
) -> FastAPI:
    """Create a FastAPI app backed by the resume chatbot."""

    chatbot = _build_chatbot(resume_directory, llm_backend)
    app = FastAPI(title="Resume Chatbot", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.chatbot = chatbot
    app.state.top_k = top_k

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(content=_render_index_html())

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        result = app.state.chatbot.ask(message, top_k=app.state.top_k)
        sources: List[Dict[str, Any]] = []
        for item in result.documents:
            metadata = item.document.metadata
            sources.append(
                {
                    "source": metadata.get("source", "resume"),
                    "title": metadata.get("title"),
                    "score": item.score,
                }
            )

        return ChatResponse(answer=result.answer, sources=[Source(**source) for source in sources])

    return app


app = create_app()


__all__ = ["create_app", "app"]
