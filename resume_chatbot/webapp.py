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
      color-scheme: dark;
      --bg-0: #030712;
      --bg-1: #0f172a;
      --bg-2: #111827;
      --text-primary: #f8fafc;
      --text-muted: #94a3b8;
      --panel: rgba(15, 23, 42, 0.78);
      --panel-border: rgba(148, 163, 184, 0.18);
      --accent-primary: #38bdf8;
      --accent-secondary: #818cf8;
      --accent-success: #34d399;
      --user-bubble: rgba(56, 189, 248, 0.18);
      --assistant-bubble: rgba(129, 140, 248, 0.14);
      --radius-large: 28px;
      font-family: 'Inter', 'SF Pro Display', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background-color: var(--bg-0);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: clamp(1.5rem, 4vw, 3rem);
      background: radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.18), transparent 42%),
                  radial-gradient(circle at 80% 15%, rgba(129, 140, 248, 0.24), transparent 45%),
                  linear-gradient(135deg, var(--bg-0), var(--bg-1) 48%, var(--bg-2));
      color: var(--text-primary);
      position: relative;
      overflow: hidden;
    }

    body::before,
    body::after {
      content: '';
      position: absolute;
      inset: 0;
      background-image: linear-gradient(120deg, rgba(148, 163, 184, 0.08) 1px, transparent 0);
      background-size: 42px 42px;
      opacity: 0.18;
      mask-image: radial-gradient(circle at center, black 35%, transparent 70%);
      pointer-events: none;
    }

    body::after {
      background-size: 72px 72px;
      opacity: 0.08;
      filter: blur(0.5px);
    }

    .glow {
      position: absolute;
      width: 420px;
      height: 420px;
      border-radius: 50%;
      filter: blur(120px);
      opacity: 0.45;
      transform: translate(-50%, -50%);
      pointer-events: none;
      mix-blend-mode: screen;
    }

    .glow.glow-a {
      top: 18%;
      left: 15%;
      background: radial-gradient(circle, rgba(56, 189, 248, 0.55), transparent 65%);
    }

    .glow.glow-b {
      bottom: 10%;
      right: -10%;
      background: radial-gradient(circle, rgba(129, 140, 248, 0.55), transparent 70%);
    }

    .app-shell {
      position: relative;
      width: min(1080px, 100%);
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: clamp(1.5rem, 3vw, 2.25rem);
      padding: clamp(1.6rem, 3vw, 2.2rem);
      border-radius: var(--radius-large);
      border: 1px solid var(--panel-border);
      background: linear-gradient(155deg, rgba(15, 23, 42, 0.78), rgba(17, 24, 39, 0.94));
      backdrop-filter: blur(22px);
      box-shadow: 0 30px 60px -22px rgba(15, 23, 42, 0.72);
      z-index: 1;
    }

    @media (max-width: 1024px) {
      .app-shell {
        grid-template-columns: 1fr;
      }
    }

    .sidebar {
      display: flex;
      flex-direction: column;
      gap: clamp(1.5rem, 3vw, 2rem);
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      padding: clamp(1.2rem, 2vw, 1.6rem);
      border-radius: 22px;
      border: 1px solid rgba(148, 163, 184, 0.14);
      background: linear-gradient(160deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.92));
      box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.06);
    }

    .brand-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }

    .logo {
      width: 48px;
      height: 48px;
      border-radius: 16px;
      display: grid;
      place-items: center;
      background: linear-gradient(145deg, rgba(56, 189, 248, 0.32), rgba(129, 140, 248, 0.38));
      border: 1px solid rgba(56, 189, 248, 0.32);
      box-shadow: 0 12px 24px -12px rgba(56, 189, 248, 0.75);
      font-weight: 600;
      letter-spacing: 0.08em;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.45rem 0.9rem;
      border-radius: 999px;
      background: rgba(52, 211, 153, 0.12);
      border: 1px solid rgba(52, 211, 153, 0.32);
      color: var(--text-primary);
      font-size: 0.85rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--accent-success);
      box-shadow: 0 0 12px rgba(52, 211, 153, 0.65);
    }

    .brand h1 {
      margin: 0;
      font-size: clamp(1.9rem, 2.6vw, 2.4rem);
      font-weight: 600;
      letter-spacing: -0.02em;
    }

    .brand h1 span {
      background: linear-gradient(120deg, var(--accent-primary), var(--accent-secondary));
      -webkit-background-clip: text;
      color: transparent;
    }

    .brand p {
      margin: 0;
      color: var(--text-muted);
      line-height: 1.6;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
    }

    .chip {
      border: none;
      border-radius: 999px;
      padding: 0.55rem 1.1rem;
      font-size: 0.9rem;
      background: rgba(148, 163, 184, 0.12);
      color: var(--text-primary);
      cursor: pointer;
      transition: transform 0.2s ease, background 0.2s ease;
    }

    .chip:hover {
      transform: translateY(-1px);
      background: rgba(129, 140, 248, 0.28);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.85rem;
    }

    .metric-card {
      padding: 1rem 1.2rem;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      background: rgba(15, 23, 42, 0.72);
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.4);
    }

    .metric-label {
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.78);
    }

    .metric-value {
      font-size: 1.35rem;
      font-weight: 600;
    }

    .metric-subtext {
      font-size: 0.82rem;
      color: rgba(148, 163, 184, 0.68);
    }

    .conversation {
      border-radius: 24px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(15, 23, 42, 0.72);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      min-height: min(70vh, 720px);
      position: relative;
    }

    .conversation::before {
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.14), transparent 55%);
      opacity: 0.65;
      pointer-events: none;
    }

    .chat-window {
      flex: 1;
      padding: clamp(1.5rem, 3vw, 2rem);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      position: relative;
      z-index: 1;
      scroll-behavior: smooth;
    }

    .bubble {
      padding: 1.1rem 1.25rem;
      border-radius: 20px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      line-height: 1.65;
      backdrop-filter: blur(9px);
      box-shadow: 0 18px 40px -24px rgba(15, 23, 42, 0.85);
      animation: fadeIn 0.3s ease;
      max-width: 82%;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .bubble strong {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: rgba(148, 163, 184, 0.86);
    }

    .bubble span {
      white-space: pre-wrap;
    }

    .bubble.user {
      align-self: flex-end;
      background: var(--user-bubble);
      border-color: rgba(56, 189, 248, 0.4);
      color: var(--text-primary);
    }

    .bubble.assistant {
      align-self: flex-start;
      background: var(--assistant-bubble);
      border-color: rgba(129, 140, 248, 0.4);
    }

    .sources {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
    }

    .source-badge {
      border-radius: 999px;
      padding: 0.35rem 0.75rem;
      font-size: 0.78rem;
      background: rgba(148, 163, 184, 0.16);
      color: rgba(226, 232, 240, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.22);
    }

    form {
      padding: clamp(1.2rem, 3vw, 1.8rem);
      border-top: 1px solid rgba(148, 163, 184, 0.18);
      background: linear-gradient(160deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.88));
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 1rem;
      position: relative;
      z-index: 1;
    }

    textarea {
      resize: none;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.24);
      border-radius: 18px;
      padding: 1rem 1.1rem;
      color: var(--text-primary);
      font-size: 1rem;
      line-height: 1.6;
      min-height: 92px;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
      font-family: inherit;
    }

    textarea:focus {
      outline: none;
      border-color: rgba(56, 189, 248, 0.6);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.22);
    }

    button[type="submit"] {
      border: none;
      border-radius: 18px;
      padding: 0 2.1rem;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
      color: #020617;
      transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
    }

    button[type="submit"]:hover {
      transform: translateY(-1px);
      box-shadow: 0 16px 26px -18px rgba(56, 189, 248, 0.8);
    }

    button[type="submit"]:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      box-shadow: none;
    }

    .footer-note {
      margin-top: auto;
      font-size: 0.78rem;
      color: rgba(148, 163, 184, 0.64);
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }

    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(4px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @media (max-width: 720px) {
      body {
        padding: 1.25rem;
      }

      .app-shell {
        padding: 1.4rem;
      }

      form {
        grid-template-columns: 1fr;
      }

      .bubble {
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
  <span class=\"glow glow-a\"></span>
  <span class=\"glow glow-b\"></span>
  <div class=\"app-shell\">
    <aside class=\"sidebar\">
      <section class=\"brand\">
        <div class=\"brand-top\">
          <div class=\"logo\">CV</div>
          <div class=\"status-pill\">
            <span class=\"status-dot\"></span>
            Online
          </div>
        </div>
        <div>
          <h1>Resume <span>Chatbot</span></h1>
          <p>Ask focused questions about Charlie (a.k.a. Yutian Yang) and get evidence-backed answers tailored for hiring teams.</p>
        </div>
        <div class=\"chips\">
          <button type=\"button\" class=\"chip\" data-question=\"What are Charlie's most impactful projects?\">Impactful projects</button>
          <button type=\"button\" class=\"chip\" data-question=\"Summarize Charlie's experience at Pinecone.\">Pinecone work</button>
          <button type=\"button\" class=\"chip\" data-question=\"Which technical skills does Charlie highlight?\">Technical skills</button>
          <button type=\"button\" class=\"chip\" data-question=\"Outline Charlie's academic background.\">Education</button>
        </div>
      </section>
      <section class=\"metrics\">
        <article class=\"metric-card\">
          <span class=\"metric-label\">Context</span>
          <span class=\"metric-value\">16 docs</span>
          <span class=\"metric-subtext\">Curated resume knowledge base</span>
        </article>
        <article class=\"metric-card\">
          <span class=\"metric-label\">Focus</span>
          <span class=\"metric-value\">RAG</span>
          <span class=\"metric-subtext\">Retrieval-augmented insights</span>
        </article>
        <article class=\"metric-card\">
          <span class=\"metric-label\">Latency</span>
          <span class=\"metric-value\">&lt;1s</span>
          <span class=\"metric-subtext\">Deterministic responses</span>
        </article>
      </section>
      <p class=\"footer-note\">Crafted with modern product design cues from OpenAI, Tesla, Notion, and Uber.</p>
    </aside>
    <section class=\"conversation\">
      <main class=\"chat-window\" id=\"chat-window\">
        <div class=\"bubble assistant\">
          <strong>Assistant</strong>
          <span>Hey there! I'm online and ready to answer anything about Charlie's resume. Pick a prompt or ask your own question.</span>
        </div>
      </main>
      <form id=\"chat-form\">
        <textarea id=\"chat-input\" placeholder=\"Ask about experience, impact, tools, education...\" required></textarea>
        <button type=\"submit\">
          <span class=\"button-label\">Send</span>
          <svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">
            <path d=\"M3 12L21 3L14 12L21 21L3 12Z\" fill=\"currentColor\"/>
          </svg>
        </button>
      </form>
    </section>
  </div>

  <script>
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const chatWindow = document.getElementById('chat-window');
    const chips = document.querySelectorAll('.chip');
    const submitButton = form.querySelector('button[type="submit"]');
    const buttonLabel = submitButton.querySelector('.button-label');

    function formatSource(source) {
      const title = source.title ? ` :: ${source.title}` : '';
      const score = Number.isFinite(source.score) ? ` · ${source.score.toFixed(3)}` : '';
      return `${source.source}${title}${score}`;
    }

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

        sources.forEach((source) => {
          const item = document.createElement('span');
          item.className = 'source-badge';
          item.textContent = formatSource(source);
          list.appendChild(item);
        });

        bubble.appendChild(list);
      }

      chatWindow.appendChild(bubble);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    chips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const question = chip.getAttribute('data-question');
        input.value = question;
        input.focus({ preventScroll: true });
      });
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) {
        return;
      }

      appendBubble('user', message);
      input.value = '';

      submitButton.disabled = true;
      buttonLabel.textContent = 'Sending';

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });

        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          throw new Error(error.detail || 'Request failed');
        }

        const data = await response.json();
        appendBubble('assistant', data.answer, data.sources || []);
      } catch (error) {
        appendBubble('assistant', error.message || 'Something went wrong.');
      } finally {
        submitButton.disabled = false;
        buttonLabel.textContent = 'Send';
        input.focus({ preventScroll: true });
      }
    });
  </script>
</body>
</html>"""


def create_app(
    *,
    resume_directory: Path = Path("data/resume"),
    llm_backend: str = "simple",
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
