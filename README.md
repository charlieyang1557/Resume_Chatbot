# Resume Chatbot

**Live demo: [resume-chatbot-production.up.railway.app](https://resume-chatbot-production.up.railway.app)**

A portfolio-ready RAG chatbot that answers questions about Charlie Yang's resume using hybrid retrieval and Claude.

## Architecture

```
User Query
    |
    v
Query Rewriter (Claude) -- resolves pronouns, adds specificity
    |
    v
+---+---+
| BM25  |  DenseRetriever (fastembed + FAISS)
+---+---+
    |           |
    v           v
Reciprocal Rank Fusion (alpha-weighted)
    |
    v
Top-K Chunks (with metadata)
    |
    v
Claude LLM (streaming SSE)
    |
    v
React Frontend (markdown rendering, collapsible sources)
```

**Retrieval**: BM25 handles exact keyword matches; FAISS with `BAAI/bge-small-en-v1.5` embeddings handles semantic similarity. Results are fused with Reciprocal Rank Fusion (RRF). The `alpha` parameter (0-1) controls the sparse/dense balance.

**LLM**: Claude `claude-sonnet-4-20250514` via the Anthropic SDK with streaming SSE. Falls back to a deterministic `SimpleLLM` for offline use.

**Frontend**: React + Vite with dark/light theme, markdown rendering, collapsible source cards, and typing indicators.

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

docker compose up
# Open http://localhost:8000
```

### Local Development

```bash
# Backend
pip install -r requirements.txt
uvicorn resume_chatbot.webapp:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### Offline Mode (no API key needed)

```bash
LLM_BACKEND=simple uvicorn resume_chatbot.webapp:app --reload
```

## Evaluation

Run the evaluation suite to compare retrieval strategies:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m resume_chatbot.eval
```

This sweeps `alpha=[0.0, 0.5, 1.0]`, runs 15 Q&A pairs at each setting, and outputs a comparison table showing which alpha yields the highest average relevance score.

## Project Structure

```
resume_chatbot/
  data_loader.py        # Markdown chunking by ## headers
  hybrid_retriever.py   # BM25 + FAISS + RRF
  llm.py                # BaseLLM, SimpleLLM, ClaudeLLM, OpenAIChatLLM
  query_rewriter.py     # Claude-powered query expansion
  chatbot.py            # Orchestration with streaming support
  webapp.py             # FastAPI with SSE streaming
  eval.py               # LLM-as-judge evaluation with alpha sweep
frontend/               # React + Vite + TypeScript
data/resume/            # Resume markdown files
tests/                  # pytest suite
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for Claude LLM and query rewriting |
| `LLM_BACKEND` | `simple` | `claude`, `simple`, `openai`, or `ollama` |
| `USE_HYBRID_RETRIEVER` | `true` | Set to `false` to use the lightweight TF-IDF retriever |
| `RETRIEVER_ALPHA` | `0.5` | 0.0 = pure BM25, 1.0 = pure dense |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Override Claude model |

## Tests

Run the fast default suite during development:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Before committing/deploying webapp changes, also run the production-path hybrid
retriever smoke test:

```bash
USE_HYBRID_RETRIEVER=true python -m pytest tests/test_webapp.py -q
```
