# Resume Chatbot

An end-to-end Retrieval Augmented Generation (RAG) chatbot that can answer questions about your resume. The project ships with a lightweight in-memory retriever, a simple fallback "LLM" for offline experimentation, and optional OpenAI integration for production-grade responses.

## Features

- **Markdown-first ingestion** – drop one or more `.md`/`.txt` files into `data/resume/` and the loader will automatically split them into semantic sections.
- **Pure Python retriever** – a custom TF-IDF retriever implemented with the standard library (no heavyweight ML dependencies required).
- **Pluggable LLM backends** – start with the deterministic `SimpleLLM` for local testing or switch to OpenAI with a single flag.
- **Command line chat** – interact with your resume through an interactive CLI powered by Typer and Rich.
- **Knowledge graph enrichment** – optionally describe entities and relationships in `knowledge_graph.json` to add structured
  context to retrieval.
- **Test coverage** – unit tests that exercise the retriever and the full RAG pipeline.

## Project layout

```
resume_chatbot/
├── data_loader.py       # Parses resume files into structured documents
├── retriever.py         # Lightweight TF-IDF retriever and similarity scoring
├── llm.py               # Abstractions for language models + Simple/OpenAI backends
├── chatbot.py           # High-level orchestration of retrieval + generation
├── cli.py               # Typer-powered CLI entrypoint
└── __init__.py
```

Supporting files:

- `data/resume/resume_example.md` – sample resume content to get you started.
- `data/resume/knowledge_graph.json` – structured relationships that are automatically converted into retrievable text.
- `pyproject.toml` – dependency metadata (install with `pip install -e .`).
- `tests/` – unit tests for the retriever and chatbot.

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. **Add your resume data**

- Replace the sample files in `data/resume/` with your own Markdown/Plain-text resume. Headings (e.g. `## Experience`) are used to create focused chunks for retrieval.
- (Optional) Update `knowledge_graph.json` with important entities, relationships, dates, and technologies. The CLI will load the
  graph and expose the information as additional context for retrieval.

3. **(Optional) Configure OpenAI**

   Create a `.env` file or export the variables directly in your shell:

   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_MODEL="gpt-4o-mini"  # defaults to gpt-3.5-turbo if unset
   ```

4. **Chat with your resume (CLI)**

   ```bash
   python -m resume_chatbot.cli chat \
       --resume-directory data/resume \
       --llm simple            # or "openai" if you configured an API key
   ```

   Type your questions and press <kbd>Enter</kbd>. Use `exit` or `quit` to leave the session. When the OpenAI backend is enabled, the assistant automatically adds the retrieved resume snippets to the prompt.

5. **Launch the web app**

   ```bash
   uvicorn resume_chatbot.webapp:app --reload
   ```

   Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) to use the responsive chat UI. The interface displays source snippets
   alongside each answer so you can verify grounding.

## Running tests

Run the automated test suite any time you change the code or resume content:

```bash
pytest
```

## How it works

1. **Document loading** – `data_loader.load_resume_documents` reads Markdown/text files and breaks them into sections with metadata (`title`, `source`).
2. **Vectorisation & retrieval** – `retriever.ResumeRetriever` tokenises text, computes TF-IDF style weights, and scores similarity using cosine distance.
3. **Response generation** – `chatbot.ResumeChatbot` fetches the most relevant sections, builds a context string, and forwards it to the configured LLM backend.
4. **CLI** – `python -m resume_chatbot.cli chat` wires everything together for an interactive experience.

The modular design lets you swap the retriever or LLM while keeping the rest of the pipeline untouched.

## Next steps

- Replace the sample resume with your own data and regenerate the vector store at runtime.
- Deploy the CLI with a web frontend (e.g. FastAPI + React) or integrate it into Slack/Teams bots.
- Experiment with other LLM providers by creating a new backend that implements `BaseLLM`.

Happy hacking!
