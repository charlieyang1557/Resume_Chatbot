"""Command line interface for the resume chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .chatbot import DEFAULT_SYSTEM_PROMPT, ResumeChatbot
from .corpus import load_resume_corpus
from .llm import create_llm
from .retriever import ResumeRetriever

app = typer.Typer(add_completion=False)
console = Console()


def _load_system_prompt(path: Optional[Path]) -> str:
    if path is None:
        return DEFAULT_SYSTEM_PROMPT
    text = path.read_text(encoding="utf-8").strip()
    return text or DEFAULT_SYSTEM_PROMPT


@app.command()
def chat(
    resume_directory: Path = typer.Option(
        Path("data/resume"),
        exists=True,
        readable=True,
        dir_okay=True,
        file_okay=False,
        help="Directory containing Markdown or text resume files.",
    ),
    llm: str = typer.Option(
        "simple",
        help="LLM backend to use ('simple' for offline mode or 'openai' for API-backed responses).",
    ),
    top_k: int = typer.Option(3, min=1, max=10, help="Number of resume sections to retrieve."),
    system_prompt_path: Optional[Path] = typer.Option(
        None,
        help="Optional path to a custom system prompt file.",
    ),
    history_limit: int = typer.Option(5, min=0, help="How many previous turns to retain in the prompt."),
    show_sources: bool = typer.Option(
        True,
        help="Display the resume sections that were retrieved for each answer.",
    ),
) -> None:
    """Start an interactive chat session."""

    documents = load_resume_corpus(resume_directory)
    if not documents:
        console.print(
            f"[red]No resume documents found in {resume_directory}. Add Markdown or text files and try again.[/red]"
        )
        raise typer.Exit(code=1)

    retriever = ResumeRetriever(documents)
    try:
        llm_client = create_llm(llm)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    system_prompt = _load_system_prompt(system_prompt_path)

    chatbot = ResumeChatbot(
        retriever=retriever,
        llm=llm_client,
        system_prompt=system_prompt,
        history_limit=history_limit,
    )

    console.print(Panel("Resume Chatbot\nType 'exit' or 'quit' to leave.", title="RAG"))

    while True:
        try:
            console.print("[bold cyan]You:[/bold cyan] ", end="")
            question = input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session ended by user.[/bold yellow]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            console.print("[green]Goodbye![/green]")
            break

        result = chatbot.ask(question, top_k=top_k)
        console.print(Panel(result.answer, title="Assistant", style="green"))
        if show_sources and result.documents:
            console.print("[bold magenta]Sources:[/bold magenta]")
            for item in result.documents:
                metadata = item.document.metadata
                source = metadata.get("source", "resume")
                title = metadata.get("title")
                score = f"{item.score:.3f}"
                if title:
                    console.print(f"  - {source} :: {title} (score={score})")
                else:
                    console.print(f"  - {source} (score={score})")
            console.print()


if __name__ == "__main__":
    app()
