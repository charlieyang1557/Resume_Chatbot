"""Evaluation module for the resume chatbot.

Generates 15 Q&A pairs from the resume, scores chatbot answers for
faithfulness and relevance using LLM-as-judge, and sweeps over
``alpha=[0.0, 0.5, 1.0]`` to compare retrieval strategies.

Usage::

    python -m resume_chatbot.eval
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table

from .chatbot import ResumeChatbot
from .data_loader import load_resume_documents
from .hybrid_retriever import BM25Retriever, DenseRetriever, HybridRetriever
from .llm import ClaudeLLM, SimpleLLM, create_llm

# ---------------------------------------------------------------------------
# Evaluation Q&A pairs
# ---------------------------------------------------------------------------

EVAL_PAIRS: list[dict[str, str]] = [
    # Experience — Pinecone
    {
        "question": "What dashboards did Charlie build at Pinecone?",
        "expected": "Book of Business, Account 360, and Pinecone Assistant dashboards using SQL and Sigma.",
        "category": "experience",
    },
    {
        "question": "How did Charlie reduce onboarding time at Pinecone?",
        "expected": "Documented processes in Notion, reducing onboarding time by 30%.",
        "category": "experience",
    },
    {
        "question": "What was the result of Charlie's churn analysis at Pinecone?",
        "expected": "Identified 5 key metrics using Random Forests, reducing churn by 10%.",
        "category": "experience",
    },
    # Experience — PTC
    {
        "question": "What anomaly detection models did Charlie evaluate at PTC?",
        "expected": "Prophet + Isolation Forest, Merlion, and LSTM-AE for API telemetry monitoring.",
        "category": "experience",
    },
    {
        "question": "How did Charlie use Claude at PTC?",
        "expected": "Applied AWS Bedrock's Claude 3.5 Sonnet for clustering, naming, and sentiment analysis of NPS feedback.",
        "category": "experience",
    },
    # Experience — Allschool
    {
        "question": "What impact did Charlie have at Allschool?",
        "expected": "15% reduction in project budget, increased daily active users, and accelerated class selection by 50% with a web scraper.",
        "category": "experience",
    },
    # Experience — UC Davis Research
    {
        "question": "What statistical methods did Charlie use in his research at UC Davis?",
        "expected": "GLM, Logistic Regression, Bootstrapping, and Lasso Regression for procrastination analysis.",
        "category": "experience",
    },
    # Skills
    {
        "question": "What programming languages does Charlie know?",
        "expected": "SQL, Python, and R.",
        "category": "skills",
    },
    {
        "question": "What machine learning tools does Charlie use?",
        "expected": "Scikit-learn, TensorFlow, Random Forests, SVM, CNN, Time Series.",
        "category": "skills",
    },
    {
        "question": "What data engineering tools has Charlie worked with?",
        "expected": "BigQuery, DBT, AWS, GCP, ETL, Schema Design.",
        "category": "skills",
    },
    # Education
    {
        "question": "What degrees does Charlie have?",
        "expected": "M.S. in Statistics and B.S. in Statistics and Economics, both from UC Davis.",
        "category": "education",
    },
    {
        "question": "What coursework is relevant to Charlie's ML work?",
        "expected": "Statistical Machine Learning, Algorithm Design, Optimization of Big Data Analytics, Time Series Analysis.",
        "category": "education",
    },
    # Projects
    {
        "question": "What deep learning models did Charlie use for the mushroom classification project?",
        "expected": "Random Forest, Kernel SVM, CNN, and transfer learning with pre-trained ResNet50.",
        "category": "projects",
    },
    # Cross-cutting
    {
        "question": "What BI tools has Charlie used across his roles?",
        "expected": "Tableau, Power BI, Looker, Sigma, Google Analytics, Google Looker Studio.",
        "category": "skills",
    },
    {
        "question": "Where did Charlie do his internships?",
        "expected": "Pinecone (New York), PTC/Onshape (Boston), Allschool (San Mateo).",
        "category": "experience",
    },
]


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an evaluation judge. Score the assistant's answer against the "
    "expected answer on two dimensions:\n"
    "1. **Faithfulness** (1-5): Is the answer factually consistent with the expected answer? "
    "5 = perfectly faithful, 1 = contradicts or fabricates.\n"
    "2. **Relevance** (1-5): Does the answer address the question? "
    "5 = fully answers the question, 1 = completely off-topic.\n\n"
    "Respond with ONLY two numbers separated by a comma, e.g.: 4,5"
)


@dataclass(frozen=True)
class JudgeScore:
    faithfulness: int
    relevance: int


@dataclass(frozen=True)
class EvalResult:
    question: str
    expected: str
    actual: str
    category: str
    faithfulness: int
    relevance: int


def _judge_answer(
    client: ClaudeLLM,
    question: str,
    expected: str,
    actual: str,
) -> JudgeScore:
    prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Assistant's answer: {actual}\n\n"
        "Score (faithfulness,relevance):"
    )
    try:
        response = client.generate(prompt, "", system_prompt=_JUDGE_SYSTEM)
        parts = response.strip().split(",")
        return JudgeScore(
            faithfulness=max(1, min(5, int(parts[0].strip()))),
            relevance=max(1, min(5, int(parts[1].strip()))),
        )
    except Exception:
        return JudgeScore(faithfulness=0, relevance=0)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    resume_dir: Path,
    alpha: float,
    judge: ClaudeLLM,
    llm_backend: str = "simple",
) -> List[EvalResult]:
    """Run all 15 Q&A pairs at a given alpha and return scored results."""
    documents = load_resume_documents(resume_dir)
    if not documents:
        raise RuntimeError(f"No documents found in {resume_dir}")

    sparse = BM25Retriever(documents)
    dense = DenseRetriever(documents)
    retriever = HybridRetriever(sparse, dense, alpha=alpha)
    llm = create_llm(llm_backend)
    chatbot = ResumeChatbot(retriever=retriever, llm=llm)

    results: list[EvalResult] = []
    for pair in EVAL_PAIRS:
        result = chatbot.ask(pair["question"], top_k=5)
        chatbot.reset_history()

        score = _judge_answer(
            judge, pair["question"], pair["expected"], result.answer
        )
        results.append(
            EvalResult(
                question=pair["question"],
                expected=pair["expected"],
                actual=result.answer,
                category=pair["category"],
                faithfulness=score.faithfulness,
                relevance=score.relevance,
            )
        )
    return results


def print_results(all_results: dict[float, List[EvalResult]]) -> None:
    """Print a rich table comparing alpha settings."""
    console = Console()

    # Per-alpha detail tables
    for alpha, results in all_results.items():
        table = Table(title=f"Alpha = {alpha}")
        table.add_column("Category", style="cyan", width=12)
        table.add_column("Question", width=45)
        table.add_column("Faith", justify="center", width=6)
        table.add_column("Relev", justify="center", width=6)

        for r in results:
            faith_style = "green" if r.faithfulness >= 4 else "yellow" if r.faithfulness >= 3 else "red"
            relev_style = "green" if r.relevance >= 4 else "yellow" if r.relevance >= 3 else "red"
            table.add_row(
                r.category,
                r.question[:45],
                f"[{faith_style}]{r.faithfulness}[/]",
                f"[{relev_style}]{r.relevance}[/]",
            )

        avg_f = sum(r.faithfulness for r in results) / len(results)
        avg_r = sum(r.relevance for r in results) / len(results)
        table.add_row("", "[bold]Average[/bold]", f"[bold]{avg_f:.1f}[/bold]", f"[bold]{avg_r:.1f}[/bold]")
        console.print(table)
        console.print()

    # Comparison summary
    summary = Table(title="Alpha Comparison Summary")
    summary.add_column("Alpha", justify="center")
    summary.add_column("Avg Faithfulness", justify="center")
    summary.add_column("Avg Relevance", justify="center")
    summary.add_column("", justify="center")

    best_alpha = -1.0
    best_relevance = -1.0

    for alpha, results in all_results.items():
        avg_r = sum(r.relevance for r in results) / len(results)
        if avg_r > best_relevance:
            best_relevance = avg_r
            best_alpha = alpha

    for alpha, results in all_results.items():
        avg_f = sum(r.faithfulness for r in results) / len(results)
        avg_r = sum(r.relevance for r in results) / len(results)
        marker = "[bold green]<-- BEST[/bold green]" if alpha == best_alpha else ""
        summary.add_row(str(alpha), f"{avg_f:.2f}", f"{avg_r:.2f}", marker)

    console.print(summary)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    resume_dir = Path("data/resume")
    if not resume_dir.exists():
        print(f"Resume directory not found: {resume_dir}")
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Required for LLM-as-judge scoring.")
        sys.exit(1)

    judge = ClaudeLLM()
    alphas = [0.0, 0.5, 1.0]
    all_results: dict[float, List[EvalResult]] = {}

    console = Console()
    for alpha in alphas:
        console.print(f"\n[bold]Running evaluation with alpha={alpha}...[/bold]")
        all_results[alpha] = run_eval(resume_dir, alpha, judge, llm_backend="claude")

    console.print("\n")
    print_results(all_results)


if __name__ == "__main__":
    main()
