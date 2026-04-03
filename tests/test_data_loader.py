from pathlib import Path

from resume_chatbot.data_loader import Document, _split_markdown_sections, load_resume_documents


def test_split_by_double_hash_headers():
    content = "# Title\nIntro paragraph.\n\n## Experience\nDid things.\n\n## Skills\nPython\n"
    sections = list(_split_markdown_sections(content))

    # Should produce 3 sections: intro (under #), Experience, Skills
    assert len(sections) == 3
    # Each tuple is (parent_title, section_title, body)
    assert sections[0] == ("Title", "Overview", "Intro paragraph.")
    assert sections[1] == ("Title", "Experience", "Did things.")
    assert sections[2] == ("Title", "Skills", "Python")


def test_bullets_stay_with_parent_header():
    content = "# Resume\n\n## Work\n- Built dashboards\n- Reduced churn by 10%\n\n## Skills\n- Python\n"
    sections = list(_split_markdown_sections(content))

    assert len(sections) == 2
    assert "Built dashboards" in sections[0][2]
    assert "Reduced churn" in sections[0][2]
    assert sections[1][2] == "- Python"


def test_parent_title_carried_through():
    content = "# Charlie Yang\n\n## Education\nUC Davis\n\n## Projects\nMushroom classifier\n"
    sections = list(_split_markdown_sections(content))

    for parent, _, _ in sections:
        assert parent == "Charlie Yang"


def test_no_headers_single_chunk():
    content = "Just plain text with no headers."
    sections = list(_split_markdown_sections(content))

    assert len(sections) == 1
    assert sections[0][2] == content


def test_load_resume_documents(tmp_path: Path):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    file_path = resume_dir / "resume.md"
    file_path.write_text(
        "# Summary\nLine one.\n\n## Skills\nPython\n", encoding="utf-8"
    )

    documents = load_resume_documents(resume_dir)

    assert len(documents) == 2
    assert documents[0].metadata["source"] == "resume.md"
    assert documents[0].metadata["chunk_index"] == 1
    assert documents[0].metadata["parent_title"] == "Summary"
    assert documents[1].metadata["title"] == "Skills"


def test_load_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_resume_documents(empty) == []


def test_load_nonexistent_directory(tmp_path: Path):
    assert load_resume_documents(tmp_path / "nope") == []
