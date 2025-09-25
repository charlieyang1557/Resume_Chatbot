from pathlib import Path

from resume_chatbot.data_loader import Document, _split_markdown_sections, load_resume_documents


def test_split_markdown_sections_creates_chunks(tmp_path: Path):
    content = """# Title\nIntro paragraph.\n\n## Experience\nDid things.\n\n## Skills\nPython\n"""
    sections = list(_split_markdown_sections(content))

    assert sections[0][0] == "Title"
    assert "Intro" in sections[0][1]
    assert sections[1][0] == "Experience"
    assert "Did things" in sections[1][1]


def test_load_resume_documents(tmp_path: Path):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    file_path = resume_dir / "resume.md"
    file_path.write_text("""# Summary\nLine one.\n\n## Skills\nPython\n""", encoding="utf-8")

    documents = load_resume_documents(resume_dir)

    assert len(documents) == 2
    assert documents[0].metadata["source"] == "resume.md"
    assert documents[0].metadata["chunk"] == 1
    assert documents[1].metadata["title"] == "Skills"
