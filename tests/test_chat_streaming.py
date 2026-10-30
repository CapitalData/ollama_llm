import json

import pytest

from ollama_chat_app import (
    build_library_context,
    call_ollama,
    ensure_project,
    extract_pdf_summary,
    extract_relevant_pdf_pages,
    get_project_context,
    index_pdf_into_project,
    save_memory_entry,
    search_project_library,
)


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self.lines)


def test_call_ollama_collects_streamed_chat_chunks(monkeypatch):
    calls = []
    lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}).encode(),
        json.dumps({"message": {"content": " there"}, "done": True}).encode(),
    ]

    def fake_post(url, *, json, stream, timeout):
        calls.append((url, json, stream, timeout))
        return FakeResponse(lines)

    monkeypatch.setattr("ollama_chat_app.requests.post", fake_post)

    content, elapsed = call_ollama([{"role": "user", "content": "Hi"}], "test-model")

    assert content == "Hello there"
    assert elapsed >= 0
    assert calls[0][1]["stream"] is True
    assert calls[0][2] is True


def test_project_memory_storage_round_trip(tmp_path, monkeypatch):
    db_path = tmp_path / "project_memory.sqlite3"
    monkeypatch.setattr("ollama_chat_app.PROJECT_MEMORY_DB_PATH", db_path)
    monkeypatch.setattr("ollama_chat_app.PROJECT_MEMORY_INITIALIZED", False)

    project = ensure_project("alpha", "demo project")
    topic = project["topics"][0]

    saved = save_memory_entry(
        project_id=project["id"],
        topic_id=topic["id"],
        text="Remember to keep the app local-first and document decisions.",
        source="manual-note",
        kind="note",
    )

    context = get_project_context(project["id"], topic["id"])

    assert project["name"] == "alpha"
    assert project["description"] == "demo project"
    assert topic["name"] == "General"
    assert saved["text"].startswith("Remember")
    assert any("local-first" in entry["text"] for entry in context["entries"])
    assert context["topic"]["name"] == "General"


def test_extract_pdf_summary_and_search(tmp_path, monkeypatch):
    import fitz

    pdf_path = tmp_path / "demo_book.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Local AI Systems Overview\n\nAbstract: This guide explains local-first AI systems and memory design.")
    page.insert_text((72, 160), "Chapter 1: Local-first Design\nChapter 2: Memory Systems")
    doc.save(pdf_path)
    doc.close()

    summary = extract_pdf_summary(str(pdf_path))
    assert "local-first" in summary["abstract"].lower()
    assert "local-first" in summary["toc_text"].lower()
    assert "memory" in summary["index_text"].lower()

    db_path = tmp_path / "library.sqlite3"
    monkeypatch.setattr("ollama_chat_app.PROJECT_MEMORY_DB_PATH", db_path)
    monkeypatch.setattr("ollama_chat_app.PROJECT_MEMORY_INITIALIZED", False)

    project = ensure_project("library-test", "book library")
    topic = project["topics"][0]
    indexed = index_pdf_into_project(project["id"], topic["id"], str(pdf_path), "demo_book.pdf")
    matches = search_project_library(project["id"], "local-first ai")

    assert indexed["title"] == "Local AI Systems Overview"
    assert any(match["book_id"] == indexed["id"] for match in matches)

    page_context = extract_relevant_pdf_pages(str(pdf_path), "local-first ai")
    assert any("Page" in snippet for snippet in page_context)
    assert any("local-first" in snippet.lower() for snippet in page_context)

    context = build_library_context(project["id"], topic["id"], "local-first ai")
    assert any("Page" in item for item in context)
    assert any("local-first" in item.lower() for item in context)
