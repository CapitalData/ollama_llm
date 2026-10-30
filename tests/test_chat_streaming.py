import json

import pytest

from ollama_chat_app import call_ollama, ensure_project, get_project_context, save_memory_entry


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