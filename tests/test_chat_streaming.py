import json

from ollama_chat_app import call_ollama


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