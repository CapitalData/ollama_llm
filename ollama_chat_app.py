"""Dash chat frontend for Ollama with OpenTelemetry export to Arize Phoenix."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", f"http://127.0.0.1:{OLLAMA_PORT}")
OLLAMA_CHAT_URL = f"{OLLAMA_ENDPOINT}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_ENDPOINT}/api/tags"

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

PHOENIX_OTLP_ENDPOINT = os.getenv(
    "PHOENIX_OTLP_ENDPOINT",
    os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:6006/v1/traces"),
)
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "default")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "ollama-dash-chat")

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.getenv("OLLAMA_STORAGE_DIR", str(BASE_DIR / "storage")))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CHAT_LOG_PATH = STORAGE_DIR / "ollama_chat_history.jsonl"

MAX_MESSAGES = int(os.getenv("OLLAMA_MAX_MESSAGES", "12"))
MAX_LOG_ENTRIES = int(os.getenv("OLLAMA_MAX_LOG_ENTRIES", "500"))


def setup_tracing() -> None:
    resource = Resource.create(
        {
            "service.name": OTEL_SERVICE_NAME,
            "openinference.project.name": PHOENIX_PROJECT_NAME,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=PHOENIX_OTLP_ENDPOINT, timeout=10)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


def tracer() -> trace.Tracer:
    return trace.get_tracer("ollama-chat")


def read_history() -> list[dict[str, Any]]:
    if not CHAT_LOG_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    with CHAT_LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-MAX_LOG_ENTRIES:]


def append_history(entry: dict[str, Any]) -> None:
    with CHAT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def fetch_models() -> list[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        response.raise_for_status()
        data = response.json()
        models = [item.get("name") for item in data.get("models", []) if item.get("name")]
        return models or [DEFAULT_MODEL]
    except requests.RequestException:
        return [DEFAULT_MODEL]


def ollama_health() -> bool:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def call_ollama(messages: list[dict[str, str]], model: str) -> tuple[str, float]:
    start = time.time()
    payload = {"model": model, "messages": messages, "stream": False}
    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data.get("message", {}).get("content", "<no response>")
    elapsed = (time.time() - start) * 1000
    return content, elapsed


setup_tracing()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Ollama Chat"

app.layout = dbc.Container(
    [
        dcc.Store(id="session-id", data=str(uuid.uuid4())),
        dcc.Store(id="conversation-store", data=[]),
        dcc.Store(id="model-store", data=fetch_models()),
        dcc.Interval(id="health-ticker", interval=3000, n_intervals=0),
        html.H2("Ollama Chat Console", className="mt-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Status"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="ollama-status"),
                                        html.Div(id="phoenix-status", className="mt-2"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Model"),
                                dbc.CardBody(
                                    [
                                        dcc.Dropdown(
                                            id="model-select",
                                            options=[
                                                {"label": model, "value": model}
                                                for model in fetch_models()
                                            ],
                                            value=DEFAULT_MODEL,
                                            clearable=False,
                                        ),
                                        dbc.Button(
                                            "Refresh Models",
                                            id="refresh-models",
                                            className="mt-2",
                                            color="secondary",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Recent Sessions"),
                                dbc.CardBody(html.Div(id="recent-history")),
                            ]
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Conversation"),
                                dbc.CardBody(
                                    html.Div(id="chat-window", style={"whiteSpace": "pre-wrap"})
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Textarea(
                            id="prompt-input",
                            placeholder="Ask Ollama anything...",
                            style={"minHeight": "140px"},
                        ),
                        dbc.Button("Send", id="send-button", color="success", className="mt-2"),
                        html.Div(id="send-error", className="text-danger mt-2"),
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


@callback(
    Output("model-select", "options"),
    Output("model-select", "value"),
    Input("refresh-models", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_models(_n_clicks):
    models = fetch_models()
    options = [{"label": model, "value": model} for model in models]
    return options, models[0]


@callback(
    Output("ollama-status", "children"),
    Output("phoenix-status", "children"),
    Input("health-ticker", "n_intervals"),
)
def update_status(_ticks):
    ollama_ok = ollama_health()
    ollama_text = "Ollama: online" if ollama_ok else "Ollama: offline"
    phoenix_text = f"Phoenix OTLP: {PHOENIX_OTLP_ENDPOINT} | Project: {PHOENIX_PROJECT_NAME}"
    return ollama_text, phoenix_text


@callback(
    Output("conversation-store", "data"),
    Output("chat-window", "children"),
    Output("recent-history", "children"),
    Output("prompt-input", "value"),
    Output("send-error", "children"),
    Input("send-button", "n_clicks"),
    State("prompt-input", "value"),
    State("conversation-store", "data"),
    State("model-select", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def send_prompt(_n_clicks, prompt, conversation, model, session_id):
    if not prompt:
        return conversation, _render_conversation(conversation), _render_history(), "", "Prompt is empty"

    conversation = conversation or []
    conversation.append({"role": "user", "content": prompt})

    error_message = ""
    response_text = ""
    latency_ms = 0.0

    with tracer().start_as_current_span("ollama.chat") as span:
        span.set_attribute("openinference.project.name", PHOENIX_PROJECT_NAME)
        span.set_attribute("ollama.model", model)
        span.set_attribute("ollama.prompt.length", len(prompt))
        span.set_attribute("ollama.session_id", session_id)
        try:
            response_text, latency_ms = call_ollama(conversation, model)
            span.set_attribute("ollama.response.length", len(response_text))
            span.set_attribute("ollama.latency_ms", latency_ms)
            conversation.append({"role": "assistant", "content": response_text})
        except requests.RequestException as exc:
            error_message = f"Ollama request failed: {exc}"
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, error_message))

    if response_text:
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "model": model,
            "prompt": prompt,
            "response": response_text,
            "latency_ms": latency_ms,
        }
        append_history(entry)

    conversation = conversation[-MAX_MESSAGES:]
    chat_view = _render_conversation(conversation)
    history_view = _render_history()
    return conversation, chat_view, history_view, "", error_message


def _render_conversation(conversation):
    conversation = conversation or []
    if not conversation:
        return "No messages yet."
    blocks = []
    for message in conversation:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        blocks.append(f"{role}: {content}")
    return "\n\n".join(blocks)


def _render_history():
    entries = read_history()
    if not entries:
        return "No stored sessions yet."
    items = []
    for entry in entries[-10:][::-1]:
        when = entry.get("timestamp", "")
        model = entry.get("model", "")
        preview = entry.get("prompt", "")[:80]
        items.append(html.Div(f"{when} | {model} | {preview}"))
    return items


if __name__ == "__main__":
    port = int(os.getenv("OLLAMA_CHAT_PORT", "11500"))
    app.run_server(host="0.0.0.0", port=port, debug=False)
