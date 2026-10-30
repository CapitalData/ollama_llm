"""Dash chat frontend for Ollama with OpenTelemetry export to Arize Phoenix."""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
import plotly.io as pio
from flask import redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

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
SHARED_THREADS_PATH = STORAGE_DIR / "shared_threads.json"
USERS_PATH = STORAGE_DIR / "users.json"
PROJECT_MEMORY_DB_PATH = STORAGE_DIR / "project_memory.sqlite3"
PROJECT_MEMORY_INITIALIZED = False

MAX_MESSAGES = int(os.getenv("OLLAMA_MAX_MESSAGES", "12"))
MAX_LOG_ENTRIES = int(os.getenv("OLLAMA_MAX_LOG_ENTRIES", "500"))
SHARED_VIEW_REFRESH_MS = int(os.getenv("OLLAMA_SHARED_REFRESH_MS", "3000"))
SESSION_SECRET = os.getenv(
    "OLLAMA_CHAT_SECRET_KEY",
    secrets.token_hex(32),
)
ALLOW_OPEN_REGISTRATION = os.getenv("OLLAMA_ALLOW_REGISTRATION", "1") not in ("0", "false", "False")


def _load_users() -> dict[str, dict[str, Any]]:
    if not USERS_PATH.exists():
        return {}
    try:
        with USERS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(data: dict[str, dict[str, Any]]) -> None:
    with USERS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _load_users()
    if username in users:
        return False, "That username is already taken."
    users[username] = {
        "password_hash": generate_password_hash(password),
        "created": datetime.utcnow().isoformat() + "Z",
    }
    _save_users(users)
    return True, "Account created. You can now log in."


def verify_user(username: str, password: str) -> bool:
    username = (username or "").strip().lower()
    users = _load_users()
    record = users.get(username)
    if not record:
        return False
    return check_password_hash(record.get("password_hash", ""), password or "")


def _load_shared_threads() -> dict[str, dict[str, Any]]:
    if not SHARED_THREADS_PATH.exists():
        return {}
    try:
        with SHARED_THREADS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_shared_threads(data: dict[str, dict[str, Any]]) -> None:
    with SHARED_THREADS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def is_thread_shared(session_id: str) -> bool:
    return bool(_load_shared_threads().get(session_id, {}).get("enabled"))


def get_thread_owner(session_id: str) -> str | None:
    return _load_shared_threads().get(session_id, {}).get("owner")


def set_thread_shared(session_id: str, enabled: bool, owner: str | None = None) -> None:
    data = _load_shared_threads()
    existing = data.get(session_id, {})
    if enabled:
        data[session_id] = {
            "enabled": True,
            "owner": owner or existing.get("owner"),
            "created": existing.get("created", datetime.utcnow().isoformat() + "Z"),
            "updated": datetime.utcnow().isoformat() + "Z",
        }
    else:
        if session_id in data:
            data[session_id]["enabled"] = False
            data[session_id]["updated"] = datetime.utcnow().isoformat() + "Z"
    _save_shared_threads(data)


def load_thread_messages(session_id: str) -> list[dict[str, str]]:
    """Reconstruct a conversation (user/assistant message list) from the history log."""
    messages: list[dict[str, str]] = []
    for entry in read_history():
        if entry.get("session_id") != session_id:
            continue
        prompt = entry.get("prompt")
        response = entry.get("response")
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if response:
            messages.append({"role": "assistant", "content": response})
    return messages


def configure_json_engine() -> None:
    try:
        pio.json.config.default_engine = "json"
    except Exception:
        pass


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


def _connect_project_memory_db() -> sqlite3.Connection:
    """Open a SQLite connection to the local project memory DB."""
    PROJECT_MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PROJECT_MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def initialize_project_memory() -> None:
    """Create the project/topic/memory tables if they do not exist."""
    global PROJECT_MEMORY_INITIALIZED
    if PROJECT_MEMORY_INITIALIZED:
        return
    conn = _connect_project_memory_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                summary TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id),
                FOREIGN KEY(topic_id) REFERENCES topics(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    PROJECT_MEMORY_INITIALIZED = True


def ensure_project(name: str, description: str = "") -> dict[str, Any]:
    """Create or fetch a project and a default General topic."""
    project_name = (name or "").strip()
    if not project_name:
        raise ValueError("Project name is required.")
    initialize_project_memory()
    project_id = f"project-{project_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
    conn = _connect_project_memory_db()
    try:
        existing = conn.execute(
            "SELECT * FROM projects WHERE lower(name)=lower(?)",
            (project_name,),
        ).fetchone()
        if existing is None:
            project_id = f"project-{project_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
            created_at = datetime.utcnow().isoformat() + "Z"
            conn.execute(
                "INSERT INTO projects(id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (project_id, project_name, description, created_at),
            )
            topic_id = f"topic-{project_id}-{uuid.uuid4().hex[:8]}"
            conn.execute(
                "INSERT INTO topics(id, project_id, name, summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (topic_id, project_id, "General", "", created_at, created_at),
            )
            conn.commit()
            project_row = conn.execute(
                "SELECT * FROM projects WHERE id=?",
                (project_id,),
            ).fetchone()
            topic_row = conn.execute(
                "SELECT * FROM topics WHERE id=?",
                (topic_id,),
            ).fetchone()
            return {
                "id": project_row["id"],
                "name": project_row["name"],
                "description": project_row["description"],
                "created_at": project_row["created_at"],
                "topics": [
                    {
                        "id": topic_row["id"],
                        "project_id": topic_row["project_id"],
                        "name": topic_row["name"],
                        "summary": topic_row["summary"],
                        "created_at": topic_row["created_at"],
                        "updated_at": topic_row["updated_at"],
                    }
                ],
            }

        project_row = existing
        topic_row = conn.execute(
            "SELECT * FROM topics WHERE project_id=? ORDER BY created_at LIMIT 1",
            (project_row["id"],),
        ).fetchone()
        if topic_row is None:
            topic_id = f"topic-{project_row['id']}-{uuid.uuid4().hex[:8]}"
            created_at = datetime.utcnow().isoformat() + "Z"
            conn.execute(
                "INSERT INTO topics(id, project_id, name, summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (topic_id, project_row["id"], "General", "", created_at, created_at),
            )
            conn.commit()
            topic_row = conn.execute(
                "SELECT * FROM topics WHERE id=?",
                (topic_id,),
            ).fetchone()
        settings = {
            "id": project_row["id"],
            "name": project_row["name"],
            "description": project_row["description"],
            "created_at": project_row["created_at"],
            "topics": [
                {
                    "id": topic_row["id"],
                    "project_id": topic_row["project_id"],
                    "name": topic_row["name"],
                    "summary": topic_row["summary"],
                    "created_at": topic_row["created_at"],
                    "updated_at": topic_row["updated_at"],
                }
            ],
        }
        return settings
    finally:
        conn.close()


def save_memory_entry(
    project_id: str,
    topic_id: str,
    text: str,
    source: str,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a project/topic memory entry."""
    body = (text or "").strip()
    if not body:
        raise ValueError("Memory text cannot be empty.")
    initialize_project_memory()
    conn = _connect_project_memory_db()
    try:
        entry_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        conn.execute(
            """
            INSERT INTO memory_entries(id, project_id, topic_id, kind, source, text, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (entry_id, project_id, topic_id, kind, source, body, json.dumps(metadata or {}, sort_keys=True), created_at),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM memory_entries WHERE id=?",
            (entry_id,),
        ).fetchone()
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "topic_id": row["topic_id"],
            "kind": row["kind"],
            "source": row["source"],
            "text": row["text"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def get_project_context(project_id: str, topic_id: str | None = None) -> dict[str, Any]:
    """Return saved project/topic state and memory entries for a context window."""
    initialize_project_memory()
    conn = _connect_project_memory_db()
    try:
        project_row = conn.execute(
            "SELECT * FROM projects WHERE id=?",
            (project_id,),
        ).fetchone()
        if project_row is None:
            raise ValueError(f"Unknown project id: {project_id}")

        if topic_id is None:
            topic_row = conn.execute(
                "SELECT * FROM topics WHERE project_id=? ORDER BY created_at LIMIT 1",
                (project_id,),
            ).fetchone()
        else:
            topic_row = conn.execute(
                "SELECT * FROM topics WHERE id=? AND project_id=?",
                (topic_id, project_id),
            ).fetchone()
        if topic_row is None:
            raise ValueError(f"Unknown topic id for project: {project_id}")

        rows = conn.execute(
            """
            SELECT * FROM memory_entries
            WHERE project_id=? AND topic_id=?
            ORDER BY created_at DESC
            LIMIT 25
            """,
            (project_id, topic_row["id"]),
        ).fetchall()
        entries = []
        for row in rows:
            entries.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "source": row["source"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "created_at": row["created_at"],
                }
            )
        return {
            "project": {
                "id": project_row["id"],
                "name": project_row["name"],
                "description": project_row["description"],
                "created_at": project_row["created_at"],
            },
            "topic": {
                "id": topic_row["id"],
                "project_id": topic_row["project_id"],
                "name": topic_row["name"],
                "summary": topic_row["summary"],
                "created_at": topic_row["created_at"],
                "updated_at": topic_row["updated_at"],
            },
            "entries": entries,
        }
    finally:
        conn.close()


def fetch_models() -> list[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        response.raise_for_status()
        data = response.json()
        models = [item.get("name") for item in data.get("models", []) if item.get("name")]
        return models or [DEFAULT_MODEL]
    except requests.RequestException:
        return [DEFAULT_MODEL]


INITIAL_MODELS = fetch_models()
INITIAL_MODEL = INITIAL_MODELS[0] if INITIAL_MODELS else DEFAULT_MODEL


def ollama_health() -> bool:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def call_ollama(messages: list[dict[str, str]], model: str) -> tuple[str, float]:
    start = time.time()
    timeout = int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "300"))
    payload = {"model": model, "messages": messages, "stream": True}
    chunks = []
    with requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            message = chunk.get("message") or {}
            chunks.append(message.get("content", ""))
            if chunk.get("done"):
                break
    content = "".join(chunks) or "<no response>"
    elapsed = (time.time() - start) * 1000
    return content, elapsed


configure_json_engine()
setup_tracing()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Ollama Chat"

server = app.server
server.secret_key = SESSION_SECRET

LOGIN_PATH = "/login"
REGISTER_PATH = "/register"
LOGOUT_PATH = "/logout"
PUBLIC_PATHS = {LOGIN_PATH, REGISTER_PATH, LOGOUT_PATH}


def _auth_page(title: str, body_html: str, message: str = "", message_color: str = "danger") -> str:
    alert = (
        f'<div class="alert alert-{message_color}" role="alert">{message}</div>'
        if message
        else ""
    )
    register_link = (
        f'<p class="mt-3 text-center"><a href="{REGISTER_PATH}">Create an account</a></p>'
        if ALLOW_OPEN_REGISTRATION
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet"
 href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>body{{background:#1a1a1a;color:#eee;padding-top:60px}}
.card{{background:#222;border:1px solid #333}}</style>
</head><body>
<div class="container" style="max-width:420px">
  <h3 class="text-center mb-3">{title}</h3>
  <div class="card"><div class="card-body">
    {alert}
    {body_html}
    {register_link if title == "Sign in" else ""}
  </div></div>
</div></body></html>"""


def _login_form_html(next_url: str = "/") -> str:
    return f"""<form method="post" action="{LOGIN_PATH}">
  <input type="hidden" name="next" value="{next_url}">
  <div class="mb-2"><label class="form-label">Username</label>
    <input class="form-control" name="username" autofocus required></div>
  <div class="mb-2"><label class="form-label">Password</label>
    <input class="form-control" name="password" type="password" required></div>
  <button class="btn btn-success w-100" type="submit">Sign in</button>
</form>"""


def _register_form_html() -> str:
    return f"""<form method="post" action="{REGISTER_PATH}">
  <div class="mb-2"><label class="form-label">Username</label>
    <input class="form-control" name="username" autofocus required></div>
  <div class="mb-2"><label class="form-label">Password</label>
    <input class="form-control" name="password" type="password" required minlength="6"></div>
  <button class="btn btn-primary w-100" type="submit">Create account</button>
  <p class="mt-3 text-center"><a href="{LOGIN_PATH}">Back to sign in</a></p>
</form>"""


@server.route(LOGIN_PATH, methods=["GET", "POST"])
def login_route():
    next_url = request.values.get("next", "/")
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if verify_user(username, password):
            session.clear()
            session["username"] = username.strip().lower()
            return redirect(next_url or "/")
        return _auth_page("Sign in", _login_form_html(next_url), "Invalid credentials.")
    return _auth_page("Sign in", _login_form_html(next_url))


@server.route(REGISTER_PATH, methods=["GET", "POST"])
def register_route():
    if not ALLOW_OPEN_REGISTRATION:
        return _auth_page(
            "Registration disabled",
            f'<p class="text-center"><a href="{LOGIN_PATH}">Back to sign in</a></p>',
            "Open registration is disabled. Ask the administrator to create your account.",
            message_color="warning",
        )
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        ok, msg = create_user(username, password)
        if ok:
            return _auth_page("Sign in", _login_form_html("/"), msg, message_color="success")
        return _auth_page("Create account", _register_form_html(), msg)
    return _auth_page("Create account", _register_form_html())


@server.route(LOGOUT_PATH)
def logout_route():
    session.clear()
    return redirect(LOGIN_PATH)


@server.before_request
def require_login():
    path = request.path or "/"
    if path in PUBLIC_PATHS:
        return None
    # Allow static asset and Dash-internal routes only when logged in.
    if "username" in session:
        return None
    # POST to Dash callback endpoint while unauthenticated -> 401 (so client gets a clean error).
    if path.startswith("/_dash-"):
        return ("Unauthorized", 401)
    next_url = request.full_path if request.query_string else path
    return redirect(f"{LOGIN_PATH}?next={next_url}")


def current_username() -> str | None:
    return session.get("username")

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="session-id", data=str(uuid.uuid4())),
        dcc.Store(id="conversation-store", data=[]),
        dcc.Store(id="model-store", data=INITIAL_MODELS),
        dcc.Store(id="view-mode", data="owner"),  # "owner" or "shared"
        dcc.Interval(id="health-ticker", interval=3000, n_intervals=0),
        dcc.Interval(
            id="shared-refresh-ticker",
            interval=SHARED_VIEW_REFRESH_MS,
            n_intervals=0,
            disabled=True,
        ),
        dbc.Row(
            [
                dbc.Col(html.H2("Ollama Chat Console", className="mt-3"), width=9),
                dbc.Col(
                    html.Div(id="user-info", className="text-end mt-3"),
                    width=3,
                ),
            ]
        ),
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
                                                for model in INITIAL_MODELS
                                            ],
                                            value=INITIAL_MODEL,
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
                                dbc.CardHeader("Share Thread"),
                                dbc.CardBody(
                                    [
                                        dbc.Switch(
                                            id="share-toggle",
                                            label="Enable shareable link",
                                            value=False,
                                        ),
                                        html.Div(
                                            "Anyone with the link can view this thread "
                                            "(read-only) and will see new messages as you "
                                            "send them.",
                                            className="text-muted small mt-1",
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(
                                                    id="share-url",
                                                    value="",
                                                    readonly=True,
                                                    type="text",
                                                ),
                                                dcc.Clipboard(
                                                    target_id="share-url",
                                                    title="Copy link",
                                                    style={
                                                        "display": "inline-block",
                                                        "fontSize": "1.2em",
                                                        "marginLeft": "8px",
                                                        "verticalAlign": "middle",
                                                    },
                                                ),
                                            ],
                                            className="mt-2",
                                            id="share-url-group",
                                            style={"display": "none"},
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
                                    dcc.Loading(
                                        type="circle",
                                        color="#00d4ff",
                                        children=html.Div(
                                            [
                                                html.Div(
                                                    id="view-banner",
                                                    className="mb-2",
                                                ),
                                                html.Div(
                                                    id="chat-window",
                                                    style={"whiteSpace": "pre-wrap"},
                                                ),
                                                html.Div(
                                                    id="send-error",
                                                    className="text-danger mt-2",
                                                ),
                                            ]
                                        ),
                                    )
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


def get_ollama_connection_display() -> str:
    """Return Ollama connection info including tunnel address if configured."""
    tunnel_host = os.getenv("OLLAMA_LLM_PROXY_HOST", "")
    tunnel_port = int(os.getenv("OLLAMA_LLM_PROXY_REMOTE_PORT", "19053"))
    
    if tunnel_host:
        return f"Internal: {OLLAMA_ENDPOINT} | External (tunnel): {tunnel_host}:{tunnel_port}"
    return OLLAMA_ENDPOINT


@callback(
    Output("ollama-status", "children"),
    Output("phoenix-status", "children"),
    Input("health-ticker", "n_intervals"),
)
def update_status(_ticks):
    ollama_ok = ollama_health()
    ollama_conn = get_ollama_connection_display()
    ollama_text = f"Ollama: {'online' if ollama_ok else 'offline'} ({ollama_conn})"
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
    prompt = (prompt or "").strip()
    username = current_username()
    if not username:
        return conversation, _render_conversation(conversation), _render_history(), "", "You must be signed in."

    # If thread is shared and has an owner, only the owner can post.
    owner = get_thread_owner(session_id)
    if owner and owner != username:
        return (
            conversation,
            _render_conversation(conversation),
            _render_history(),
            prompt,
            f"Read-only: only {owner} can post in this thread.",
        )

    if not prompt:
        return conversation, _render_conversation(conversation), _render_history(), "", "Prompt is empty"

    conversation = list(conversation or [])

    selected_model = model
    if not selected_model:
        models = fetch_models()
        selected_model = models[0] if models else DEFAULT_MODEL

    conversation.append({"role": "user", "content": prompt})

    error_message = ""
    response_text = ""
    latency_ms = 0.0

    with tracer().start_as_current_span("ollama.chat") as span:
        span.set_attribute("openinference.project.name", PHOENIX_PROJECT_NAME)
        span.set_attribute("ollama.model", selected_model)
        span.set_attribute("ollama.prompt.length", len(prompt))
        span.set_attribute("ollama.session_id", session_id)
        try:
            response_text, latency_ms = call_ollama(conversation, selected_model)
            span.set_attribute("ollama.response.length", len(response_text))
            span.set_attribute("ollama.latency_ms", latency_ms)
            conversation.append({"role": "assistant", "content": response_text})
        except requests.RequestException as exc:
            error_message = f"Ollama request failed: {exc}"
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, error_message))
        except Exception as exc:
            error_message = f"Unexpected error: {exc}"
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR, error_message))

    if response_text:
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "model": selected_model,
            "prompt": prompt,
            "response": response_text,
            "latency_ms": latency_ms,
        }
        append_history(entry)

    conversation = conversation[-MAX_MESSAGES:]
    chat_view = _render_conversation(conversation)
    history_view = _render_history()
    if error_message:
        status_message = f"ERROR: {error_message}"
    else:
        status_message = f"OK: Reply received from {selected_model} in {latency_ms:.0f} ms"
    return conversation, chat_view, history_view, "", status_message


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


def _parse_thread_from_search(search: str | None) -> str | None:
    if not search:
        return None
    from urllib.parse import parse_qs

    qs = parse_qs(search.lstrip("?"))
    values = qs.get("thread")
    return values[0] if values else None


@callback(
    Output("session-id", "data"),
    Output("view-mode", "data"),
    Output("conversation-store", "data", allow_duplicate=True),
    Output("share-toggle", "value"),
    Output("shared-refresh-ticker", "disabled"),
    Output("view-banner", "children"),
    Input("url", "search"),
    State("session-id", "data"),
    prevent_initial_call="initial_duplicate",
)
def handle_url(search, current_session_id):
    thread_id = _parse_thread_from_search(search)
    if not thread_id:
        return (
            current_session_id,
            "owner",
            dash.no_update,
            is_thread_shared(current_session_id),
            True,
            "",
        )

    if not is_thread_shared(thread_id):
        banner = dbc.Alert(
            "This shared thread is not available (sharing disabled or never enabled).",
            color="warning",
        )
        return current_session_id, "shared", [], False, True, banner

    messages = load_thread_messages(thread_id)
    banner = dbc.Alert(
        "Viewing shared thread (read-only). Updates appear automatically.",
        color="info",
    )
    return thread_id, "shared", messages, True, False, banner


@callback(
    Output("share-url", "value"),
    Output("share-url-group", "style"),
    Input("share-toggle", "value"),
    Input("session-id", "data"),
    State("url", "href"),
    State("view-mode", "data"),
)
def toggle_share(enabled, session_id, href, view_mode):
    # Don't allow viewers to re-share a thread they're viewing read-only.
    if view_mode == "shared":
        if enabled and href:
            return href, {"display": "flex"}
        return "", {"display": "none"}

    username = current_username()
    if not username:
        return "", {"display": "none"}

    # Only the owner (or a thread with no owner yet) can enable/disable sharing.
    existing_owner = get_thread_owner(session_id)
    if existing_owner and existing_owner != username:
        return "", {"display": "none"}

    set_thread_shared(session_id, bool(enabled), owner=username)
    if not enabled or not href:
        return "", {"display": "none"}

    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(href)
    share_url = urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, "", f"thread={session_id}", "")
    )
    return share_url, {"display": "flex"}


@callback(
    Output("conversation-store", "data", allow_duplicate=True),
    Output("chat-window", "children", allow_duplicate=True),
    Input("shared-refresh-ticker", "n_intervals"),
    State("session-id", "data"),
    State("view-mode", "data"),
    prevent_initial_call=True,
)
def refresh_shared_view(_n, session_id, view_mode):
    if view_mode != "shared" or not session_id:
        return dash.no_update, dash.no_update
    if not is_thread_shared(session_id):
        return [], dbc.Alert("Sharing has been disabled for this thread.", color="warning")
    messages = load_thread_messages(session_id)
    return messages, _render_conversation(messages)


@callback(
    Output("prompt-input", "disabled"),
    Output("send-button", "disabled"),
    Input("view-mode", "data"),
    Input("session-id", "data"),
)
def lock_input_for_shared_view(view_mode, session_id):
    username = current_username()
    owner = get_thread_owner(session_id)
    # Lock the input if: viewing a shared thread you don't own,
    # OR not logged in.
    locked = (not username) or (owner is not None and owner != username) or view_mode == "shared" and owner != username
    return locked, locked


@callback(
    Output("user-info", "children"),
    Input("session-id", "data"),  # any trigger; runs on load
)
def show_user_info(_session_id):
    username = current_username()
    if not username:
        return ""
    return html.Span(
        [
            f"Signed in as {username} \u00b7 ",
            html.A("Sign out", href=LOGOUT_PATH, className="text-info"),
        ]
    )


if __name__ == "__main__":
    port = int(os.getenv("OLLAMA_CHAT_PORT", "11500"))
    app.run(host="0.0.0.0", port=port, debug=False)
