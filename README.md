# Ollama LLM Chat Interface

Dash-based chat interface for Ollama with OpenTelemetry tracing to Phoenix/Arize.

## Purpose

This component provides:
- **Ollama server management** (`app.py`) - Starts/manages local Ollama instance
- **Chat UI** (`ollama_chat_app.py`) - Dash web interface for conversing with Ollama models
- **OpenTelemetry integration** - Exports traces to Phoenix for observability

## Usage

### Standalone Ollama Server

```bash
python app.py
```

### Chat Interface

```bash
python ollama_chat_app.py
```

Defaults to `http://127.0.0.1:8051`

### Via Control Panel

Both components are typically launched from the ACD Control Panel as part of the LLM stack.

## Configuration

Environment variables:

**Ollama Server:**
- `OLLAMA_HOST` - Bind address (default: `0.0.0.0`)
- `OLLAMA_PORT` - Port for Ollama API (default: `11434`)

**Chat UI:**
- `OLLAMA_ENDPOINT` - Ollama API endpoint (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` - Default model (default: `llama3.1`)
- `OLLAMA_MAX_MESSAGES` - Max context messages (default: `12`)
- `OLLAMA_STORAGE_DIR` - Chat history storage (default: `./storage`)
- `OLLAMA_SHARED_REFRESH_MS` - Poll interval for shared-thread viewers (default: `3000`)

**Authentication:**
- `OLLAMA_CHAT_SECRET_KEY` - Flask session signing key. **Set this in production** so sessions survive restarts. If unset a random key is generated per-process (sessions invalidated on restart).
- `OLLAMA_ALLOW_REGISTRATION` - `1` (default) lets visitors create accounts at `/register`. Set to `0` to disable self-signup; new users must then be added by editing `storage/users.json` directly.

**OpenTelemetry Tracing:**
- `PHOENIX_PROJECT_NAME` - Project tag for traces (default: `default`)
- `PHOENIX_OTLP_ENDPOINT` - Phoenix trace endpoint (default: `http://127.0.0.1:6006/v1/traces`)
- `OTEL_SERVICE_NAME` - Service identifier (default: `ollama-dash-chat`)

## Data Storage

Stored under `OLLAMA_STORAGE_DIR` (default `./storage/`, gitignored):

| File | Contents |
|------|----------|
| `ollama_chat_history.jsonl` | One JSON line per turn: `{id, timestamp, session_id, model, prompt, response, latency_ms}` |
| `users.json` | `{username: {password_hash, created}}` — Werkzeug PBKDF2 hashes |
| `shared_threads.json` | `{session_id: {enabled, owner, created, updated}}` — sharing registry |

## Authentication

The entire chat UI is gated by a Flask session login. Visiting any page (or
posting any Dash callback) while signed out redirects to `/login`.

| Route | Purpose |
|-------|---------|
| `/login` | Sign in with username + password |
| `/register` | Create an account (disabled by `OLLAMA_ALLOW_REGISTRATION=0`) |
| `/logout` | Clear the session and return to login |

Passwords are hashed with `werkzeug.security.generate_password_hash` (PBKDF2)
before being written to `storage/users.json`. The current username appears in the
top-right header alongside a "Sign out" link.

To create users from the shell instead of the registration form:

```python
from ollama_chat_app import create_user
create_user("alice", "her-strong-password")
```

## Shareable Threads

Each browser session is assigned a UUID (`session_id`) that tags every chat
turn in the history log. The **Share Thread** card in the sidebar exposes that
session as a shareable, live-updating URL.

**As the owner:**

1. Sign in and start chatting normally.
2. Flip on **Enable shareable link**. A URL of the form
   `http://<host>:11500/?thread=<session_id>` appears with a copy button.
3. Keep chatting — every new turn is appended to the log and pushed to viewers
   on their next poll (default every 3 seconds, see `OLLAMA_SHARED_REFRESH_MS`).
4. Flip the switch off to revoke access. Existing viewers see a banner saying
   sharing was disabled and stop receiving updates.

**As a viewer:**

1. Open the shared URL. If not signed in, you'll be sent to `/login` first
   (then redirected back).
2. The conversation loads in **read-only** mode: the prompt textarea and Send
   button are disabled, and a banner reads
   *"Viewing shared thread (read-only). Updates appear automatically."*
3. The view auto-refreshes via a `dcc.Interval`; no manual reload needed.

**Ownership rules (enforced server-side in the callbacks):**

| Action | Allowed for |
|--------|-------------|
| Toggle sharing on/off | Thread owner only (the user who first enabled it) |
| Post new messages | Thread owner only — others get *"Read-only: only `<owner>` can post in this thread."* |
| View shared thread | Any signed-in user with the link |
| Visit any page | Signed-in users only |

The owner is recorded in `shared_threads.json` the first time sharing is
enabled, and is the only account that can disable sharing or post new turns.

## Features

- Multi-turn conversations with context management
- Model selection from available Ollama models
- Streaming responses
- OpenTelemetry span tracing for every chat interaction
- Project-based trace tagging
- Automatic chat history persistence
- **Login-gated access with per-user accounts**
- **Shareable read-only thread links with live updates**

## Requirements

- Python 3.11+
- Ollama installed locally
- Dash + Dash Bootstrap Components
- Werkzeug (password hashing — installed transitively with Flask/Dash)
- OpenTelemetry SDK + OTLP exporter

## Part of ACD Meta-Repo

See the [meta-repo documentation](https://github.com/CapitalData/acd-meta-repo) for full stack setup.
