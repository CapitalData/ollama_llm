# Ollama LLM — Quick Start

## Prerequisites

| Tool | Minimum | Install |
|------|---------|---------|
| Python | 3.11+ | Via `uv` or [python.org](https://www.python.org) |
| uv | 0.4+ | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Ollama | latest | [ollama.com](https://ollama.com) |

## Environment Setup

```bash
# From the meta-repo root (shared venv):
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r ollama_llm/requirements.txt

# Or standalone:
cd ollama_llm
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Model Management

**Important:** Before using the chat app, you must install at least one model. The 404 error `{"error":"model 'model_name' not found"}` means the model isn't installed.

### Required Models

| Model | Size | Purpose |
|-------|------|----------|
| **llama3.1** | 4.9 GB | Default chat model (recommended) |
| llama3.2:1b | 1.3 GB | Lightweight alternative for limited RAM |
| mistral | 4.1 GB | Alternative general-purpose model |

### Install Models

**If `ollama` CLI is available:**
```bash
ollama pull llama3.1
ollama pull mistral
```

**If `ollama` CLI is not in PATH (use API):**
```bash
# Install llama3.1 (default model)
curl -X POST http://127.0.0.1:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.1"}'

# Or without streaming output:
curl -X POST http://127.0.0.1:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.1", "stream": false}'
```

**Note:** Large models (4-7 GB) can take several minutes to download depending on internet speed.

### List Installed Models

**CLI:**
```bash
ollama list
```

**API:**
```bash
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool
```

### Test a Model

```bash
curl -X POST http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

Successful response returns status 200 with the model's reply. A 404 error means the model isn't installed.

### Delete a Model

**CLI:**
```bash
ollama rm llama3.1
```

**API:**
```bash
curl -X DELETE http://127.0.0.1:11434/api/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.1"}'
```

## Run

### Ollama Server Wrapper

```bash
python app.py
```

Starts/manages the local Ollama instance on port **11434**.

### Chat Interface

```bash
python ollama_chat_app.py
```

Opens a Dash chat UI on port **11500**. The first page you see is `/login` —
follow the **First-Time Login** steps below before chatting.

### First-Time Login

1. Visit `http://127.0.0.1:11500/` — you'll be redirected to `/login`.
2. Click **"Create an account"**, pick a username + password (≥6 chars), submit.
3. Sign in. Your username appears top-right with a **Sign out** link.
4. To lock down the instance afterward, restart with
   `OLLAMA_ALLOW_REGISTRATION=0` so no new accounts can be created via the web.
   You can still add users from a Python shell:

   ```bash
   python -c "from ollama_chat_app import create_user; print(create_user('bob','his-password'))"
   ```

For production, also set a stable session key so logins survive restarts:

```bash
export OLLAMA_CHAT_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"
```

### Sharing a Thread

Once signed in:

1. Start chatting — every turn is tagged with your browser's `session_id`.
2. In the sidebar **Share Thread** card, flip **Enable shareable link** on.
3. Copy the generated URL (`http://<host>:11500/?thread=<id>`) and send it to
   a teammate.
4. They must sign in (or register, if open) before they can view. Once in,
   they see the conversation read-only and the page auto-refreshes every
   ~3 seconds, so your new turns appear without a manual reload.
5. Flip the switch off to revoke. Viewers see *"Sharing has been disabled for
   this thread."* on their next poll.

Only the owner (the account that first enabled sharing) can post messages or
toggle sharing for that thread. Anyone else hitting Send gets
*"Read-only: only `<owner>` can post in this thread."*

### Via Control Panel

Both components are typically launched from the ACD Control Panel LLM panel.
See [meta-repo QUICKSTART](../QUICKSTART.md) for the full stack startup order.

## Configuration

Set these in your environment or a `.env` file in the meta-repo root:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `0.0.0.0` | Ollama bind address |
| `OLLAMA_PORT` | `11434` | Ollama API port |
| `OLLAMA_ENDPOINT` | `http://127.0.0.1:11434` | Chat UI → Ollama API |
| `OLLAMA_MODEL` | `llama3.1` | Default model |
| `OLLAMA_MAX_MESSAGES` | `12` | Max context window messages |
| `OLLAMA_STORAGE_DIR` | `./storage` | Chat history directory |
| `OLLAMA_SHARED_REFRESH_MS` | `3000` | Poll interval (ms) for shared-thread viewers |
| `OLLAMA_CHAT_SECRET_KEY` | _(random per process)_ | Flask session signing key. **Set in production** so logins survive restarts. |
| `OLLAMA_ALLOW_REGISTRATION` | `1` | `0` disables self-signup at `/register` |
| `PHOENIX_PROJECT_NAME` | `default` | Trace project tag |
| `PHOENIX_OTLP_ENDPOINT` | `http://127.0.0.1:6006/v1/traces` | Phoenix trace endpoint |
| `OTEL_SERVICE_NAME` | `ollama-dash-chat` | OpenTelemetry service name |

## Connection Strings

Use these to connect to your running Ollama instance from code.

### Local (same machine)

| Service | URL |
|---------|-----|
| Ollama API | `http://127.0.0.1:11434` |
| Chat UI | `http://127.0.0.1:11500` |
| Phoenix Traces | `http://127.0.0.1:6006` |

### Remote (other machines on your network)

Replace `<YOUR_IP>` with the host machine's IP address.
Find it with: `ipconfig getifaddr en0` (macOS) or `hostname -I` (Linux).

| Service | URL |
|---------|-----|
| Ollama API | `http://<YOUR_IP>:11434` |
| Chat UI | `http://<YOUR_IP>:11500` |
| Phoenix Traces | `http://<YOUR_IP>:6006` |

### Code Snippets

**Python (requests):**
```python
import requests
OLLAMA_URL = "http://127.0.0.1:11434"  # or http://<YOUR_IP>:11434

response = requests.post(f"{OLLAMA_URL}/api/generate", json={
    "model": "llama3.1",
    "prompt": "Hello, world!",
    "stream": False
})
print(response.json()["response"])
```

**Python (ollama library):**
```python
import ollama
client = ollama.Client(host="http://127.0.0.1:11434")
response = client.chat(model="llama3.1", messages=[
    {"role": "user", "content": "Hello, world!"}
])
print(response["message"]["content"])
```

**curl:**
```bash
curl http://127.0.0.1:11434/api/generate \
  -d '{"model": "llama3.1", "prompt": "Hello", "stream": false}'
```

## Data Storage

| Path | Contains | Gitignored |
|------|----------|------------|
| `storage/ollama_chat_history.jsonl` | Chat conversation log (one JSON line per turn) | Yes |
| `storage/users.json` | Account credentials (`{username: {password_hash, created}}`) | Yes |
| `storage/shared_threads.json` | Thread-sharing registry (`{session_id: {enabled, owner, ...}}`) | Yes |

## Lift and Shift — Data Migration

User data in this component lives entirely under `storage/`:

```bash
# Export (from source machine):
cp -r ollama_llm/storage/ /tmp/ollama_backup/

# Import (on destination machine):
mkdir -p ollama_llm/storage
cp /tmp/ollama_backup/*.jsonl /tmp/ollama_backup/*.json ollama_llm/storage/
```

Be sure to also carry over (or regenerate) `OLLAMA_CHAT_SECRET_KEY` — if it
changes, all existing login sessions are invalidated (users have to sign in
again) but their accounts and threads are preserved.

Or use the meta-repo migration scripts (`acd_lift.sh` / `acd_shift.sh`) which
handle this automatically. See [meta-repo QUICKSTART](../QUICKSTART.md#7-lift-and-shift--data-migration).

## Troubleshooting

### "Connection refused" on chat UI
Ollama server must be running first. Start it with `python app.py` or via the control panel.

### "404 Client Error: Not Found" or "model 'xyz' not found"

This means the model isn't installed in Ollama. Install it first:

**With CLI:**
```bash
ollama pull llama3.1
```

**Without CLI (API method):**
```bash
curl -X POST http://127.0.0.1:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.1"}'
```

See the **Model Management** section above for complete instructions and common models.

### Traces not appearing in Phoenix
Verify `PHOENIX_OTLP_ENDPOINT` points to a running Phoenix instance (port 6006).

### Stuck on `/login` after restart
Expected if `OLLAMA_CHAT_SECRET_KEY` isn't set — the session cookie is signed
with a random per-process key, so restarts invalidate all sessions. Set a
stable key (see the **First-Time Login** section) and restart once more.

### Shared link shows "not available"
The owner has either disabled sharing or never enabled it on that
`session_id`. Ask them to flip **Enable shareable link** back on. Each browser
tab gets its own `session_id`, so an owner sharing tab A doesn't share tab B.

### "Read-only: only `<owner>` can post"
You're signed in as a different account than the one that started the thread.
Only the original owner can post; everyone else views read-only. To take over,
the owner must disable sharing or you must sign in as them.

### Lost admin password
Stop the server, edit `storage/users.json` to remove the affected entry, then
re-register the username (or call `create_user()` from a Python shell).
