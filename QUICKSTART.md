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

Opens a Dash chat UI on port **11500**.

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
| `storage/ollama_chat_history.jsonl` | Chat conversation log | Yes |

## Lift and Shift — Data Migration

The only user data in this component is the chat history file:

```bash
# Export (from source machine):
cp ollama_llm/storage/ollama_chat_history.jsonl /tmp/ollama_backup/

# Import (on destination machine):
mkdir -p ollama_llm/storage
cp /tmp/ollama_backup/ollama_chat_history.jsonl ollama_llm/storage/
```

Or use the meta-repo migration scripts (`acd_lift.sh` / `acd_shift.sh`) which
handle this automatically. See [meta-repo QUICKSTART](../QUICKSTART.md#7-lift-and-shift--data-migration).

## Troubleshooting

### "Connection refused" on chat UI
Ollama server must be running first. Start it with `python app.py` or via the control panel.

### Model not found
Pull the model first: `ollama pull llama3.1`

### Traces not appearing in Phoenix
Verify `PHOENIX_OTLP_ENDPOINT` points to a running Phoenix instance (port 6006).
