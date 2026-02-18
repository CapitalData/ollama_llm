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

**OpenTelemetry Tracing:**
- `PHOENIX_PROJECT_NAME` - Project tag for traces (default: `default`)
- `PHOENIX_OTLP_ENDPOINT` - Phoenix trace endpoint (default: `http://127.0.0.1:6006/v1/traces`)
- `OTEL_SERVICE_NAME` - Service identifier (default: `ollama-dash-chat`)

## Data Storage

Chat history is stored in `storage/ollama_chat_history.jsonl` (excluded from git).

## Features

- Multi-turn conversations with context management
- Model selection from available Ollama models
- Streaming responses
- OpenTelemetry span tracing for every chat interaction
- Project-based trace tagging
- Automatic chat history persistence
- Clear chat functionality

## Requirements

- Python 3.11+
- Ollama installed locally
- Dash + Dash Bootstrap Components
- OpenTelemetry SDK + OTLP exporter

## Part of ACD Meta-Repo

See the [meta-repo documentation](https://github.com/CapitalData/acd-meta-repo) for full stack setup.
