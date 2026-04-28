"""Ollama wrapper for the Control Panel.

Starts (or attaches to) a local Ollama server so the control panel can manage it
like any other reactor process.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "0.0.0.0")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_ENDPOINT = f"http://127.0.0.1:{OLLAMA_PORT}"
OLLAMA_TAGS_URL = f"{OLLAMA_ENDPOINT}/api/tags"

_child_process: subprocess.Popen[str] | None = None
_owns_process = False


def _log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _check_endpoint() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=2) as response:
            return 200 <= response.status < 300
    except Exception:
        # Catch URLError, RemoteDisconnected, ConnectionRefused, etc.
        return False


def _ensure_ollama() -> bool:
    if shutil.which("ollama"):
        return True

    _log("Ollama binary not found on PATH.")
    if sys.platform == "darwin":
        _log("Install with: brew install ollama")
    else:
        _log("Install with: curl -fsSL https://ollama.com/install.sh | sh")
    return False


def _forward_output(stream: subprocess.PIPE[str]) -> None:
    for line in iter(stream.readline, ""):
        if not line:
            break
        print(line.rstrip(), flush=True)


def _start_ollama() -> None:
    global _child_process, _owns_process

    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST
    env["OLLAMA_PORT"] = str(OLLAMA_PORT)

    _child_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    _owns_process = True

    if _child_process.stdout:
        thread = threading.Thread(
            target=_forward_output, args=(_child_process.stdout,), daemon=True
        )
        thread.start()

    _log(f"Started ollama serve (pid {_child_process.pid})")


def _wait_for_health(timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _check_endpoint():
            return True
        time.sleep(1)
    return False


def _handle_shutdown(signum: int, _frame) -> None:
    _log(f"Received signal {signum}; shutting down")
    if _child_process and _owns_process:
        _child_process.terminate()
        try:
            _child_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _child_process.kill()
    raise SystemExit(0)


def _get_connection_info() -> str:
    """Return connection address(es) for Ollama."""
    local_addr = OLLAMA_ENDPOINT
    tunnel_host = os.getenv("OLLAMA_LLM_PROXY_HOST", "")
    tunnel_port = int(os.getenv("OLLAMA_LLM_PROXY_REMOTE_PORT", "19053"))
    
    if tunnel_host:
        return f"Local: {local_addr} | External (via tunnel): {tunnel_host}:{tunnel_port}"
    return local_addr


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    if not _ensure_ollama():
        return 1

    if _check_endpoint():
        _log("Ollama already running; monitoring existing service")
    else:
        _log("Ollama not detected; starting server")
        _start_ollama()
        if _wait_for_health():
            _log(f"Ollama ready at {_get_connection_info()}")
        else:
            _log("Ollama did not pass health checks yet; will keep monitoring")

    try:
        while True:
            if _check_endpoint():
                _log("Ollama healthy")
                time.sleep(30)
            else:
                _log("Ollama not responding; waiting")
                time.sleep(5)
    except KeyboardInterrupt:
        _handle_shutdown(signal.SIGINT, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
