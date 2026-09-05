#!/usr/bin/env python
"""WorkOS Web UI Launcher with Port Auto-Reclaim and Ctrl+Z Instant-Termination."""

import os
import signal
import socket
import subprocess
import time

import uvicorn

from workos_engine.infra_manager import ensure_all_infrastructure
from workos_engine.ui.app import app


def free_port_if_bound(port: int = 8000) -> None:
    """Detects if port is held by any existing/suspended process and terminates it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_bound = s.connect_ex(("127.0.0.1", port)) == 0

    if not is_bound:
        return

    print(f"⚠️ Port {port} is occupied by an existing process. Reclaiming port...")
    try:
        # Find PID using lsof
        res = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        pids = [p.strip() for p in res.stdout.strip().split("\n") if p.strip()]
        for pid_str in pids:
            if pid_str and pid_str != str(os.getpid()):
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                    print(f"  ✔ Terminated stale process PID {pid_str} holding port {port}")
                except ProcessLookupError:
                    pass
        time.sleep(0.5)
    except Exception:
        # Fallback to fuser
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
        time.sleep(0.5)


def setup_clean_exit_handlers(port: int = 8000) -> None:
    """
    Intercepts SIGTSTP (Ctrl+Z) and SIGINT (Ctrl+C).
    Instead of suspending into the background (which leaves port 8000 locked),
    this forces an immediate, clean termination and socket release.
    """

    def _exit_handler(sig, frame):
        sig_name = (
            "Ctrl+Z (SIGTSTP)"
            if sig == signal.SIGTSTP
            else ("Ctrl+C (SIGINT)" if sig == signal.SIGINT else f"Signal {sig}")
        )
        print(
            f"\n✦ Intercepted {sig_name} — Instantly terminating Argus OS and releasing port {port}..."
        )
        free_port_if_bound(port)
        os._exit(0)

    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _exit_handler)
    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)


if __name__ == "__main__":
    setup_clean_exit_handlers(port=8000)
    free_port_if_bound(port=8000)

    print("✦ Initializing Argus OS Infrastructure (Ollama, Elasticsearch, SearXNG)...")
    ensure_all_infrastructure(verbose=True)

    print("✦ Starting Argus OS Executive Web Interface on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
