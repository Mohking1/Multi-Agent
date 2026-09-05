"""WorkOS Infrastructure Auto-Starter and Health Manager.

Ensures all required local backing services (Ollama with GTX 1650-safe flags,
Docker-compose containers for Elasticsearch and SearXNG) are automatically detected,
started, and verified healthy whenever WorkOS or its UI starts up.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("workos.infra")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


def _check_http(url: str, timeout: float = 2.0) -> bool:
    """Checks if an HTTP endpoint is reachable and returning a 2xx/3xx/4xx status."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            return resp.status_code < 500
    except Exception:
        return False


def is_ollama_running(base_url: str = "http://127.0.0.1:11434") -> bool:
    """Checks if Ollama inference engine is reachable on the specified base URL."""
    return _check_http(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)


def is_elasticsearch_running(url: str = "http://localhost:9200") -> bool:
    """Checks if Elasticsearch is reachable."""
    return _check_http(url.rstrip("/"), timeout=1.5)


def is_searxng_running(url: str = "http://localhost:8080") -> bool:
    """Checks if SearXNG metasearch is reachable."""
    return _check_http(url.rstrip("/"), timeout=1.5)


def is_docker_available() -> bool:
    """Checks if Docker command line and daemon socket are reachable."""
    if not shutil.which("docker"):
        return False
    try:
        res = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
        )
        return res.returncode == 0
    except Exception:
        return False


def ensure_docker_daemon() -> bool:
    """Attempts to start docker daemon if inactive (e.g., via systemctl)."""
    if is_docker_available():
        return True

    logger.info("Docker daemon is not responding. Attempting to start docker service...")
    try:
        subprocess.run(["systemctl", "start", "docker"], timeout=5.0)
        time.sleep(2.0)
        return is_docker_available()
    except Exception as e:
        logger.debug(f"Could not start docker via systemctl: {e}")
        return False


def ensure_docker_containers(timeout: int = 30) -> dict[str, bool]:
    """
    Checks if Elasticsearch and SearXNG containers are running.
    If either is down, executes `docker compose up -d` in project root.
    """
    es_up = is_elasticsearch_running()
    searx_up = is_searxng_running()

    if es_up and searx_up:
        return {"elasticsearch": True, "searxng": True}

    if not is_docker_available():
        ensure_docker_daemon()

    if not is_docker_available():
        logger.warning(
            "Docker daemon is not reachable. Cannot auto-start Elasticsearch / SearXNG containers."
        )
        return {"elasticsearch": es_up, "searxng": searx_up}

    logger.info(
        "One or more Docker services are down. Starting containers via `docker compose up -d`..."
    )
    try:
        cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"]
        subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True, timeout=60.0
        )
    except Exception as e:
        logger.error(f"Failed to execute docker compose up -d: {e}")
        return {"elasticsearch": is_elasticsearch_running(), "searxng": is_searxng_running()}

    # Poll for health with timeout
    deadline = time.time() + timeout
    while time.time() < deadline:
        es_up = is_elasticsearch_running()
        searx_up = is_searxng_running()
        if es_up and searx_up:
            break
        time.sleep(1.0)

    return {"elasticsearch": is_elasticsearch_running(), "searxng": is_searxng_running()}


def ensure_ollama(
    port: int = 11434,
    timeout: int = 20,
    models_dir: str = "/usr/share/ollama/.ollama/models",
) -> bool:
    """
    Checks if Ollama is running on the dedicated port with hardware-optimized flags.
    If down, spawns a fully detached background Ollama daemon.
    """
    base_url = f"http://127.0.0.1:{port}"
    if is_ollama_running(base_url):
        return True
    if port != 11435 and is_ollama_running("http://127.0.0.1:11435"):
        return True

    ollama_bin = shutil.which("ollama") or "/usr/local/bin/ollama"
    if not os.path.exists(ollama_bin):
        logger.error(f"Ollama binary not found at '{ollama_bin}'. Cannot auto-start Ollama.")
        return False

    logger.info(
        f"Starting hardware-optimized Ollama daemon on {base_url} (FlashAttention=0, FP16 KV cache)..."
    )

    env = os.environ.copy()
    env.update(
        {
            "OLLAMA_HOST": f"127.0.0.1:{port}",
            "OLLAMA_MODELS": models_dir,
            "OLLAMA_FLASH_ATTENTION": "0",  # Disable unsupported FlashAttention on GTX 1650
            "OLLAMA_KV_CACHE_TYPE": "f16",  # Native FP16 on CUDA without CPU dequantization loop
            "OLLAMA_NUM_PARALLEL": "1",
        }
    )

    try:
        # start_new_session=True creates a new session/process group (setsid)
        # so the daemon process outlives the launcher / terminal.
        subprocess.Popen(
            [ollama_bin, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        logger.error(f"Failed to spawn Ollama daemon: {e}")
        return False

    # Poll until ready
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_ollama_running(base_url):
            logger.info(f"Ollama daemon is ready on {base_url}!")
            return True
        time.sleep(0.5)

    logger.warning(f"Ollama daemon did not respond within {timeout}s on {base_url}.")
    return False


def ensure_all_infrastructure(verbose: bool = True) -> dict[str, Any]:
    """
    Comprehensive entrypoint to check and auto-boot all backing services.
    Ensures:
      1. Ollama on port 11435 with native FP16 and FlashAttention=0.
      2. Elasticsearch container (port 9200).
      3. SearXNG container (port 8080).

    Returns a status dictionary suitable for API responses and CLI banners.
    """
    t0 = time.time()
    if verbose:
        print("\n🔍 Checking WorkOS Infrastructure Services...")

    # 1. Ollama
    ollama_ok = ensure_ollama(port=11434)
    active_ollama_url = (
        "http://127.0.0.1:11434"
        if is_ollama_running("http://127.0.0.1:11434")
        else (
            "http://127.0.0.1:11435"
            if is_ollama_running("http://127.0.0.1:11435")
            else "http://127.0.0.1:11434"
        )
    )
    if verbose:
        status_symbol = "✔" if ollama_ok else "✖"
        print(
            f"  [{status_symbol}] Ollama Inference Engine ({active_ollama_url}) — {'Online (12K FP16 Safe)' if ollama_ok else 'Failed/Offline'}"
        )

    # 2. Docker Containers (Elasticsearch & SearXNG)
    docker_status = ensure_docker_containers()
    es_ok = docker_status.get("elasticsearch", False)
    searx_ok = docker_status.get("searxng", False)

    if verbose:
        status_es = "✔" if es_ok else "✖"
        status_searx = "✔" if searx_ok else "✖"
        print(
            f"  [{status_es}] Elasticsearch Hybrid RAG (localhost:9200) — {'Online' if es_ok else 'Offline'}"
        )
        print(
            f"  [{status_searx}] SearXNG Local Metasearch (localhost:8080) — {'Online' if searx_ok else 'Offline'}"
        )
        duration = time.time() - t0
        print(f"✦ Infrastructure check completed in {duration:.2f}s\n")

    all_ready = ollama_ok and es_ok and searx_ok
    return {
        "status": "ready"
        if all_ready
        else ("partial" if (ollama_ok or es_ok or searx_ok) else "offline"),
        "ready": all_ready,
        "services": {
            "ollama": {
                "name": "Ollama LLM Engine",
                "url": active_ollama_url,
                "online": ollama_ok,
                "notes": "12,288 Context (FP16, GTX 1650 Safe)",
            },
            "elasticsearch": {
                "name": "Elasticsearch RAG",
                "url": "http://localhost:9200",
                "online": es_ok,
                "notes": "Hybrid BM25 + Dense Vectors",
            },
            "searxng": {
                "name": "SearXNG Web Metasearch",
                "url": "http://localhost:8080",
                "online": searx_ok,
                "notes": "Zero-Cloud Private Search",
            },
        },
    }


def get_infrastructure_status() -> dict[str, Any]:
    """Lightweight health probe without triggering auto-start commands."""
    ollama_ok = is_ollama_running("http://127.0.0.1:11435")
    es_ok = is_elasticsearch_running("http://localhost:9200")
    searx_ok = is_searxng_running("http://localhost:8080")
    docker_ok = is_docker_available()

    all_ready = ollama_ok and es_ok and searx_ok
    return {
        "status": "ready"
        if all_ready
        else ("partial" if (ollama_ok or es_ok or searx_ok) else "offline"),
        "ready": all_ready,
        "docker_daemon": docker_ok,
        "services": {
            "ollama": {
                "name": "Ollama LLM Engine",
                "url": "http://127.0.0.1:11435",
                "online": ollama_ok,
                "notes": "12,288 Context (FP16, GTX 1650 Safe)",
            },
            "elasticsearch": {
                "name": "Elasticsearch RAG",
                "url": "http://localhost:9200",
                "online": es_ok,
                "notes": "Hybrid BM25 + Dense Vectors",
            },
            "searxng": {
                "name": "SearXNG Web Metasearch",
                "url": "http://localhost:8080",
                "online": searx_ok,
                "notes": "Zero-Cloud Private Search",
            },
        },
    }


if __name__ == "__main__":
    import json

    result = ensure_all_infrastructure(verbose=True)
    print(json.dumps(result, indent=2))
