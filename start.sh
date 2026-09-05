#!/usr/bin/env bash
# ==============================================================================
# WorkOS Complete Executive AI Operating System Launcher
#
# Automatically ensures backing infrastructure is up (Ollama 12K FP16,
# Elasticsearch RAG, SearXNG Metasearch) and starts the Web UI on port 8000.
# ==============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Dynamic Python environment detection
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$HOME/miniconda3/envs/Multi_Agent/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda3/envs/Multi_Agent/bin/python"
elif [ -x "$HOME/miniconda3/envs/Multi_Agent/bin/python3.12" ]; then
    PYTHON_BIN="$HOME/miniconda3/envs/Multi_Agent/bin/python3.12"
elif [ -x "$HOME/anaconda3/envs/Multi_Agent/bin/python" ]; then
    PYTHON_BIN="$HOME/anaconda3/envs/Multi_Agent/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Error: Python 3 not found!"
    exit 1
fi

echo "============================================================"
echo "✦ WorkOS — Autonomous Executive AI Operating System"
echo "✦ Auto-Booting Infrastructure (Ollama, Elasticsearch, SearXNG)..."
echo "============================================================"

# Reclaim port 8000 if previously bound or stopped
fuser -k 8000/tcp 2>/dev/null || true

# Launch WorkOS UI (ui.py verifies & boots Ollama, Elasticsearch, SearXNG automatically)
exec "$PYTHON_BIN" ui.py
