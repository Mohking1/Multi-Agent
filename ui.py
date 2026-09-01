#!/usr/bin/env python
"""WorkOS Web UI Launcher."""

import uvicorn

from workos_engine.ui.app import app

if __name__ == "__main__":
    print("✦ Starting WorkOS Minimalist Web UI on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
