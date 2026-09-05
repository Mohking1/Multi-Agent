"""Root pytest configuration and global isolation fixtures for WorkOS tests.

Guarantees that tests never touch or mutate the production memory database (workos_memory.db)
or production document vault (data/vault).
"""

import os
import shutil
import tempfile
import pytest

from config import get_config


@pytest.fixture(autouse=True, scope="session")
def isolate_test_environment(tmp_path_factory):
    """Session-wide isolation fixture creating a dedicated ephemeral sandbox."""
    test_sandbox = tmp_path_factory.mktemp("workos_isolated_test_env")
    test_db = str(test_sandbox / "isolated_test_memory.db")
    test_vault = str(test_sandbox / "isolated_test_vault")

    os.environ["WORKOS_TEST_MODE"] = "1"
    os.environ["WORKOS_MEMORY_DB"] = test_db
    os.environ["WORKOS_VAULT_DIR"] = test_vault

    # Reload global configuration instance with test paths
    cfg = get_config(reload=True)

    # If UI app module is already imported, update its instances
    try:
        import workos_engine.ui.app as ui_app

        ui_app.config = cfg
        ui_app.vault = None
        ui_app.planner = ui_app.ExecutivePlanner(config=cfg)
        ui_app.memory = ui_app.planner.memory
    except Exception:
        pass

    yield

    shutil.rmtree(str(test_sandbox), ignore_errors=True)
