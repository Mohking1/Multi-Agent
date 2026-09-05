"""Tests verifying strict memory and vault isolation in test runs."""

import json
import os
import sqlite3
from pathlib import Path

from config import get_config, is_test_environment
from workos_engine.memory.db import MemoryDB
from workos_engine.memory.networks import CognitiveMemoryEngine
from workos_engine.tools.doc_tools import DocToolKit
from workos_engine.types import MemoryItem, MemoryNetwork
from workos_engine.vault import DocumentVault


def test_environment_detected_as_test():
    """Verify that is_test_environment detects test mode."""
    assert is_test_environment() is True


def test_config_paths_are_isolated():
    """Verify that default config points to ephemeral test directories."""
    cfg = get_config()
    assert cfg.memory_db_path != "workos_memory.db"
    assert "test" in cfg.memory_db_path.lower() or "/tmp" in cfg.memory_db_path

    assert str(cfg.vault_dir) != "data/vault"
    assert "test" in str(cfg.vault_dir).lower() or "/tmp" in str(cfg.vault_dir)


def test_memory_db_safeguard_prevents_production_pollution():
    """Verify that even explicit attempts to open workos_memory.db in test mode redirect."""
    # Attempt to open production DB
    db = MemoryDB("workos_memory.db")
    assert db.db_path != os.path.abspath("workos_memory.db")

    # Write a test memory item
    item = MemoryItem(
        id="mem_isolation_test",
        network=MemoryNetwork.BELIEFS,
        wing="test_wing",
        hall="test_hall",
        key="test_key",
        content="This must never appear in production memory",
    )
    db.insert_memory(item)

    # Verify production DB is completely untouched
    prod_conn = sqlite3.connect("workos_memory.db")
    cur = prod_conn.cursor()
    cur.execute("SELECT count(*) FROM memories WHERE id = 'mem_isolation_test'")
    count = cur.fetchone()[0]
    prod_conn.close()
    assert count == 0


def test_vault_safeguard_prevents_production_pollution(tmp_path):
    """Verify that DocumentVault and DocToolKit default instances never write to data/vault."""
    vault = DocumentVault()
    assert vault.vault_dir != Path("data/vault").resolve()

    # Store a dummy document
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("Test isolation dummy")
    doc = vault.store_document(dummy)

    # Check production vault catalog
    with open("data/vault/catalog.json") as f:
        prod_catalog = json.load(f)
    assert doc.doc_id not in prod_catalog

    # DocToolKit default instantiation
    toolkit = DocToolKit()
    assert toolkit.vault.vault_dir != Path("data/vault").resolve()
