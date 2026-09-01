import json
import os
import re
import sqlite3
import time
from typing import Any

from workos_engine.types import MemoryItem, MemoryNetwork


class MemoryDB:
    """
    SQLite + FTS5 backed persistent storage for the 4-network cognitive memory engine.
    Supports BM25 ranking, spatial Loci queries, and belief superseding.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            # Enable WAL mode and foreign keys if file-backed
            if self.db_path != ":memory:":
                self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA foreign_keys = ON;")

            # 1. Main memories table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    network TEXT NOT NULL,
                    wing TEXT NOT NULL,
                    hall TEXT NOT NULL,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}',
                    superseded_by TEXT DEFAULT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

            # 2. Virtual FTS5 table for BM25 ranking
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    id UNINDEXED,
                    content,
                    key,
                    wing,
                    hall,
                    network,
                    tokenize = 'unicode61'
                );
                """
            )

            # 3. Triggers to keep memories_fts synchronized
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_memories_insert AFTER INSERT ON memories
                BEGIN
                    INSERT INTO memories_fts(id, content, key, wing, hall, network)
                    VALUES (new.id, new.content, new.key, new.wing, new.hall, new.network);
                END;
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_memories_delete AFTER DELETE ON memories
                BEGIN
                    DELETE FROM memories_fts WHERE id = old.id;
                END;
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_memories_update AFTER UPDATE ON memories
                BEGIN
                    DELETE FROM memories_fts WHERE id = old.id;
                    INSERT INTO memories_fts(id, content, key, wing, hall, network)
                    VALUES (new.id, new.content, new.key, new.wing, new.hall, new.network);
                END;
                """
            )

            # 4. Entities table for structured entity resolution
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    entity_name TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    attributes TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

            # 5. Associative links table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS links (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY (source_id, target_id, relation_type)
                );
                """
            )

            # 6. Performance indices
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_network ON memories(network);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_locus ON memories(wing, hall);"
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded_by);"
            )

    def _row_to_memory_item(self, row: sqlite3.Row) -> MemoryItem:
        raw_meta = row["metadata"]
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
        return MemoryItem(
            id=row["id"],
            network=MemoryNetwork(row["network"]),
            wing=row["wing"],
            hall=row["hall"],
            key=row["key"],
            content=row["content"],
            confidence=float(row["confidence"]),
            metadata=meta,
            superseded_by=row["superseded_by"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def insert_memory(self, item: MemoryItem) -> str:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, network, wing, hall, key, content, confidence,
                    metadata, superseded_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    item.id,
                    item.network.value
                    if isinstance(item.network, MemoryNetwork)
                    else str(item.network),
                    item.wing,
                    item.hall,
                    item.key,
                    item.content,
                    item.confidence,
                    json.dumps(item.metadata or {}),
                    item.superseded_by,
                    item.created_at or time.time(),
                    item.updated_at or time.time(),
                ),
            )
        return item.id

    def get_memory(self, item_id: str) -> MemoryItem | None:
        cursor = self.conn.execute("SELECT * FROM memories WHERE id = ?;", (item_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_memory_item(row)
        return None

    def supersede_key(
        self,
        key: str,
        wing: str | None = None,
        hall: str | None = None,
        new_id: str | None = None,
    ) -> int:
        """
        Marks all existing active memories with matching key (and optionally wing/hall)
        as superseded by new_id.
        """
        query = "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE key = ? AND superseded_by IS NULL"
        params: list[Any] = [new_id, time.time(), key]
        if wing:
            query += " AND wing = ?"
            params.append(wing)
        if hall:
            query += " AND hall = ?"
            params.append(hall)

        with self.conn:
            cursor = self.conn.execute(query, params)
            return cursor.rowcount

    def search_fts(
        self,
        query: str,
        network: str | MemoryNetwork | None = None,
        wing: str | None = None,
        hall: str | None = None,
        include_superseded: bool = False,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """
        Performs full-text search with BM25 ranking across memories.
        Safely formats query terms and falls back to LIKE search if FTS5 syntax fails.
        """
        net_val = (
            network.value
            if isinstance(network, MemoryNetwork)
            else (str(network) if network else None)
        )

        # Sanitize query words for FTS5
        words = re.findall(r"\w+", query)
        if not words:
            # Empty query -> return recent items
            return self.query_memories(
                network=net_val,
                wing=wing,
                hall=hall,
                active_only=not include_superseded,
                limit=limit,
            )

        fts_query = " OR ".join(f'"{w}"*' for w in words)

        sql = """
            SELECT m.*, bm25(memories_fts) as rank
            FROM memories_fts f
            JOIN memories m ON f.id = m.id
            WHERE memories_fts MATCH ?
        """
        params: list[Any] = [fts_query]

        if net_val:
            sql += " AND m.network = ?"
            params.append(net_val)
        if wing:
            sql += " AND m.wing = ?"
            params.append(wing)
        if hall:
            sql += " AND m.hall = ?"
            params.append(hall)
        if not include_superseded:
            sql += " AND m.superseded_by IS NULL"

        sql += " ORDER BY rank ASC LIMIT ?"
        params.append(limit)

        try:
            cursor = self.conn.execute(sql, params)
            rows = cursor.fetchall()
            return [self._row_to_memory_item(r) for r in rows]
        except sqlite3.OperationalError:
            # Fallback to LIKE search
            like_sql = """
                SELECT * FROM memories
                WHERE (content LIKE ? OR key LIKE ?)
            """
            like_pat = f"%{query.strip()}%"
            like_params: list[Any] = [like_pat, like_pat]
            if net_val:
                like_sql += " AND network = ?"
                like_params.append(net_val)
            if wing:
                like_sql += " AND wing = ?"
                like_params.append(wing)
            if hall:
                like_sql += " AND hall = ?"
                like_params.append(hall)
            if not include_superseded:
                like_sql += " AND superseded_by IS NULL"
            like_sql += " ORDER BY updated_at DESC LIMIT ?"
            like_params.append(limit)

            cursor = self.conn.execute(like_sql, like_params)
            rows = cursor.fetchall()
            return [self._row_to_memory_item(r) for r in rows]

    def query_memories(
        self,
        network: str | MemoryNetwork | None = None,
        wing: str | None = None,
        hall: str | None = None,
        key: str | None = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[MemoryItem]:
        net_val = (
            network.value
            if isinstance(network, MemoryNetwork)
            else (str(network) if network else None)
        )
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []

        if net_val:
            sql += " AND network = ?"
            params.append(net_val)
        if wing:
            sql += " AND wing = ?"
            params.append(wing)
        if hall:
            sql += " AND hall = ?"
            params.append(hall)
        if key:
            sql += " AND key = ?"
            params.append(key)
        if active_only:
            sql += " AND superseded_by IS NULL"

        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()
        return [self._row_to_memory_item(r) for r in rows]

    def get_active_beliefs(self, wing: str | None = None) -> list[MemoryItem]:
        return self.query_memories(
            network=MemoryNetwork.BELIEFS,
            wing=wing,
            active_only=True,
            limit=100,
        )

    def get_loci_structure(self, active_only: bool = True) -> dict[str, dict[str, list[str]]]:
        """
        Returns mapping of wing -> hall -> [list of active keys].
        """
        sql = "SELECT DISTINCT wing, hall, key FROM memories WHERE 1=1"
        if active_only:
            sql += " AND superseded_by IS NULL"
        sql += " ORDER BY wing ASC, hall ASC, key ASC;"

        cursor = self.conn.execute(sql)
        rows = cursor.fetchall()

        structure: dict[str, dict[str, list[str]]] = {}
        for r in rows:
            w = r["wing"]
            h = r["hall"]
            k = r["key"]
            if w not in structure:
                structure[w] = {}
            if h not in structure[w]:
                structure[w][h] = []
            if k not in structure[w][h]:
                structure[w][h].append(k)

        return structure

    def upsert_entity(
        self,
        entity_name: str,
        entity_type: str,
        summary: str,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO entities (entity_name, entity_type, summary, attributes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_name) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    summary = excluded.summary,
                    attributes = excluded.attributes,
                    updated_at = excluded.updated_at;
                """,
                (
                    entity_name,
                    entity_type,
                    summary,
                    json.dumps(attributes or {}),
                    now,
                    now,
                ),
            )
        return entity_name

    def get_entity(self, entity_name: str) -> dict[str, Any] | None:
        cursor = self.conn.execute("SELECT * FROM entities WHERE entity_name = ?;", (entity_name,))
        row = cursor.fetchone()
        if row:
            raw_attrs = row["attributes"]
            attrs = json.loads(raw_attrs) if isinstance(raw_attrs, str) else (raw_attrs or {})
            return {
                "entity_name": row["entity_name"],
                "entity_type": row["entity_type"],
                "summary": row["summary"],
                "attributes": attrs,
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
        return None

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
