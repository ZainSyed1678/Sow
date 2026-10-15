"""Local execution state and deduplication store using Python's built-in sqlite3."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_ml_notion_researcher.logging_utils import get_logger

logger = get_logger("storage.sqlite")


class SQLiteStore:
    """Thread-safe SQLite storage for tracking processed items and local run state.

    # ponytail: stdlib sqlite3 used directly without heavy ORM packages.
    """

    def __init__(self, db_path: str | Path = "local_state/researcher.db"):
        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_items (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    target_database TEXT,
                    processed_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_canonical_url
                ON processed_items (canonical_url)
            """)
            conn.commit()

    def is_processed(self, item_id: str) -> bool:
        """Checks if an item ID was already processed in local storage."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM processed_items WHERE id = ? LIMIT 1",
                (item_id,),
            )
            return cursor.fetchone() is not None

    def mark_processed(
        self,
        item_id: str,
        source_type: str,
        canonical_url: str,
        title: str,
        target_database: str,
        status: str = "published",
    ) -> None:
        """Records an item as processed."""
        now = datetime.now(UTC).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_items
                (id, source_type, canonical_url, title, target_database, processed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (item_id, source_type, canonical_url, title, target_database, now, status),
            )
            conn.commit()

    def filter_unprocessed(self, items: list[Any]) -> list[Any]:
        """Returns only items that have not yet been processed."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id FROM processed_items")
            processed_ids = {row["id"] for row in cursor.fetchall()}

        return [item for item in items if item.id not in processed_ids]

    def get_total_processed_count(self) -> int:
        """Returns the total number of items recorded in storage."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) AS cnt FROM processed_items")
            row = cursor.fetchone()
            return row["cnt"] if row else 0
