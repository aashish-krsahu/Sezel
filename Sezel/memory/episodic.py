"""Episodic memory: persistent SQLite store for cross-session persistence."""

import sqlite3
import asyncio
from pathlib import Path
from ..core.type import Turn, Affect


class EpisodicStore:
    """SQLite-backed persistent episodic memory store."""

    def __init__(self, path: str | Path = "sezel.db"):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def _ensure_connection(self) -> sqlite3.Connection:
        """Ensure connection is open."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_db()
        return self._conn

    def _init_db(self) -> None:
        """Initialize the database schema if needed."""
        conn = self._ensure_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                valence REAL DEFAULT 0.0,
                arousal REAL DEFAULT 0.0,
                dominance REAL DEFAULT 0.0,
                salience REAL DEFAULT 0.0
            )
        """)
        conn.commit()

    async def write(self, turn: Turn, affect: Affect | None = None) -> None:
        """Write a turn to episodic memory."""
        if affect is None:
            affect = Affect()

        conn = self._ensure_connection()

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        def _write():
            conn.execute(
                """INSERT INTO episodes 
                   (ts, role, content, valence, arousal, dominance) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (turn.ts, turn.role, turn.content,
                 affect.valence, affect.arousal, affect.dominance)
            )
            conn.commit()

        await loop.run_in_executor(None, _write)

    async def recent(self, n: int = 10) -> list[Turn]:
        """Retrieve the last n turns from episodic memory."""
        conn = self._ensure_connection()

        loop = asyncio.get_event_loop()
        def _read():
            cursor = conn.execute(
                """SELECT ts, role, content FROM episodes 
                   ORDER BY id DESC LIMIT ?""",
                (n,)
            )
            rows = cursor.fetchall()
            return rows

        rows = await loop.run_in_executor(None, _read)
        # Reverse to restore chronological order, newest last
        return [Turn(role=r[1], content=r[2], ts=r[0]) for r in reversed(rows)]

    async def all(self, limit: int = 1000) -> list[Turn]:
        """Retrieve all turns from episodic memory."""
        conn = self._ensure_connection()

        loop = asyncio.get_event_loop()
        def _read():
            cursor = conn.execute(
                """SELECT ts, role, content FROM episodes 
                   ORDER BY id ASC LIMIT ?""",
                (limit,)
            )
            return cursor.fetchall()

        rows = await loop.run_in_executor(None, _read)
        return [Turn(role=r[1], content=r[2], ts=r[0]) for r in rows]

    async def count(self) -> int:
        """Get the total number of episodes stored."""
        conn = self._ensure_connection()

        loop = asyncio.get_event_loop()
        def _count():
            cursor = conn.execute("SELECT COUNT(*) FROM episodes")
            return cursor.fetchone()[0]

        return await loop.run_in_executor(None, _count)

    async def clear(self) -> None:
        """Clear all episodes from the database."""
        conn = self._ensure_connection()

        loop = asyncio.get_event_loop()
        def _clear():
            conn.execute("DELETE FROM episodes")
            conn.commit()

        await loop.run_in_executor(None, _clear)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            loop = asyncio.get_event_loop()
            def _close():
                self._conn.close()
            await loop.run_in_executor(None, _close)
            self._conn = None

