import sqlite3
from contextlib import contextmanager

from .config import SETTINGS


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    modality TEXT NOT NULL,
    status TEXT NOT NULL,
    queue_name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    input_path TEXT,
    input_text TEXT,
    input_filename TEXT,
    input_content_type TEXT,
    input_size_bytes INTEGER,
    result_json TEXT,
    error_message TEXT,
    task_id TEXT,
    idempotency_key TEXT
);
"""

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL",
)

MISSING_COLUMNS = {
    "queue_name": "TEXT",
    "started_at": "TEXT",
    "completed_at": "TEXT",
    "attempts": "INTEGER NOT NULL DEFAULT 0",
    "input_filename": "TEXT",
    "input_content_type": "TEXT",
    "input_size_bytes": "INTEGER",
    "task_id": "TEXT",
    "idempotency_key": "TEXT",
}


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")


def _ensure_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(jobs)").fetchall()
    existing = {str(row[1]) for row in rows}
    for name, ddl in MISSING_COLUMNS.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}")


def init_db() -> None:
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SETTINGS.database_path, timeout=30)
    try:
        _apply_pragmas(conn)
        conn.execute(SCHEMA)
        _ensure_columns(conn)
        for statement in INDEXES:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_conn():
    conn = sqlite3.connect(SETTINGS.database_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        _apply_pragmas(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()
