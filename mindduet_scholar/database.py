"""SQLite schema and connection handling for personal learning records."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    file_type TEXT NOT NULL,
    title TEXT NOT NULL,
    headings_json TEXT NOT NULL,
    links_json TEXT NOT NULL,
    math_blocks INTEGER NOT NULL DEFAULT 0,
    content_text TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_title_idx ON documents(title);

CREATE TABLE IF NOT EXISTS learning_units (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    source_anchor TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    goal TEXT NOT NULL,
    learning_unit_id INTEGER REFERENCES learning_units(id) ON DELETE SET NULL,
    session_type TEXT NOT NULL DEFAULT 'practice' CHECK(session_type IN ('practice', 'inquiry')),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'completed', 'abandoned')),
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS sessions_status_idx ON sessions(status, created_at DESC);

CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS session_messages_session_idx
    ON session_messages(session_id, created_at, id);

CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    body TEXT NOT NULL,
    revision_of_id INTEGER REFERENCES attempts(id) ON DELETE SET NULL,
    diagnosis TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_session_idx ON attempts(session_id, created_at);

CREATE TABLE IF NOT EXISTS hint_requests (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    attempt_id INTEGER REFERENCES attempts(id) ON DELETE SET NULL,
    tier INTEGER NOT NULL CHECK(tier BETWEEN 1 AND 3),
    body TEXT NOT NULL,
    provider TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS misconceptions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempt_misconceptions (
    attempt_id INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
    misconception_id INTEGER NOT NULL REFERENCES misconceptions(id) ON DELETE CASCADE,
    PRIMARY KEY (attempt_id, misconception_id)
);

CREATE TABLE IF NOT EXISTS mastery (
    id INTEGER PRIMARY KEY,
    concept TEXT NOT NULL UNIQUE,
    confidence REAL NOT NULL DEFAULT 0.0 CHECK(confidence BETWEEN 0.0 AND 1.0),
    evidence_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    concept TEXT NOT NULL,
    due_on TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled' CHECK(status IN ('scheduled', 'completed', 'skipped')),
    evidence TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS reviews_due_idx ON reviews(status, due_on);

CREATE TABLE IF NOT EXISTS codex_conversations (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'archived')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS codex_conversations_active_document_idx
    ON codex_conversations(document_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS codex_messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES codex_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS codex_messages_conversation_idx
    ON codex_messages(conversation_id, created_at, id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            session_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "session_type" not in session_columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN session_type TEXT NOT NULL DEFAULT 'practice'"
                )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
