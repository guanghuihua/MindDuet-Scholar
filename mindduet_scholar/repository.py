"""Persistence operations for the learning workflow."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .database import Database
from .types import IndexedDocument, Review


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, database: Database):
        self.database = database

    def replace_documents(self, documents: list[IndexedDocument]) -> None:
        indexed_at = _now()
        paths = [document.relative_path for document in documents]
        with self.database.connection() as connection:
            for document in documents:
                connection.execute(
                    """
                    INSERT INTO documents (
                        relative_path, file_type, title, headings_json, links_json,
                        math_blocks, content_text, content_hash, modified_at, byte_size, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(relative_path) DO UPDATE SET
                        file_type=excluded.file_type, title=excluded.title,
                        headings_json=excluded.headings_json, links_json=excluded.links_json,
                        math_blocks=excluded.math_blocks, content_text=excluded.content_text,
                        content_hash=excluded.content_hash, modified_at=excluded.modified_at,
                        byte_size=excluded.byte_size, indexed_at=excluded.indexed_at
                    """,
                    (
                        document.relative_path,
                        document.file_type,
                        document.title,
                        json.dumps(document.headings, ensure_ascii=False),
                        json.dumps(document.links, ensure_ascii=False),
                        document.math_blocks,
                        document.text,
                        document.content_hash,
                        document.modified_at.isoformat(),
                        document.byte_size,
                        indexed_at,
                    ),
                )
            if paths:
                placeholders = ", ".join("?" for _ in paths)
                connection.execute(f"DELETE FROM documents WHERE relative_path NOT IN ({placeholders})", paths)
            else:
                connection.execute("DELETE FROM documents")

    def search_documents(self, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            if query.strip():
                term = f"%{query.strip()}%"
                rows = connection.execute(
                    """SELECT id, relative_path, file_type, title, headings_json, math_blocks, indexed_at
                    FROM documents WHERE title LIKE ? COLLATE NOCASE
                    OR relative_path LIKE ? COLLATE NOCASE OR content_text LIKE ? COLLATE NOCASE
                    ORDER BY title LIMIT ?""",
                    (term, term, term, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT id, relative_path, file_type, title, headings_json, math_blocks, indexed_at
                    FROM documents ORDER BY title LIMIT ?""",
                    (limit,),
                ).fetchall()
        return [self._document_row(row) for row in rows]

    def documents_under(self, path_prefix: str) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """SELECT id, relative_path, file_type, title, headings_json, math_blocks, indexed_at
                FROM documents WHERE relative_path LIKE ? ORDER BY relative_path""",
                (f"{path_prefix}%",),
            ).fetchall()
        return [self._document_row(row) for row in rows]

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._document_row(row, include_text=True) if row else None

    @staticmethod
    def _document_row(row: Any, include_text: bool = False) -> dict[str, Any]:
        document = dict(row)
        document["headings"] = json.loads(document.pop("headings_json"))
        document.pop("links_json", None)
        if not include_text:
            document.pop("content_text", None)
        return document

    def start_session(
        self,
        goal: str,
        document_id: int | None = None,
        source_anchor: str | None = None,
        session_type: str = "practice",
    ) -> int:
        if session_type not in {"practice", "inquiry"}:
            raise ValueError(f"Unsupported session type: {session_type}")
        now = _now()
        with self.database.connection() as connection:
            unit = connection.execute(
                "INSERT INTO learning_units (title, source_document_id, source_anchor, created_at) VALUES (?, ?, ?, ?)",
                (goal, document_id, source_anchor, now),
            )
            session = connection.execute(
                "INSERT INTO sessions (goal, learning_unit_id, session_type, created_at) VALUES (?, ?, ?, ?)",
                (goal, unit.lastrowid, session_type, now),
            )
            return int(session.lastrowid)

    def list_sessions(self, active_only: bool = False) -> list[dict[str, Any]]:
        statement = """
            SELECT sessions.*, documents.title AS source_title
            FROM sessions
            LEFT JOIN learning_units ON learning_units.id = sessions.learning_unit_id
            LEFT JOIN documents ON documents.id = learning_units.source_document_id
        """
        if active_only:
            statement += " WHERE sessions.status = 'active'"
        statement += " ORDER BY sessions.created_at DESC"
        with self.database.connection() as connection:
            return [dict(row) for row in connection.execute(statement).fetchall()]

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """SELECT sessions.*, documents.id AS source_document_id, documents.title AS source_title,
                    documents.relative_path AS source_path
                    FROM sessions
                    LEFT JOIN learning_units ON learning_units.id = sessions.learning_unit_id
                    LEFT JOIN documents ON documents.id = learning_units.source_document_id
                    WHERE sessions.id = ?""",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            session = dict(row)
            session["attempts"] = [dict(item) for item in connection.execute(
                "SELECT * FROM attempts WHERE session_id = ? ORDER BY created_at", (session_id,)
            ).fetchall()]
            session["hints"] = [dict(item) for item in connection.execute(
                "SELECT * FROM hint_requests WHERE session_id = ? ORDER BY created_at", (session_id,)
            ).fetchall()]
            session["messages"] = [dict(item) for item in connection.execute(
                "SELECT id, role, body, created_at FROM session_messages WHERE session_id = ? ORDER BY created_at, id",
                (session_id,),
            ).fetchall()]
            return session

    def set_session_type(self, session_id: int, session_type: str) -> None:
        if session_type not in {"practice", "inquiry"}:
            raise ValueError(f"Unsupported session type: {session_type}")
        with self.database.connection() as connection:
            connection.execute("UPDATE sessions SET session_type = ? WHERE id = ?", (session_type, session_id))

    def add_session_message(self, session_id: int, role: str, body: str) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported session message role: {role}")
        with self.database.connection() as connection:
            cursor = connection.execute(
                "INSERT INTO session_messages (session_id, role, body, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, body, _now()),
            )
            return int(cursor.lastrowid)

    def add_attempt(self, session_id: int, body: str, diagnosis: str | None, misconceptions: list[tuple[str, str]]) -> int:
        now = _now()
        with self.database.connection() as connection:
            attempt = connection.execute(
                "INSERT INTO attempts (session_id, body, diagnosis, created_at) VALUES (?, ?, ?, ?)",
                (session_id, body, diagnosis, now),
            )
            attempt_id = int(attempt.lastrowid)
            for name, description in misconceptions:
                connection.execute(
                    """INSERT INTO misconceptions (name, description, occurrences, last_seen_at)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(name) DO UPDATE SET occurrences = occurrences + 1, last_seen_at = excluded.last_seen_at""",
                    (name, description, now),
                )
                misconception = connection.execute("SELECT id FROM misconceptions WHERE name = ?", (name,)).fetchone()
                connection.execute(
                    "INSERT OR IGNORE INTO attempt_misconceptions (attempt_id, misconception_id) VALUES (?, ?)",
                    (attempt_id, misconception["id"]),
                )
            goal = connection.execute("SELECT goal FROM sessions WHERE id = ?", (session_id,)).fetchone()["goal"]
            confidence = 0.2 if misconceptions else 0.35
            connection.execute(
                """INSERT INTO mastery (concept, confidence, evidence_count, updated_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(concept) DO UPDATE SET
                    confidence = MIN(1.0, MAX(0.0, (mastery.confidence * mastery.evidence_count + excluded.confidence) / (mastery.evidence_count + 1))),
                    evidence_count = mastery.evidence_count + 1, updated_at = excluded.updated_at""",
                (goal, confidence, now),
            )
            return attempt_id

    def add_hint(self, session_id: int, attempt_id: int | None, body: str, provider: str) -> int:
        with self.database.connection() as connection:
            cursor = connection.execute(
                "INSERT INTO hint_requests (session_id, attempt_id, tier, body, provider, created_at) VALUES (?, ?, 1, ?, ?, ?)",
                (session_id, attempt_id, body, provider, _now()),
            )
            return int(cursor.lastrowid)

    def schedule_reviews(self, concept: str) -> int:
        now = _now()
        scheduled = 0
        with self.database.connection() as connection:
            for offset in (1, 7):
                due_on = (date.today() + timedelta(days=offset)).isoformat()
                existing = connection.execute(
                    "SELECT id FROM reviews WHERE concept = ? AND due_on = ? AND status = 'scheduled'", (concept, due_on)
                ).fetchone()
                if not existing:
                    connection.execute(
                        "INSERT INTO reviews (concept, due_on, created_at) VALUES (?, ?, ?)", (concept, due_on, now)
                    )
                    scheduled += 1
        return scheduled

    def due_reviews(self, include_future: bool = False) -> list[Review]:
        statement = "SELECT * FROM reviews WHERE status = 'scheduled'"
        values: tuple[str, ...] = ()
        if not include_future:
            statement += " AND due_on <= ?"
            values = (date.today().isoformat(),)
        statement += " ORDER BY due_on, concept"
        with self.database.connection() as connection:
            rows = connection.execute(statement, values).fetchall()
        return [Review(row["id"], row["concept"], date.fromisoformat(row["due_on"]), row["status"], row["evidence"]) for row in rows]

    def complete_review(self, review_id: int, evidence: str) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE reviews SET status = 'completed', evidence = ?, completed_at = ? WHERE id = ?",
                (evidence, _now(), review_id),
            )

    def complete_session(self, session_id: int) -> None:
        with self.database.connection() as connection:
            connection.execute("UPDATE sessions SET status = 'completed', completed_at = ? WHERE id = ?", (_now(), session_id))

    def reopen_session(self, session_id: int) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET status = 'active', completed_at = NULL WHERE id = ? AND status = 'completed'",
                (session_id,),
            )
            return cursor.rowcount > 0

    def active_codex_conversation(self, document_id: int) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """SELECT * FROM codex_conversations
                WHERE document_id = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1""",
                (document_id,),
            ).fetchone()
            if not row:
                return None
            conversation = dict(row)
            conversation["messages"] = [
                dict(message)
                for message in connection.execute(
                    """SELECT id, role, body, created_at FROM codex_messages
                    WHERE conversation_id = ? ORDER BY created_at, id""",
                    (row["id"],),
                ).fetchall()
            ]
            return conversation

    def list_codex_conversations(self) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """SELECT codex_conversations.*, documents.title, documents.relative_path
                FROM codex_conversations
                JOIN documents ON documents.id = codex_conversations.document_id
                ORDER BY codex_conversations.created_at"""
            ).fetchall()
        return [dict(row) for row in rows]

    def update_codex_thread_id(self, conversation_id: int, thread_id: str) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE codex_conversations SET thread_id = ?, updated_at = ? WHERE id = ?",
                (thread_id, _now(), conversation_id),
            )

    def create_codex_conversation(self, document_id: int, thread_id: str) -> dict[str, Any]:
        now = _now()
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE codex_conversations SET status = 'archived', updated_at = ? WHERE document_id = ? AND status = 'active'",
                (now, document_id),
            )
            cursor = connection.execute(
                """INSERT INTO codex_conversations (document_id, thread_id, status, created_at, updated_at)
                VALUES (?, ?, 'active', ?, ?)""",
                (document_id, thread_id, now, now),
            )
            return {
                "id": int(cursor.lastrowid),
                "document_id": document_id,
                "thread_id": thread_id,
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }

    def add_codex_message(self, conversation_id: int, role: str, body: str) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported Codex message role: {role}")
        now = _now()
        with self.database.connection() as connection:
            cursor = connection.execute(
                "INSERT INTO codex_messages (conversation_id, role, body, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, body, now),
            )
            connection.execute(
                "UPDATE codex_conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            return int(cursor.lastrowid)

    def archive_codex_conversation(self, document_id: int) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """UPDATE codex_conversations SET status = 'archived', updated_at = ?
                WHERE document_id = ? AND status = 'active'""",
                (_now(), document_id),
            )
            return cursor.rowcount > 0

    def dashboard(self) -> dict[str, Any]:
        with self.database.connection() as connection:
            return {
                "documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "active_sessions": connection.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'").fetchone()[0],
                "due_reviews": connection.execute(
                    "SELECT COUNT(*) FROM reviews WHERE status = 'scheduled' AND due_on <= ?", (date.today().isoformat(),)
                ).fetchone()[0],
                "recent_sessions": [dict(row) for row in connection.execute(
                    "SELECT id, goal, status, created_at FROM sessions ORDER BY created_at DESC LIMIT 5"
                ).fetchall()],
            }

    def report(self) -> dict[str, list[dict[str, Any]]]:
        with self.database.connection() as connection:
            return {
                "misconceptions": [dict(row) for row in connection.execute(
                    "SELECT * FROM misconceptions ORDER BY occurrences DESC, last_seen_at DESC"
                ).fetchall()],
                "mastery": [dict(row) for row in connection.execute(
                    "SELECT * FROM mastery ORDER BY updated_at DESC"
                ).fetchall()],
                "reviews": [
                    {"id": review.id, "concept": review.concept, "due_on": review.due_on.isoformat(), "status": review.status}
                    for review in self.due_reviews(include_future=True)
                ],
            }
