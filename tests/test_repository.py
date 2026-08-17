from __future__ import annotations

from datetime import date, timedelta
import sqlite3

from mindduet_scholar.database import Database
from mindduet_scholar.indexer import NotesIndexer
from mindduet_scholar.repository import Repository


def test_learning_evidence_and_reviews_are_persisted(settings, notes_root) -> None:
    database = Database(settings.database_path)
    database.initialize()
    repository = Repository(database)
    documents, result = NotesIndexer().scan(notes_root)
    assert result.errors == []
    repository.replace_documents(documents)
    source = repository.search_documents("quotient")[0]

    session_id = repository.start_session("Prove the quotient property", source["id"], "Exercise 32.6")
    attempt_id = repository.add_attempt(
        session_id,
        "Clearly, therefore the map works by definition.",
        "A conclusion is asserted without an explicit supporting rule or theorem.",
        [("missing justification", "A conclusion is asserted without an explicit supporting rule or theorem.")],
    )
    repository.add_hint(session_id, attempt_id, "Name the map's defining property first.", "local")
    assert repository.schedule_reviews("Prove the quotient property") == 2

    session = repository.get_session(session_id)
    assert session is not None
    assert len(session["attempts"]) == 1
    assert len(session["hints"]) == 1
    report = repository.report()
    assert report["misconceptions"][0]["name"] == "missing justification"
    assert report["mastery"][0]["evidence_count"] == 1
    assert {item["due_on"] for item in report["reviews"]} == {
        (date.today() + timedelta(days=1)).isoformat(),
        (date.today() + timedelta(days=7)).isoformat(),
    }


def test_codex_conversation_messages_are_persisted_per_document(settings, notes_root) -> None:
    database = Database(settings.database_path)
    database.initialize()
    repository = Repository(database)
    documents, _ = NotesIndexer().scan(notes_root)
    repository.replace_documents(documents)
    source = repository.search_documents("quotient")[0]

    conversation = repository.create_codex_conversation(source["id"], "thread-1")
    repository.add_codex_message(conversation["id"], "user", "解释商群。")
    repository.add_codex_message(conversation["id"], "assistant", "先回忆陪集的定义。")

    restored = repository.active_codex_conversation(source["id"])
    assert restored is not None
    assert restored["thread_id"] == "thread-1"
    assert [message["role"] for message in restored["messages"]] == ["user", "assistant"]

    assert repository.archive_codex_conversation(source["id"]) is True
    assert repository.active_codex_conversation(source["id"]) is None
    replacement = repository.create_codex_conversation(source["id"], "thread-2")
    assert replacement["messages"] == []


def test_inquiry_session_messages_and_type_are_persisted(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    repository = Repository(database)

    session_id = repository.start_session("Understand quotient notation", session_type="inquiry")
    repository.add_session_message(session_id, "user", r"Is \(\mathbb Z_n\) a quotient?")
    repository.add_session_message(session_id, "assistant", r"Yes: \(\mathbb Z/n\mathbb Z\).")

    session = repository.get_session(session_id)
    assert session is not None
    assert session["session_type"] == "inquiry"
    assert [message["role"] for message in session["messages"]] == ["user", "assistant"]

    repository.complete_session(session_id)
    completed = repository.get_session(session_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["completed_at"] is not None

    assert repository.reopen_session(session_id) is True
    reopened = repository.get_session(session_id)
    assert reopened is not None
    assert reopened["status"] == "active"
    assert reopened["completed_at"] is None
    assert [message["body"] for message in reopened["messages"]] == [
        r"Is \(\mathbb Z_n\) a quotient?",
        r"Yes: \(\mathbb Z/n\mathbb Z\).",
    ]
    assert repository.reopen_session(session_id) is False


def test_database_adds_session_type_to_existing_database(settings) -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            """CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                goal TEXT NOT NULL,
                learning_unit_id INTEGER,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                completed_at TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO sessions (goal, created_at) VALUES (?, ?)",
            ("Existing session", "2026-08-17T00:00:00+00:00"),
        )

    Database(settings.database_path).initialize()

    with sqlite3.connect(settings.database_path) as connection:
        row = connection.execute("SELECT session_type FROM sessions WHERE id = 1").fetchone()
    assert row == ("practice",)
