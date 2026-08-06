from __future__ import annotations

from datetime import date, timedelta

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
