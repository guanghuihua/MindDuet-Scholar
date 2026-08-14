from __future__ import annotations


def test_full_learning_loop_through_web(client) -> None:
    index_response = client.post("/index", headers={"HX-Request": "true"})
    assert index_response.status_code == 200
    assert "Indexed 2 source documents" in index_response.text

    library = client.get("/library?q=Quotient")
    assert library.status_code == 200
    assert "Quotient groups" in library.text
    document_id = client.app.state.repository.search_documents("Quotient")[0]["id"]

    created = client.post(
        "/sessions",
        data={"goal": "Show the quotient map is well defined", "document_id": str(document_id), "source_anchor": "Exercise 32.6"},
    )
    assert created.status_code == 200
    sessions = client.app.state.repository.list_sessions()
    session_id = sessions[0]["id"]

    blocked_hint = client.post(f"/sessions/{session_id}/hints")
    assert "Write an attempt before asking for a hint" in blocked_hint.text

    attempt = client.post(
        f"/sessions/{session_id}/attempts",
        data={"body": "Clearly, therefore the claim follows by definition."},
    )
    assert attempt.status_code == 200
    hint = client.post(f"/sessions/{session_id}/hints")
    assert hint.status_code == 200
    assert "Tier 1 hint" in hint.text

    reviews = client.post(f"/sessions/{session_id}/reviews")
    assert reviews.status_code == 200
    report = client.get("/reports")
    assert "Recurring signals" in report.text
    assert "missing justification" in report.text


def test_pdf_reader_saves_current_context(client, notes_root) -> None:
    (notes_root / "sample.pdf").write_bytes(b"%PDF-1.4\n% MindDuet test PDF\n")

    index_response = client.post("/index", headers={"HX-Request": "true"})
    assert index_response.status_code == 200
    document_id = client.app.state.repository.search_documents("sample")[0]["id"]

    reader = client.get(f"/reader?document_id={document_id}")
    assert reader.status_code == 200
    assert "PDF reading mode" in reader.text

    pdf_file = client.get(f"/reader/documents/{document_id}/file")
    assert pdf_file.status_code == 200
    assert pdf_file.content.startswith(b"%PDF")

    saved = client.post(
        "/reader/context",
        json={
            "document_id": document_id,
            "page": 7,
            "scale": 1.2,
            "selected_text": "Since every boundary is a cycle.",
            "page_text": "Since every boundary is a cycle, the quotient is well-defined.",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["page"] == 7

    current = client.get("/reader/context")
    assert current.status_code == 200
    assert current.json()["relative_path"] == "sample.pdf"
    assert current.json()["selected_text"] == "Since every boundary is a cycle."
    assert current.json()["page_text"] == "Since every boundary is a cycle, the quotient is well-defined."
