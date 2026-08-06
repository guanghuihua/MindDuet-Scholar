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
