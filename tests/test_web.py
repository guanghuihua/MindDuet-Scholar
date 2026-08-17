from __future__ import annotations


class FakeCodexAssistant:
    def __init__(self) -> None:
        self.interrupted = False
        self.reset_called = False
        self.session_interrupted = False

    def status(self, document_id: int) -> dict[str, object]:
        return {
            "available": True,
            "mode": "read-only",
            "conversation_id": 4,
            "active": False,
            "messages": [{"role": "assistant", "body": r"已有结论：\(G/N\)。"}],
        }

    async def stream_reply(self, document, message: str, pdf_context):
        assert document["title"] == "sample"
        assert message == "解释这里"
        assert pdf_context["page"] == 7
        yield "start", {"conversation_id": 4, "turn_id": "turn-test"}
        yield "delta", {"delta": "这里使用了"}
        yield "delta", {"delta": r"第一同构定理 \(G/\ker\varphi\cong\operatorname{Im}\varphi\)。"}
        yield "done", {"message_id": 9}

    async def interrupt(self, document_id: int) -> bool:
        self.interrupted = True
        return True

    async def reset(self, document_id: int) -> bool:
        self.reset_called = True
        return True

    def session_status(self, session) -> dict[str, object]:
        return {
            "available": True,
            "mode": "inquiry",
            "session_id": session["id"],
            "active": False,
            "messages": session["messages"],
        }

    async def stream_session_reply(self, session, message: str):
        assert session["session_type"] == "inquiry"
        assert message == "为什么这两个记号同构？"
        yield "start", {"session_id": session["id"], "turn_id": "session-turn-test"}
        yield "delta", {"delta": r"它们都表示整数模 \(n\) 的循环群。"}
        yield "done", {"message_id": 10}

    async def interrupt_session(self, session_id: int) -> bool:
        self.session_interrupted = True
        return True

    async def close(self) -> None:
        return None


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


def test_library_search_form_includes_query(client, notes_root) -> None:
    (notes_root / "Linear Algebra Done Right.pdf").write_bytes(b"%PDF-1.4\n% test\n")

    client.post("/index", headers={"HX-Request": "true"})
    library = client.get("/library")
    assert library.status_code == 200
    assert 'hx-include="#library-search"' in library.text

    results = client.get("/library?q=linear%20algebra%20done%20right", headers={"HX-Request": "true"})
    assert results.status_code == 200
    assert "Linear Algebra Done Right" in results.text


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
    assert pdf_file.headers["content-disposition"].startswith("inline;")

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


def test_pdf_reader_embeds_streaming_codex_assistant(client, notes_root) -> None:
    (notes_root / "sample.pdf").write_bytes(b"%PDF-1.4\n% MindDuet test PDF\n")
    client.post("/index", headers={"HX-Request": "true"})
    document_id = client.app.state.repository.search_documents("sample")[0]["id"]
    client.post(
        "/reader/context",
        json={
            "document_id": document_id,
            "page": 7,
            "scale": 1.2,
            "selected_text": "First isomorphism theorem",
            "page_text": "The quotient by the kernel is isomorphic to the image.",
        },
    )
    assistant = FakeCodexAssistant()
    client.app.state.codex_assistant = assistant

    reader = client.get(f"/reader?document_id={document_id}")
    assert reader.status_code == 200
    assert "Study assistant" in reader.text
    assert 'data-assistant-stream-url="/reader/assistant/stream"' in reader.text

    status = client.get(f"/reader/assistant?document_id={document_id}")
    assert status.status_code == 200
    assert status.json()["mode"] == "read-only"
    assert status.json()["messages"][0]["role"] == "assistant"

    streamed = client.post(
        "/reader/assistant/stream",
        json={"document_id": document_id, "message": "解释这里"},
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "event: start" in streamed.text
    assert "event: delta" in streamed.text
    assert "第一同构定理" in streamed.text
    assert "event: done" in streamed.text

    stopped = client.post("/reader/assistant/interrupt", json={"document_id": document_id})
    assert stopped.json() == {"interrupted": True}
    assert assistant.interrupted is True

    reset = client.post("/reader/assistant/reset", json={"document_id": document_id})
    assert reset.json() == {"reset": True, "archived": True}
    assert assistant.reset_called is True


def test_inquiry_session_embeds_persistent_codex_dialogue(client) -> None:
    created = client.post(
        "/sessions",
        data={
            "goal": "Understand Z/nZ and Z_n",
            "session_type": "inquiry",
        },
    )
    assert created.status_code == 200
    session = client.app.state.repository.list_sessions()[0]
    session_id = session["id"]
    client.app.state.repository.add_session_message(
        session_id,
        "assistant",
        r"先看元素 \([a])\) 的含义。",
    )
    assistant = FakeCodexAssistant()
    client.app.state.codex_assistant = assistant

    detail = client.get(f"/sessions/{session_id}")
    assert detail.status_code == 200
    assert "Ask &amp; Understand" in detail.text
    assert "Ask, connect, and refine" in detail.text
    assert "Save attempt" not in detail.text
    assert "Schedule 1d + 7d reviews" not in detail.text
    assert "Previous message" not in detail.text
    assert "session_navigation.js" not in detail.text
    assert "session_assistant.js" in detail.text

    status = client.get(f"/sessions/{session_id}/assistant")
    assert status.status_code == 200
    assert status.json()["mode"] == "inquiry"
    assert status.json()["messages"][0]["role"] == "assistant"

    streamed = client.post(
        f"/sessions/{session_id}/assistant/stream",
        json={"message": "为什么这两个记号同构？"},
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "event: start" in streamed.text
    assert "整数模" in streamed.text
    assert "event: done" in streamed.text

    stopped = client.post(
        f"/sessions/{session_id}/assistant/interrupt",
        json={"session_id": session_id},
    )
    assert stopped.json() == {"interrupted": True}
    assert assistant.session_interrupted is True


def test_practice_session_keeps_attempt_workflow_and_rejects_chat(client) -> None:
    session_id = client.app.state.repository.start_session("Prove a quotient property")

    detail = client.get(f"/sessions/{session_id}")
    assert "Practice &amp; Prove" in detail.text
    assert "Save attempt" in detail.text
    assert "session_assistant.js" not in detail.text

    status = client.get(f"/sessions/{session_id}/assistant")
    assert status.status_code == 409


def test_completed_inquiry_session_can_be_reopened_with_history(client) -> None:
    repository = client.app.state.repository
    session_id = repository.start_session("Continue understanding quotient groups", session_type="inquiry")
    repository.add_session_message(session_id, "user", "What is a coset?")
    repository.add_session_message(session_id, "assistant", "A coset is a translated subgroup.")

    completed = client.post(f"/sessions/{session_id}/complete")
    assert completed.status_code == 200
    completed_detail = client.get(f"/sessions/{session_id}")
    assert "Reopen session" in completed_detail.text
    assert "session_assistant.js" not in completed_detail.text
    assert "A coset is a translated subgroup." in completed_detail.text

    reopened = client.post(f"/sessions/{session_id}/reopen")
    assert reopened.status_code == 200
    reopened_detail = client.get(f"/sessions/{session_id}")
    assert "Complete session" in reopened_detail.text
    assert "Reopen session" not in reopened_detail.text
    assert "session_assistant.js" in reopened_detail.text
    assert "A coset is a translated subgroup." in reopened_detail.text

    session = repository.get_session(session_id)
    assert session is not None
    assert session["status"] == "active"
    assert session["completed_at"] is None


def test_active_session_cannot_be_reopened(client) -> None:
    session_id = client.app.state.repository.start_session("Still active")

    response = client.post(f"/sessions/{session_id}/reopen")

    assert response.status_code == 409
