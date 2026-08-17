from __future__ import annotations

import asyncio

from mindduet_scholar.codex_assistant import CodexAssistant


class AlreadyStoppedTurn:
    async def interrupt(self) -> None:
        raise RuntimeError("JSON-RPC error -32600: no active turn to interrupt")


def test_interrupt_is_idempotent_when_turn_already_stopped(settings) -> None:
    assistant = CodexAssistant(settings, repository=None)  # type: ignore[arg-type]
    assistant._active_turns[("document", 7)] = AlreadyStoppedTurn()

    assert asyncio.run(assistant.interrupt(7)) is False


def test_document_and_session_turns_with_same_id_do_not_collide(settings) -> None:
    assistant = CodexAssistant(settings, repository=None)  # type: ignore[arg-type]
    assistant._active_turns[("document", 7)] = None

    assert assistant.session_status({"id": 7, "messages": []})["active"] is False
    assistant._active_turns[("session", 7)] = None
    assert assistant.session_status({"id": 7, "messages": []})["active"] is True


def test_session_prompt_preserves_goal_history_and_source() -> None:
    prompt = CodexAssistant._build_session_prompt(
        {
            "goal": "Understand quotient notation",
            "source_title": "Abstract Algebra",
            "source_path": "library/algebra.pdf",
        },
        r"Why is \(\mathbb Z/n\mathbb Z\cong\mathbb Z_n\)?",
        [{"role": "assistant", "body": "先区分集合与记号。"}],
    )

    assert "Understand quotient notation" in prompt
    assert "library/algebra.pdf" in prompt
    assert "先区分集合与记号" in prompt
    assert r"\mathbb Z/n\mathbb Z" in prompt
