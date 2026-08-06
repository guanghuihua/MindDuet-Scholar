"""Constrained tutoring: diagnosis and small hints, never an answer dump."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Settings


MISCONCEPTION_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("definition misuse", "A definition appears to be invoked without its required conditions.", ("by definition", "definition")),
    ("missing justification", "A conclusion is asserted without an explicit supporting rule or theorem.", ("therefore", "hence", "obvious", "clearly")),
    ("quantifier confusion", "Quantifier language may need a more careful order or domain.", ("for all", "exists", "arbitrary")),
    ("notation ambiguity", "A symbol is used without being introduced or its role changes mid-attempt.", ("let", "assume", "denote")),
)


@dataclass(frozen=True, slots=True)
class Hint:
    text: str
    provider: str


class Tutor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def diagnose(self, attempt: str) -> list[tuple[str, str]]:
        normalized = attempt.lower()
        findings: list[tuple[str, str]] = []
        for name, description, triggers in MISCONCEPTION_RULES:
            if any(trigger in normalized for trigger in triggers):
                findings.append((name, description))
        if not findings and len(attempt.split()) < 12:
            findings.append(("incomplete attempt", "The attempt is too short to expose the first mathematical dependency."))
        return findings[:2]

    def tier_one_hint(self, goal: str, attempt: str) -> Hint:
        """Return one orienting question. External calls are optional and guarded."""
        if self.settings.ai_api_key and self.settings.ai_model:
            remote_hint = self._remote_hint(goal, attempt)
            if remote_hint:
                return Hint(remote_hint, "openai-compatible")
        diagnosis = self.diagnose(attempt)
        if diagnosis:
            name, description = diagnosis[0]
            return Hint(
                f"First check: {description} Before continuing, name the exact definition or theorem that licenses your next step.",
                "local",
            )
        return Hint(
            f"First check: restate the target of '{goal}' in your own notation, then identify the one fact your next line must use. Do not calculate further yet.",
            "local",
        )

    def _remote_hint(self, goal: str, attempt: str) -> str | None:
        prompt = (
            "Give exactly one tier-one Socratic hint for a mathematics learner. "
            "Do not state a solution, proof outline, theorem name not supplied by the learner, or more than one next action. "
            f"Goal: {goal}\nLearner attempt: {attempt}"
        )
        payload = json.dumps({"model": self.settings.ai_model, "input": prompt, "max_output_tokens": 120}).encode()
        request = Request(
            f"{self.settings.ai_base_url}/responses",
            data=payload,
            headers={"Authorization": f"Bearer {self.settings.ai_api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=15) as response:  # nosec B310: URL comes from explicit local config
                data = json.loads(response.read().decode("utf-8"))
            text = str(data.get("output_text", "")).strip()
            return self._valid_tier_one(text)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            return None

    @staticmethod
    def _valid_tier_one(text: str) -> str | None:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned or len(cleaned) > 500:
            return None
        answer_markers = ("solution", "therefore the proof", "the answer is", "full proof")
        if any(marker in cleaned.lower() for marker in answer_markers):
            return None
        return cleaned
