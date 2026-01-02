from __future__ import annotations

from typing import Sequence

from fastapi import HTTPException
from openai import OpenAI

MODERATION_MODEL = "omni-moderation-latest"


class SafetyValidator:
    """Reusable validation helper for questions and free-text lists."""

    def __init__(self, api_key: str | None, model: str = MODERATION_MODEL):
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is required to validate input.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def validate_question(self, question: str) -> None:
        question_text = (question or "").strip()
        if not question_text:
            raise HTTPException(status_code=400, detail="question is required")
        self._moderate([question_text], field="question")

    def validate_text_list(self, values: Sequence[str], field: str = "text list") -> None:
        items = [str(val).strip() for val in (values or []) if str(val).strip()]
        if not items:
            return
        self._moderate(items, field=field)

    def _moderate(self, items: Sequence[str], field: str) -> None:
        try:
            result = self.client.moderations.create(model=self.model, input=items)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{field} validation failed.") from exc

        flagged_indices: list[int] = []
        try:
            flagged_indices = [idx for idx, item in enumerate(result.results or []) if getattr(item, "flagged", False)]
        except Exception:
            flagged_indices = []

        if flagged_indices:
            raise HTTPException(status_code=400, detail=f"{field} contains disallowed content .")
