from __future__ import annotations

import requests

from pipeline.celery_tasks import flashcards as flashcard_tasks


class _OkResponse:
    status_code = 200
    ok = True

    def raise_for_status(self) -> None:
        return None


def test_post_flashcard_finalize_callback_retries_transient_failures(monkeypatch) -> None:
    attempts: list[int] = []

    def _fake_post(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.RequestException("temporary callback failure")
        return _OkResponse()

    monkeypatch.setattr(flashcard_tasks.requests, "post", _fake_post)
    monkeypatch.setattr(flashcard_tasks.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("FLASHCARD_FINALIZE_CALLBACK_ATTEMPTS", "3")
    monkeypatch.setenv("FLASHCARD_FINALIZE_CALLBACK_DELAY_SECONDS", "0")

    flashcard_tasks._post_flashcard_finalize_callback(
        {
            "job_id": "job-1",
            "deck_id": 7,
            "title": "Deck",
            "metadata": {
                "callback_url": "http://callback.local/finalize",
                "callback_token": "secret",
            },
        },
        {"title": "Deck", "generated": 5, "total": 5},
        status_value="completed",
    )

    assert len(attempts) == 3
