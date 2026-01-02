"""
Minimal FSRS-style scheduler primitives.

This is intentionally simple and self-contained so it can be swapped out later
for a full FSRS implementation or SM-2. The goal is to demonstrate how to
maintain per-card state and update it on a review event.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

Grade = Literal["again", "hard", "good", "easy"]


@dataclass
class CardState:
    difficulty: float = 1.3  # Higher means harder; keep within [1.0, 2.5]
    stability: float = 2.5  # Roughly "retention half-life" in days
    last_review: dt.datetime = dt.datetime.now(dt.timezone.utc)
    interval: float = 1.0  # Days until next review


def update_fsrs(state: CardState, grade: Grade, now: dt.datetime | None = None) -> CardState:
    """
    Update card state based on the review grade.

    This is a simplified, single-step FSRS-like updater. It is NOT a full FSRS
    model; it just illustrates how difficulty can be nudged upward even on a
    "good" response and how stability/interval evolve with each review.
    """

    if now is None:
        now = dt.datetime.now(dt.timezone.utc)

    elapsed_days = max((now - state.last_review).total_seconds() / 86400.0, 0.01)

    difficulty = _adjust_difficulty(state.difficulty, grade)
    stability = _adjust_stability(state.stability, grade, elapsed_days)

    # Derive next interval: tie it to stability with a minimum of 1 day.
    interval = max(1.0, stability)

    return CardState(
        difficulty=difficulty,
        stability=stability,
        last_review=now,
        interval=interval,
    )


def _adjust_difficulty(difficulty: float, grade: Grade) -> float:
    # "good" intentionally nudges difficulty upward to match the requested behavior.
    deltas = {
        "again": 0.2,
        "hard": 0.1,
        "good": 0.05,
        "easy": -0.05,
    }
    new_diff = difficulty + deltas[grade]
    return float(min(max(new_diff, 1.0), 2.5))


def _adjust_stability(stability: float, grade: Grade, elapsed_days: float) -> float:
    # Crude forgetting curve: higher elapsed time reduces retention.
    retention = max(0.1, pow(0.9, elapsed_days / max(stability, 0.1)))

    if grade == "again":
        new_stability = stability * 0.7 * retention
    elif grade == "hard":
        new_stability = stability * (0.9 + retention * 0.05)
    elif grade == "good":
        new_stability = stability * (1.1 + retention * 0.1)
    else:  # easy
        new_stability = stability * (1.25 + retention * 0.1)

    return max(0.5, min(new_stability, 180.0))
