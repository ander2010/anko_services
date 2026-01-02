import datetime as dt
import pathlib
import sys

import pytest

# Ensure the repository root is on the path for direct test runs.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline.utils.scheduler import CardState, update_fsrs


def test_good_review_increases_difficulty_and_interval():
    base_time = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    state = CardState(difficulty=1.2, stability=2.0, last_review=base_time, interval=2.0)

    updated = update_fsrs(state, grade="good", now=base_time + dt.timedelta(days=1))

    assert updated.difficulty > state.difficulty
    assert updated.interval >= state.interval
    assert updated.last_review > state.last_review


@pytest.mark.parametrize("grade", ["again", "hard", "good", "easy"])
def test_grade_mapping_stays_within_bounds(grade):
    state = CardState(
        difficulty=1.5,
        stability=5.0,
        last_review=dt.datetime.now(dt.timezone.utc),
        interval=5.0,
    )
    updated = update_fsrs(state, grade=grade)

    assert 1.0 <= updated.difficulty <= 2.5
    assert 0.5 <= updated.stability <= 180.0
    assert updated.interval >= 1.0
