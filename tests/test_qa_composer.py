from __future__ import annotations

from pipeline.utils.types import ChunkCandidate
from pipeline.workflow.qa import QAComposer


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def is_active(self) -> bool:
        return True

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        call_number = len(self.calls)
        return [
            {
                "question": f"Question {call_number}",
                "answers": [f"Answer {call_number}"],
                "type": "short_answer",
                "options": [],
                "explanation": "",
            }
        ]


def _chunk(*, page: int, text: str, tokens: int, importance: float, document_id: str) -> ChunkCandidate:
    return ChunkCandidate(
        page=page,
        text=text,
        tokens=tokens,
        importance=importance,
        tags=["payment"],
        metadata={"document_id": document_id, "page": str(page)},
    )


def test_qa_composer_keeps_single_low_importance_bundle() -> None:
    generator = _FakeGenerator()
    composer = QAComposer(ga_generator=generator, importance_floor=2.0, target_questions=1)

    qa_pairs = composer.generate(
        [
            _chunk(
                page=1,
                text="A single-page receipt with one useful chunk of content.",
                tokens=120,
                importance=1.0,
                document_id="23",
            )
        ],
        max_answer_words=40,
        ga_format="multiple_choice",
    )

    assert len(qa_pairs) == 1
    assert len(generator.calls) == 1


def test_qa_composer_keeps_all_low_importance_bundles_when_every_bundle_is_below_floor() -> None:
    generator = _FakeGenerator()
    composer = QAComposer(ga_generator=generator, importance_floor=2.0, target_questions=2)

    qa_pairs = composer.generate(
        [
            _chunk(page=1, text="First chunk.", tokens=120, importance=1.0, document_id="55"),
            _chunk(page=2, text="Second chunk.", tokens=120, importance=1.5, document_id="55"),
        ],
        max_answer_words=40,
        ga_format="multiple_choice",
    )

    assert len(qa_pairs) == 2
    assert len(generator.calls) == 2


def test_qa_composer_still_skips_low_importance_bundles_when_other_bundles_meet_floor() -> None:
    generator = _FakeGenerator()
    composer = QAComposer(ga_generator=generator, importance_floor=2.0, target_questions=2)

    qa_pairs = composer.generate(
        [
            _chunk(page=1, text="Low importance chunk.", tokens=120, importance=1.0, document_id="77"),
            _chunk(page=2, text="High importance chunk.", tokens=120, importance=3.0, document_id="77"),
        ],
        max_answer_words=40,
        ga_format="multiple_choice",
    )

    assert len(qa_pairs) == 1
    assert len(generator.calls) == 1
