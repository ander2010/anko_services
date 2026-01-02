from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from pipeline.workflow.llm import LLMChunkAssessment, LLMImportanceClient
from pipeline.utils.types import ChunkCandidate

CONCEPT_KEYWORDS = {
    "Definition": ["define", "definition", "means", "refers to"],
    "Fact": ["date", "year", "value", "equals", "measures"],
    "Explanation": ["because", "therefore", "explains", "results"],
    "Example": ["example", "such as", "e.g.", "for instance"],
}


@dataclass
class ImportanceResult:
    relevance: bool
    importance_score: float
    concept_type: str


class ImportanceScorer:
    """Placeholder that approximates LLM-based scoring using simple heuristics."""

    def __init__(self, relevance_threshold: float = 0.4, llm_client: Optional[LLMImportanceClient] = None) -> None:
        self.relevance_threshold = relevance_threshold
        self.llm_client = llm_client

    def _concept_type(self, text: str) -> str:
        lowered = text.lower()
        for concept, needles in CONCEPT_KEYWORDS.items():
            if any(needle in lowered for needle in needles):
                return concept
        return "Explanation"

    def _importance_score(self, text: str) -> float:
        score = 0.0
        lowered = text.lower()
        for concept, needles in CONCEPT_KEYWORDS.items():
            if any(needle in lowered for needle in needles):
                score += 0.8
        score += min(3.0, len(text) / 400)
        return min(5.0, round(score, 2))

    def _heuristic_score(self, chunk: ChunkCandidate) -> ChunkCandidate:
        concept_type = self._concept_type(chunk.text)
        importance = self._importance_score(chunk.text)
        relevance = importance / 5.0 >= self.relevance_threshold
        chunk.concept_type = concept_type
        chunk.importance = importance
        chunk.relevance = relevance
        return chunk

    def _apply_llm_result(self, chunk: ChunkCandidate, assessment: LLMChunkAssessment) -> ChunkCandidate:
        chunk.relevance = bool(assessment.relevance)
        chunk.importance = max(0.0, min(5.0, assessment.importance))
        chunk.concept_type = assessment.concept_type or chunk.concept_type
        chunk.tags = assessment.tags or chunk.tags
        chunk.difficulty = assessment.difficulty or chunk.difficulty
        logging.info(
            "LLM assessment applied | page=%s relevance=%s importance=%.2f concept=%s tags=%s difficulty=%s text_preview=%s",
            getattr(chunk, "page", None),
            chunk.relevance,
            chunk.importance,
            chunk.concept_type,
            chunk.tags,
            chunk.difficulty,
            (chunk.text[:120] + "...") if getattr(chunk, "text", None) and len(chunk.text) > 120 else getattr(chunk, "text", None),
        )
        return chunk

    def score_chunk(self, chunk: ChunkCandidate) -> ChunkCandidate:
        if self.llm_client and self.llm_client.is_active:
            try:
                assessment = self.llm_client.assess_chunk(chunk.text, chunk.page)
                return self._apply_llm_result(chunk, assessment)
            except RuntimeError as exc:
                logging.warning("LLM scoring failed, falling back to heuristics: %s", exc)
        return self._heuristic_score(chunk)

    def score_chunks(self, chunks: List[ChunkCandidate]) -> List[ChunkCandidate]:
        return [self.score_chunk(chunk) for chunk in chunks]
