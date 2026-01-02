from __future__ import annotations

from typing import Dict, Iterable, List

from pipeline.utils.types import ChunkCandidate


class MetadataEnricher:
    """Encapsulates heuristics for tags, difficulty, and metadata assembly."""

    SUBJECT_KEYWORDS = {
        "math": ["theorem", "equation", "integral", "derivative", "algebra"],
        "physics": ["force", "energy", "velocity", "quantum", "particle"],
        "biology": ["cell", "gene", "enzyme", "organism"],
        "chemistry": ["molecule", "reaction", "compound", "bond"],
        "history": ["war", "treaty", "revolution", "empire"],
    }

    DIFFICULTY_THRESHOLDS = {
        "easy": 80,
        "medium": 180,
    }

    def __init__(self, subject_keywords: Dict[str, List[str]] | None = None, difficulty_thresholds: Dict[str, int] | None = None) -> None:
        self.subject_keywords = subject_keywords or self.SUBJECT_KEYWORDS
        self.difficulty_thresholds = difficulty_thresholds or self.DIFFICULTY_THRESHOLDS

    def infer_tags(self, text: str) -> List[str]:
        lowered = text.lower()
        tags: List[str] = []
        for subject, needles in self.subject_keywords.items():
            if any(word in lowered for word in needles):
                tags.append(subject)
        return tags or ["general"]

    def infer_difficulty(self, tokens: int, importance: float) -> str:
        easy_threshold = self.difficulty_thresholds["easy"]
        medium_threshold = self.difficulty_thresholds["medium"]
        if tokens <= easy_threshold:
            return "easy" if importance < 3 else "medium"
        if tokens <= medium_threshold:
            return "medium"
        return "hard"

    def enrich(self, chunks: Iterable[ChunkCandidate], page_confidence: Dict[int, float]) -> List[ChunkCandidate]:
        enriched: List[ChunkCandidate] = []
        for chunk in chunks:
            chunk.tags = chunk.tags or self.infer_tags(chunk.text)
            chunk.difficulty = chunk.difficulty or self.infer_difficulty(chunk.tokens, chunk.importance)
            chunk.metadata = {
                "page": chunk.page,
                "difficulty": chunk.difficulty,
                "importance": chunk.importance,
                "concept_type": chunk.concept_type,
                "tags": chunk.tags,
                "ocr_confidence": round(page_confidence.get(chunk.page, 0.0), 2),
                "relevance": chunk.relevance,
            }
            enriched.append(chunk)
        return enriched
