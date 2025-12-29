from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OCRPageResult:
    """Raw OCR output for a single PDF page."""

    page: int
    raw_text: str
    cleaned_text: str
    confidence: float


@dataclass
class Paragraph:
    """Normalized paragraph derived from OCR content."""

    page: int
    text: str


@dataclass
class ChunkCandidate:
    """Intermediate representation used before embeddings are produced."""

    page: int
    text: str
    tokens: int
    importance: float = 0.0
    relevance: bool = True
    concept_type: str = "Explanation"
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    metadata: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class ChunkEmbedding:
    """Final chunk payload containing embeddings and metadata."""

    text: str
    embedding: List[float]
    metadata: Dict[str, Optional[str]] = field(default_factory=dict)
