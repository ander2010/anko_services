from __future__ import annotations

from typing import Any, List, Optional, Tuple

from workflow.chunking import adaptive_chunk_paragraphs, enforce_chunk_quality
from workflow.importance import ImportanceScorer
from workflow.llm import LLMImportanceClient, LLMQuestionGenerator, QAFormat
from workflow.sections import SectionReader
from workflow.qa import QAComposer
from pipeline.logging_config import get_logger
from pipeline.metadata import MetadataEnricher
from pipeline.normalization import TextNormalizer
from pipeline.types import ChunkCandidate

logger = get_logger(__name__)


class WorkflowCore:
    """Coordinates the ingestion pipeline."""

    def __init__(self, llm_client: Optional[LLMImportanceClient] = None, ga_generator: Optional[LLMQuestionGenerator] = None, ga_workers: int = 4) -> None:
        self.llm_client = llm_client
        self.ga_composer = QAComposer(ga_generator, ga_workers=ga_workers)
        self.metadata_enricher = MetadataEnricher()
        self.normalizer = TextNormalizer()

    def run(self, args: Any, on_progress: Optional[Any] = None) -> Tuple[List[ChunkCandidate], List[dict]]:
        logger.info("Starting workflow for %s", getattr(args, "input", None))

        def progress(page_idx: int, total: int) -> None:
            if on_progress:
                on_progress({"stage": "ocr", "page": page_idx, "total_pages": total})

        sections = SectionReader.read(getattr(args, "input"), getattr(args, "input_type", "pdf"), getattr(args, "dpi", 300), getattr(args, "lang", "eng"), on_progress=progress)
        logger.info("Read %s pages", len(sections))

        page_confidence = {int(s.get("page", 0) if isinstance(s, dict) else getattr(s, "page", 0)): float(s.get("confidence", 0.0) if isinstance(s, dict) else getattr(s, "confidence", 0.0)) for s in sections}

        paragraphs = self.normalizer.segment_into_paragraphs(sections, min_chars=getattr(args, "min_paragraph_chars", 40))
        logger.info("Segmented into %s paragraphs", len(paragraphs))

        chunk_candidates = adaptive_chunk_paragraphs(paragraphs, max_tokens=getattr(args, "max_chunk_tokens", 220), overlap=getattr(args, "chunk_overlap", 40))
        chunk_candidates = enforce_chunk_quality(chunk_candidates, min_tokens=getattr(args, "min_chunk_tokens", 40), max_tokens=getattr(args, "max_chunk_tokens", 220))
        logger.info("Chunked into %s candidates", len(chunk_candidates))

        scorer = ImportanceScorer(relevance_threshold=float(getattr(args, "importance_threshold", 0.4)), llm_client=self.llm_client)
        scored_chunks = scorer.score_chunks(chunk_candidates)
        scored_chunks = [c for c in scored_chunks if c.relevance]
        logger.info("Scored %s relevant chunks", len(scored_chunks))

        enriched_chunks = self.metadata_enricher.enrich(scored_chunks, page_confidence)
        logger.info("Enriched metadata for %s chunks", len(enriched_chunks))

        max_chunks = getattr(args, "max_chunks", None)
        if max_chunks is not None:
            try:
                max_chunks_int = int(max_chunks)
                enriched_chunks = enriched_chunks[:max_chunks_int]
                logger.info("Truncated to %s chunks due to max_chunks", len(enriched_chunks))
            except (TypeError, ValueError):
                logger.warning("Invalid max_chunks: %r", max_chunks)

        qa_pairs: List[dict] = []
        if getattr(args, "skip_ga", False):
            logger.info("Skipping QA generation (skip_ga=True)")
        else:
            qa_pairs = self.ga_composer.generate(enriched_chunks, max_answer_words=getattr(args, "ga_answer_length", 60), ga_format=getattr(args, "qa_format", QAFormat.VARIETY.value))
            logger.info("Generated %s QA pairs", len(qa_pairs))

        logger.info("Workflow completed for %s", getattr(args, "input", None))
        return enriched_chunks, qa_pairs
