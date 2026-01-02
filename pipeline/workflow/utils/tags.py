from __future__ import annotations

from typing import List, Sequence

from pipeline.utils.logging_config import get_logger
from pipeline.workflow.llm import LLMQuestionGenerator

logger = get_logger(__name__)


def collect_tags_from_payload(chunks: Sequence[dict] | None, embeddings: Sequence[dict] | None) -> List[str]:
    tag_set = set()
    for chunk in chunks or []:
        for tag in chunk.get("tags", []) or []:
            tag_set.add(str(tag))
    for emb in embeddings or []:
        meta = emb.get("metadata") or {}
        for tag in meta.get("tags", []) or []:
            tag_set.add(str(tag))
    return sorted(tag_set)


def ensure_llm_active_warning(llm_generator: LLMQuestionGenerator | None) -> None:
    if llm_generator and not llm_generator.is_active:
        logger.warning("LLM is inactive (missing API key); tagging will use fallback.")


def infer_tags_with_llm(llm_generator: LLMQuestionGenerator | None, text: str, fallback_max: int = 5, warn: bool = True) -> List[str]:
    if llm_generator and llm_generator.is_active:
        try:
            return llm_generator.tag_text(text)
        except Exception:
            logger.warning("LLM tagging failed; falling back to heuristic", exc_info=warn)
    if warn:
        logger.warning("LLM inactive; using heuristic tags")
    return [part.strip() for part in text.split(",") if part.strip()][:fallback_max]
