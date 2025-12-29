from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Optional, Sequence, Tuple

from pipeline.llm import LLMQuestionGenerator, QAFormat
from pipeline.logging_config import get_logger
from pipeline.types import ChunkCandidate

logger = get_logger(__name__)


class QAComposer:
    """Creates QA pairs using either LLM output or a heuristic fallback."""

    SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")

    def __init__(
        self,
        ga_generator: Optional[LLMQuestionGenerator] = None,
        min_context_tokens: int = 80,
        context_token_budget: int = 600,
        summary_token_budget: int = 360,
        importance_floor: float = 2.0,
        ga_workers: int = 4,
        theme_hint: Optional[str] = None,
        difficulty_hint: Optional[str] = None,
        target_questions: Optional[int] = None,
    ) -> None:
        self.ga_generator = ga_generator
        self.min_context_tokens = int(min_context_tokens)
        self.context_token_budget = int(context_token_budget)
        self.summary_token_budget = int(summary_token_budget)
        self.importance_floor = float(importance_floor)
        self.ga_workers = int(ga_workers)
        self.theme_hint = theme_hint
        self.difficulty_hint = difficulty_hint
        self.target_questions = target_questions

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text.split())

    def _summarize_text(self, text: str) -> str:
        """Trim overly long prompts while keeping contiguous sentences."""
        sentences = [segment.strip() for segment in self.SENTENCE_SPLIT.split(text) if segment.strip()]
        if not sentences:
            return text
        summary: List[str] = []
        token_budget = 0
        for sentence in sentences:
            toks = self._estimate_tokens(sentence)
            if token_budget + toks > self.summary_token_budget and summary:
                break
            summary.append(sentence)
            token_budget += toks
        logger.info("QA summarize  | orig_tokens=%s summary_tokens=%s", self._estimate_tokens(text), token_budget)
        return " ".join(summary)

    def _bundle_chunks(self, chunks: Sequence[ChunkCandidate]) -> List[dict]:
        bundled: List[dict] = []
        logger.info("QA bundle start | chunks=%s", len(chunks))
        idx = 0
        total = len(chunks)
        while idx < total:
            bundle_chunks: List[ChunkCandidate] = [chunks[idx]]
            tokens = chunks[idx].tokens or self._estimate_tokens(chunks[idx].text)
            max_importance = getattr(chunks[idx], "importance", 0) or 0
            idx += 1
            while tokens < self.min_context_tokens and idx < total:
                next_chunk = chunks[idx]
                bundle_chunks.append(next_chunk)
                tokens += next_chunk.tokens or self._estimate_tokens(next_chunk.text)
                max_importance = max(max_importance, getattr(next_chunk, "importance", 0) or 0)
                idx += 1
            text = "\n\n".join(ch.text.strip() for ch in bundle_chunks if getattr(ch, "text", "").strip())
            if self._estimate_tokens(text) > self.context_token_budget:
                text = self._summarize_text(text)
            bundle = {
                "text": text,
                "tokens": self._estimate_tokens(text),
                "chunks": bundle_chunks,
                "max_importance": max_importance,
                "pages": sorted(int(getattr(ch, "page", 0) or ch.metadata.get("page", 0)) for ch in bundle_chunks),
                "tags": sorted({tag for ch in bundle_chunks for tag in (getattr(ch, "tags", None) or ch.metadata.get("tags", []) or [])}),
            }
            bundled.append(bundle)
            logger.info("QA bundle made | idx=%s size=%s tokens=%s max_imp=%.2f pages=%s tags=%s", len(bundled) - 1, len(bundle_chunks), bundle["tokens"], max_importance, bundle["pages"], bundle["tags"])
        return bundled

    @staticmethod
    def _question_key(question: str, answers: List[str]) -> Tuple[str, Tuple[str, ...]]:
        normalized_question = " ".join(question.lower().split())
        normalized_answers = tuple(sorted("".join(ans.lower().split()) for ans in answers))
        return normalized_question, normalized_answers

    def generate_stream(self, chunks: Sequence[ChunkCandidate], max_answer_words: int, ga_format: QAFormat | str, progress_cb: Optional[Any] = None) -> Iterator[dict]:
        fmt = QAFormat.from_value(ga_format) if not isinstance(ga_format, QAFormat) else ga_format
        seen_questions: set[Tuple[str, Tuple[str, ...]]] = set()

        bundles = list(self._bundle_chunks(chunks))
        total_bundles = len(bundles)
        logger.info("QA stream start | bundles=%s chunks=%s format=%s", total_bundles, len(chunks), fmt)
        emitted = 0

        for bundle_idx, bundle in enumerate(bundles):
            if bundle["max_importance"] < self.importance_floor:
                if progress_cb:
                    try:
                        progress_cb(None, {"bundle_index": bundle_idx, "total_bundles": total_bundles, "skipped": True})
                    except Exception:
                        logger.warning("A progress callback failed, continuing.")
                logger.info("QA bundle skip | idx=%s importance=%.2f floor=%.2f", bundle_idx, bundle["max_importance"], self.importance_floor)
                yield from ()
                continue

            generated: List[dict] = []
            if self.ga_generator:
                is_active_attr = getattr(self.ga_generator, "is_active", None)
                try:
                    is_active = is_active_attr() if callable(is_active_attr) else bool(is_active_attr)
                except Exception:
                    is_active = False
                if is_active:
                    logger.info("QA LLM call   | bundle=%s/%s pages=%s tags=%s tokens=%s", bundle_idx + 1, total_bundles, bundle.get("pages"), bundle.get("tags"), bundle.get("tokens"))
                    try:
                        # Use a small thread pool to avoid blocking the caller
                        with ThreadPoolExecutor(max_workers=self.ga_workers) as executor:
                            future = executor.submit(
                                self.ga_generator.generate,
                                text=bundle["text"],
                                page=getattr(bundle["chunks"][0], "page", None),
                                tags=bundle.get("tags", []),
                                max_answer_words=max_answer_words,
                                mode=fmt,
                                theme_hint=self.theme_hint,
                                difficulty_hint=self.difficulty_hint,
                                target_questions=self.target_questions,
                            )
                            generated = future.result() or []
                            logger.info("QA LLM result | bundle=%s/%s items=%s", bundle_idx + 1, total_bundles, len(generated))
                    except Exception as exc:
                        logging.warning("LL QA generation failed; falling back to heuristics: %s", exc)
                else:
                    logger.debug("LLM QA generator inactive or unavailable; using heuristic generation.")

            # Process generated items
            for item in generated:
                answers = item.get("answers") or item.get("correct_answer") or []
                if isinstance(answers, str):
                    answers = [answers]
                answers = [str(a).strip() for a in answers if str(a).strip()]
                question_text = (item.get("question") or "").strip()
                if not question_text or not answers:
                    logger.info("QA drop empty | bundle=%s/%s", bundle_idx + 1, total_bundles)
                    continue

                question_key = self._question_key(question_text, answers)
                if question_key in seen_questions:
                    logger.info("QA duplicate  | bundle=%s/%s question=%s", bundle_idx + 1, total_bundles, question_text[:80])
                    continue
                seen_questions.add(question_key)

                question_metadata = dict(getattr(bundle["chunks"][0], "metadata", None) or {})
                chunk_ids = [ch.metadata.get("chunk_id") for ch in bundle["chunks"] if isinstance(getattr(ch, "metadata", None), dict) and ch.metadata.get("chunk_id")]

                fmt_val = fmt.value if hasattr(fmt, "value") else str(fmt)

                ga_item = {
                    "question": question_text,
                    "correct_response": "; ".join(answers),
                    "context": bundle["text"],
                    "metadata": {**question_metadata, "type": item.get("type", "short_answer"), "options": item.get("options", []), "answers": answers, "explanation": item.get("explanation", "")},
                    "page": bundle["chunks"][0].page if hasattr(bundle["chunks"][0], "page") else None,
                    "pages": bundle["pages"],
                    "tags": bundle.get("tags", []),
                    "format": fmt_val,
                    "source_chunks": len(bundle["chunks"]),
                    "chunk_ids": chunk_ids,
                }

                if progress_cb:
                    try:
                        progress_cb(ga_item, {"bundle_index": bundle_idx, "total_bundles": total_bundles, "skipped": False})
                    except Exception as e :
                        logger.warning(f"QA progress callback failed; continuing. Exception : {str(e)}")

                emitted += 1
                yield ga_item

        logger.info("QA stream done  | bundles=%s emitted=%s", total_bundles, emitted)

    def generate(self, chunks: Sequence[ChunkCandidate], max_answer_words: int, ga_format: QAFormat | str, progress_cb: Optional[Any] = None) -> List[dict]:
        logger.info("QA generate start | chunks=%s format=%s", len(chunks), ga_format)
        qa: List[dict] = []
        for idx, item in enumerate(self.generate_stream(chunks, max_answer_words=max_answer_words, ga_format=ga_format, progress_cb=progress_cb)):
            item["section"] = idx + 1
            qa.append(item)
            if progress_cb:
                try:
                    progress_cb(item, idx + 1)
                except Exception:
                    logger.warning("A progress callback failed; continuing.")
        logger.info("QA generate done  | total_pairs=%s", len(qa))
        return qa
