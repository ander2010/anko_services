from __future__ import annotations

import os
import re
from typing import Any

from pipeline.utils.logging_config import get_logger
from pipeline.workflow.llm import LLMOutputSummarizer
from pipeline.workflow.utils.request_models import GenerationKind, SourceBundle

logger = get_logger("pipeline.title_generation")

_KIND_SUFFIX = {
    GenerationKind.ASSESSMENT.value: "Assessment",
    GenerationKind.FLASHCARDS.value: "Flashcards",
}

_FINAL_FALLBACK = {
    GenerationKind.ASSESSMENT.value: "Generated Assessment",
    GenerationKind.FLASHCARDS.value: "Generated Flashcards",
}


def _clean_phrase(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -_/")


def _title_case_phrase(value: str) -> str:
    words = []
    for raw_word in _clean_phrase(value).split():
        if raw_word.isupper() and len(raw_word) <= 5:
            words.append(raw_word)
        else:
            words.append(raw_word.capitalize())
    return " ".join(words)


def _dedupe_phrases(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = _clean_phrase(value)
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(phrase)
    return cleaned


def _coerce_source_bundle(source_bundle: SourceBundle | dict[str, Any] | None) -> SourceBundle:
    if isinstance(source_bundle, SourceBundle):
        return source_bundle
    if isinstance(source_bundle, dict):
        return SourceBundle(**source_bundle)
    return SourceBundle()


def _extract_seed_phrases(source_bundle: SourceBundle, metadata: dict[str, Any] | None = None) -> list[str]:
    metadata = metadata or {}
    phrases = []
    phrases.extend(source_bundle.normalized_title_hints())
    phrases.extend(source_bundle.normalized_tags())
    extra_hints = metadata.get("title_hints")
    if isinstance(extra_hints, list):
        phrases.extend(str(item) for item in extra_hints)
    collection_label = metadata.get("collection_label") or metadata.get("collection_title")
    if collection_label:
        phrases.append(str(collection_label))
    return _dedupe_phrases(phrases)


def build_fallback_title(*, kind: GenerationKind | str, source_bundle: SourceBundle | dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> str:
    kind_value = GenerationKind(kind).value if not isinstance(kind, GenerationKind) else kind.value
    normalized_bundle = _coerce_source_bundle(source_bundle)
    phrases = _extract_seed_phrases(normalized_bundle, metadata=metadata)
    if phrases:
        primary = phrases[:2]
        if len(primary) == 1:
            base = _title_case_phrase(primary[0])
            if len(base.split()) >= 2:
                return base
            return f"{base} {_KIND_SUFFIX[kind_value]}"
        return _title_case_phrase(" and ".join(primary))
    return _FINAL_FALLBACK[kind_value]


def resolve_generation_title(
    *,
    kind: GenerationKind | str,
    source_bundle: SourceBundle | dict[str, Any] | None = None,
    provided_title: str | None = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> str:
    explicit_title = _title_case_phrase(provided_title or "")
    if explicit_title:
        return explicit_title

    normalized_bundle = _coerce_source_bundle(source_bundle)
    fallback_title = build_fallback_title(kind=kind, source_bundle=normalized_bundle, metadata=metadata)
    phrases = _extract_seed_phrases(normalized_bundle, metadata=metadata)
    if not phrases:
        return fallback_title

    summarizer = LLMOutputSummarizer(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if not summarizer.is_active:
        return fallback_title

    try:
        generated = summarizer.generate_title(
            "\n".join(phrases),
            label=GenerationKind(kind).value if not isinstance(kind, GenerationKind) else kind.value,
            fallback_title=fallback_title,
        )
    except Exception:
        logger.warning("LLM title generation failed", exc_info=True)
        return fallback_title

    normalized_title = _title_case_phrase(generated)
    return normalized_title or fallback_title
