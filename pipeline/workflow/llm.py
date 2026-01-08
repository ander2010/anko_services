from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, List, Optional, Sequence

from openai import OpenAI

from pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)


class LLMChunkAssessment:
    def __init__(self, relevance: bool, importance: float, concept_type: str, tags: Sequence[str], difficulty: str) -> None:
        self.relevance = relevance
        self.importance = importance
        self.concept_type = concept_type
        self.tags = list(tags)
        self.difficulty = difficulty


class LLMImportanceClient:
    """Wraps an OpenAI client to request structured evaluations for each chunk.

    If no API key is provided, it falls back to a dummy key and remains inactive.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", dummy_key: str = "sk-dummy") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", dummy_key)
        self.model = model
        self.dummy_key = dummy_key
        self._client: Optional[OpenAI] = None
        if self.api_key and self.api_key != self.dummy_key:
            self._client = OpenAI(api_key=self.api_key)

    @property
    def is_active(self) -> bool:
        return self._client is not None

    def assess_chunk(self, text: str, page: int) -> LLMChunkAssessment:
        if not self.is_active:
            raise RuntimeError("LLM client is not configured with a valid API key.")

        prompt = (
            "You are assisting with exam or study preparation. For the provided chunk of material, "
            "evaluate whether it is useful for studying or learning from this document. "
            "Rate how important this chunk is for understanding or preparing based on the document itself "
            "on a 0–5 scale (0 = not useful, 5 = essential to learn from this document). "
            "Identify the content type using a neutral classification that works for any document "
            "(Statement, Description, Explanation, Example, Narrative, Instruction, Reference, or Other). "
            "Assign a difficulty label (easy, medium, hard) based on how challenging it is to understand, "
            "and provide up to three topic tags derived strictly from the content.\n"
            "Respond with strict JSON using the following schema:\n"
            '{\n'
            '  "is_exam_relevant": true,\n'
            '  "importance": 4.5,\n'
            '  "content_type": "Narrative",\n'
            '  "difficulty": "medium",\n'
            '  "tags": ["lyrics", "theme"]\n'
            "}"
        )


        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Page: {page}\nChunk:\n{text}"},
                ],
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        data = self._parse_response(content)
        return LLMChunkAssessment(
            relevance=data.get("is_exam_relevant", True),
            importance=float(data.get("importance", 2.5)),
            concept_type=data.get("concept_type", "Explanation"),
            tags=data.get("tags", []) or [],
            difficulty=data.get("difficulty", "medium"),
        )

    @staticmethod
    def _parse_response(content: str) -> dict:
        return _extract_json(content)


class QAFormat(str, Enum):
    VARIETY = "variety"
    TRUE_FALSE = "true_false"
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"

    @classmethod
    def from_value(cls, value: Optional[str | "QAFormat"]) -> "QAFormat":
        if isinstance(value, cls):
            return value
        normalized = (value or cls.VARIETY.value).lower()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.VARIETY


QA_FORMAT_CHOICES = tuple(fmt.value for fmt in QAFormat)


class LLMQuestionGenerator:
    """Creates varied assessment questions (true/false, single, multi select) for a chunk."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", dummy_key: str = "sk-dummy") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", dummy_key)
        self.model = model
        self.dummy_key = dummy_key
        self._client: Optional[OpenAI] = None
        if self.api_key and self.api_key != self.dummy_key:
            self._client = OpenAI(api_key=self.api_key)

    @property
    def is_active(self) -> bool:
        return self._client is not None

    def generate(
        self,
        *,
        text: str,
        page: int,
        tags: Optional[List[str]] = None,
        max_answer_words: int = 60,
        mode: str | QAFormat = QAFormat.VARIETY,
        theme_hint: Optional[str] = None,
        difficulty_hint: Optional[str] = None,
        target_questions: Optional[int] = None,
    ) -> List[dict]:
        if not self.is_active:
            raise RuntimeError("LLM question generator is not configured with a valid API key.")

        qa_mode = QAFormat.from_value(mode)
        topic_hint = theme_hint or (", ".join(tags or []) or "general knowledge")
        target_count = target_questions or 0
        difficulty_text = f"Target difficulty: {difficulty_hint}." if difficulty_hint else "Difficulty: mixed."
        quantity_text = (
            f"Return no more than {target_count} high-quality questions (fewer is fine if the content is thin)." if target_count > 0 else "Produce as many high-quality questions as the content supports."
        )

        type_instruction_map = {
            QAFormat.VARIETY: "Produce a mix of true/false, single-select, and multi-select items.",
            QAFormat.TRUE_FALSE: "Generate only true/false questions.",
            QAFormat.SINGLE_CHOICE: "Generate only single-select questions (exactly one correct option).",
            QAFormat.MULTIPLE_CHOICE: "Generate only multi-select questions (two or more correct options).",
        }
        required_type_map = {
            QAFormat.VARIETY: "true_false | single_select | multi_select",
            QAFormat.TRUE_FALSE: "true_false",
            QAFormat.SINGLE_CHOICE: "single_select",
            QAFormat.MULTIPLE_CHOICE: "multi_select",
        }

        type_instruction = type_instruction_map[qa_mode]
        required_type = required_type_map[qa_mode]

        prompt = (
            "You transform arbitrary text into HIGH-QUALITY study questions for LEARNING and understanding.\n"
            "PRIMARY GOAL:\n"
            "Help the learner acquire and retain domain knowledge.\n"
            "Questions must teach concepts, rules, and reasoning - NOT test familiarity with the source text.\n"
            "MANDATORY LEARNING VALUE GATE (NON-NEGOTIABLE):\n"
            "Before generating ANY question, decide whether the chunk contains at least one stand-alone concept that a learner could understand and retain without seeing the source.\n"
            'If the chunk does NOT contain such knowledge, return EXACTLY: {"questions": []}\n\n'
            "ABSOLUTE PROHIBITIONS (HARD FAIL):\n"
            "You MUST NOT generate questions that:\n"
            "- Refer to a document, text, chunk, passage, or source\n"
            "- Ask what is 'mentioned', 'stated', 'listed', or 'according to' anything\n"
            "- Test recall of wording, examples, or narrative context\n"
            "- Depend on formatting, section numbers, tables, figures, or editorial structure\n\n"
            "INVALID QUESTION PATTERNS (NEVER PRODUCE):\n"
            "- \"Which of the following is mentioned...\"\n"
            "- \"According to the text..\"\n"
            "- \"Which statements about the document are true?\"\n"
            "- \"What does this section say about...?\"\n\n"
            "MANDATORY CONCEPT EXTRACTION STEP:\n"
            "Before writing a question, internally extract a GENERAL, DOMAIN-LEVEL FACT, RULE, OR PROCESS expressed in the chunk.\n"
            "Write the question to test understanding or correct application of that knowledge.\n\n"
            "LEARNING VALIDITY TEST:\n"
            "A learner who has never seen this text must still find the question clear, answerable, and educational.\n"
            "If this test fails, DO NOT generate the question.\n\n"
            "QUESTION DESIGN RULES:\n"
            "- Each question must focus on exactly ONE concept\n"
            "- Focus on definitions, rules, constraints, processes, conditions, classifications, or cause-effect relationships\n"
            "- Prefer questions that reinforce correct mental models\n\n"
            f"{type_instruction}\n\n"
            f"{difficulty_text}\n"
            f"{quantity_text}\n\n"
            "CONSTRUCTION RULES:\n"
            "1. true_false -> exactly 2 options ['True', 'False']; exactly 1 correct answer\n"
            "2. single_select -> 3-4 unique options; exactly 1 correct answer\n"
            "3. multi_select -> 3-5 unique options; at least 2 correct answers\n\n"
            "EVERY QUESTION MUST:\n"
            "- Be self-contained and unambiguous\n"
            "- Be phrased as general knowledge, not source comprehension\n"
            "- Help the learner build or test understanding\n\n"
            "REQUIRED FIELDS:\n"
            '- "type"\n'
            '- "question"\n'
            '- "options"\n'
            '- "correct_answer" (array of option strings)\n'
            f'- "explanation" (<= {max_answer_words} words, explaining WHY the answer is correct)\n\n'
            "FACTUAL CONSTRAINT:\n"
            "Use ONLY facts explicitly present in the chunk.\n"
            "Do NOT invent terminology, values, or implications.\n\n"
            "OUTPUT FORMAT (STRICT JSON ONLY):\n"
            "{\n"
            '  "questions": [\n'
            "    {\n"
            f'      "type": "{required_type}",\n'
            '      "question": "...",\n'
            '      "options": [...],\n'
            '      "correct_answer": [...],\n'
            '      "explanation": "..."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "The source text is CONTEXT ONLY - it must never be referenced directly."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Page: {page}\nTopics: {topic_hint}\nRequested question mode: {qa_mode.value}\nChunk:\n{text}"},
                ],
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI QA generation failed: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        data = _extract_json(content)
        questions = data.get("questions") or []

        normalized: List[dict] = []
        banned_phrases = {"the data type", "the type mentioned", "the passage", "the excerpt", "the section mentioned"}

        for item in questions:
            question_text = (item.get("question") or "").strip()
            lowered = question_text.lower()
            if len(question_text.split()) < 6 or any(phrase in lowered for phrase in banned_phrases):
                continue

            answers = item.get("correct_answer") or item.get("answers") or []
            options = item.get("options") or []
            explanation = (item.get("explanation") or "").strip()
            q_type = (item.get("type") or "true_false").strip()

            if not answers:
                continue

            if q_type == "true_false":
                options = ["True", "False"]
                answers = [a for a in answers if a in options]
                if len(answers) != 1:
                    continue
            else:
                options = [str(opt).strip() for opt in options if str(opt).strip()]
                options = list(dict.fromkeys(options))
                if q_type == "single_select" and len(options) < 3:
                    continue
                if q_type == "multi_select" and len(options) < 3:
                    continue
                answers = [a for a in answers if a in options]
                if q_type == "single_select" and len(answers) != 1:
                    continue
                if q_type == "multi_select" and len(answers) < 2:
                    continue

            normalized.append(
                {
                    "type": q_type,
                    "question": question_text,
                    "options": options,
                    "answers": [str(answer).strip() for answer in answers if str(answer).strip()],
                    "explanation": explanation,
                }
            )

        return normalized

    def tag_text(self, text: str, *, max_tags: int = 5) -> List[str]:
        """Generate concise tags for a chunk; returns empty when inactive."""
        if not self.is_active:
            return []
        prompt = (
            "You are assigning semantic tags for retrieval. Follow these rules:\n"
            "- 3-5 tags, lower-case noun phrases (1-3 words)\n"
            "- Reflect the main concepts/topics; ignore formatting, bullets, page numbers\n"
            "- Avoid generic terms (page, section, document, text, list, introduction)\n"
            "- No verbs/imperatives; no duplicates; no punctuation beyond hyphens\n"
            "- If no meaningful tags exist, return an empty array\n"
            "- Respond with a strict JSON array of strings only\n\n"
            f"Text: {text[:2000]}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=96,
            messages=[
                {"role": "system", "content": "Return ONLY a JSON array of tag strings."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "[]"
        data = _extract_json(content)
        if isinstance(data, list):
            return [str(tag) for tag in data if tag][:max_tags]
        return []


class LLMFlashcardGenerator:
    """Generates concise flashcards (front/back) for spaced repetition."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", dummy_key: str = "sk-dummy") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", dummy_key)
        self.model = model
        self.dummy_key = dummy_key
        self._client: Optional[OpenAI] = None
        if self.api_key and self.api_key != self.dummy_key:
            self._client = OpenAI(api_key=self.api_key)

    @property
    def is_active(self) -> bool:
        return self._client is not None

    def generate(
        self,
        *,
        topics: Sequence[str] | None = None,
        documents: Sequence[str] | None = None,
        difficulty: str | None = None,
        count: int = 1,
        avoid_fronts: Sequence[str] | None = None,
        prompt_context: str | None = None,
    ) -> list[dict[str, str]]:
        if not self.is_active:
            raise RuntimeError("LLM flashcard generator is not configured with a valid API key.")
        if count <= 0:
            return []

        topic_line = ", ".join(topics or []) or "general study topics"
        doc_line = ", ".join(documents or []) or "the provided materials"
        difficulty_hint = difficulty or "medium"

        system_prompt = (
            "You are an expert learning designer creating high-quality flashcards for spaced repetition.\n"
            "Your objective is to produce cards that maximize long-term retention, understanding, and transfer across ANY subject domain.\n\n"

            "UNIVERSAL LEARNING PRINCIPLES (MANDATORY):\n"
            "- Each flashcard must teach EXACTLY ONE atomic idea.\n"
            "- Cards must be self-contained and understandable without external context.\n"
            "- Prefer understanding, application, or discrimination over memorization.\n"
            "- Avoid trivia, rote facts, or list-based recall.\n\n"

            "FRONT (QUESTION) DESIGN:\n"
            "- Ask a precise, self-contained question that reveals true understanding of the idea.\n"
            "- Choose the question form dynamically (why, how, when, what distinguishes, how would you apply).\n"
            "- Avoid yes/no questions and vague prompts.\n"
            "- Do not reference sources, documents, or prior text.\n\n"

            "BACK (ANSWER) DESIGN:\n"
            "- Provide a clear, concise answer in 1–3 sentences.\n"
            "- Add one brief explanation or example ONLY if it improves comprehension.\n"
            "- Do not restate the question or include unnecessary qualifiers.\n\n"

            "ADAPTIVE CONTENT SELECTION:\n"
            "- Identify concepts that are stable, transferable, and worth remembering.\n"
            "- Skip concepts that cannot be expressed as strong atomic flashcards.\n"
            "- Avoid generating multiple cards that test the same idea in different wording.\n\n"

            "COGNITIVE OPTIMIZATION:\n"
            "- Minimize cognitive load.\n"
            "- Maximize discriminative power (the card clearly separates understanding from guessing).\n"
            "- Prefer durable knowledge over short-term recall.\n\n"

            f"DIFFICULTY CALIBRATION:\n"
            f"- Target difficulty: {difficulty_hint}.\n"
            "- Adjust abstraction level, terminology, and examples dynamically to match this difficulty.\n\n"

            "OUTPUT CONTRACT (STRICT):\n"
            'Return ONLY valid JSON with this exact structure:\n'
            '{"cards": [{"front": "string", "back": "string"}]}.\n'
            "- No markdown, no commentary, no extra fields.\n"
        )


        avoid_list = [str(f).strip() for f in (avoid_fronts or []) if str(f).strip()]
        avoid_text = ""
        if avoid_list:
            avoid_text = "Do NOT generate cards that test the same idea as any of these fronts:\n- " + "\n- ".join(avoid_list) + "\n"

        clean_cards: list[dict[str, str]] = []
        base_avoid = [str(f).strip() for f in (avoid_fronts or []) if str(f).strip()]

        for _ in range(count):
            avoid_list = base_avoid + [c.get("front") for c in clean_cards[-8:]]
            avoid_text = ""
            if avoid_list:
                avoid_text = "Do NOT generate cards that test the same idea as any of these fronts:\n- " + "\n- ".join(avoid_list) + "\n"

            user_prompt = (
                "Generate a flashcard \n"
                "Ensure each card stands alone without context from the source.\n"
                f"{avoid_text}"
                f"{prompt_context or ''}"
            )

            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0.5,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as exc:
                raise RuntimeError(f"OpenAI flashcard generation failed: {exc}") from exc

            content = resp.choices[0].message.content or "{}"
            try:
                data = json.loads(content)
            except Exception:
                data = {}

            cards = data.get("cards") if isinstance(data, dict) else None
            if isinstance(cards, list):
                candidates = cards
            elif isinstance(data, dict) and "front" in data and "back" in data:
                candidates = [data]
            else:
                candidates = []

            for item in candidates:
                if not isinstance(item, dict):
                    continue
                front = str(item.get("front") or "").strip()
                back = str(item.get("back") or "").strip()
                if front and back:
                    clean_cards.append({"front": front, "back": back})
                    break  # move to next card request

        return clean_cards[:count]


def _extract_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            start = content.index("{")
            end = content.rindex("}")
            return json.loads(content[start : end + 1])
        except Exception:
            try:
                start = content.index("[")
                end = content.rindex("]")
                return json.loads(content[start : end + 1])
            except Exception:
                logger.warning("Failed to parse JSON content: %s", content)
                return {}
