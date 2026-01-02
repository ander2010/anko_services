# Spaced Repetition Scheduler Notes

This repo currently focuses on ingestion/OCR/QA generation. To make it usable as a spaced-repetition system (like Anki), we need a scheduler to decide when each card is due. Two widely used approaches:

- **SM-2 (classic Anki)** — simple, deterministic, compatible with Anki’s default behavior.
  - State per card: `EF` (easiness factor), `interval` (days), `repetitions`.
  - Review input: 0–5 grade (fail=0–2, pass=3–5).
  - Update rules (per review):
    - If grade < 3: `repetitions = 0`, `interval = 1` (a lapse).
    - Else: `repetitions += 1`; if `repetitions` is 1 ⇒ `interval=1`; if 2 ⇒ `interval=6`; else `interval = interval * EF` (rounded).
    - Adjust EF: `EF = EF + (0.1 - (5 - grade)*(0.08 + (5 - grade)*0.02))`; clamp EF ≥ 1.3. Higher grades nudge EF up; lower grades push it down.
  - Pros: tiny state, predictable, works offline, easy to export/import as .apkg. Cons: does not adapt well to irregular gaps; “easy”/“hard” signals are coarse.

- **FSRS (Free Spaced Repetition Scheduler)** — a modern, data-driven model.
  - State per card: `stability` (memory half-life), `difficulty` (perceived hardness), plus timestamps (`last_review`, `due`). Many implementations also track lapses and a per-user/global parameter set.
  - Core idea: retention decays along a forgetting curve; reviews change stability and difficulty. “Again” shrinks stability; “good/easy” boost it; difficulty moves with user feedback.
  - Intervals are derived from stability: longer stability ⇒ longer next interval. Irregular review gaps are handled explicitly by factoring elapsed time into the update.
  - Pros: better retention/time tradeoff, graceful under missed reviews, tunable on real logs. Cons: more math/state; needs consistent timestamp handling and maybe parameter fitting.

## What “difficulty increases on good” means
- In a strict SM-2, “good” usually leaves EF roughly stable or slightly up. If you want “good” to *increase* difficulty, you’re departing from SM-2 and moving toward a custom rule (e.g., nudging difficulty upward so the model stretches intervals more cautiously).
- In an FSRS-style approach, you can explicitly raise `difficulty` after a “good” grade while still boosting `stability`, leading to moderate interval growth but acknowledging the card feels harder. This behavior should be documented and tested so it’s intentional, not accidental.

## Suggested data model (minimum)
- Per card: `id`, `difficulty`, `stability`, `last_review_at`, `due_at`, `interval_days`, `lapses`, `repetitions`.
- Optionally per user/profile: global parameters for FSRS tuning or SM-2 overrides.

## API/UX flow to bolt onto this service
- **Queue**: `/reviews/next?limit=n` returns due cards ordered by `due_at`.
- **Submit**: `/reviews/{card_id}/grade` accepts `grade` (`again/hard/good/easy` or 0–5), runs the scheduler update, writes new `due_at`, `difficulty`, `stability`, `interval_days`, and appends a review log row.
- **UI mapping**: four buttons (Again/Hard/Good/Easy) or the 0–5 numeric scale; keep it consistent with the scheduler you pick.

## Implementation guidance
- Pick one path: SM-2 for compatibility/simplicity; FSRS for adaptivity. Mixing rules (e.g., “good increases difficulty”) is fine but document it clearly.
- Keep timestamps timezone-aware (UTC) and store both `last_review_at` and `due_at`.
- Persist updates transactionally with the review log so you never lose history if a write fails.
- Add tests that assert your chosen behaviors (e.g., “good” increases difficulty, “again” shrinks stability, intervals don’t go below 1 day, difficulty stays within bounds).
- When ready, export/import: for Anki parity, stick to SM-2 fields or provide an export step that flattens your richer state to Anki-compatible intervals and ease.

## SAT (Stability–Activation–Threshold) concept
If you’ve seen “SAT” in modern SuperMemo discussions, it is a way to pick the next question and update memory state using three pieces:
- **Stability (S):** how long the memory is expected to last (like FSRS stability). Higher S → slower forgetting.
- **Activation (A):** current recall probability, derived from stability and elapsed time since last review via a forgetting curve. Often `A = exp(-t / S_eff)` where `t` is elapsed time; variants include difficulty scaling.
- **Threshold (T):** target recall probability at which an item should be reviewed (e.g., 0.9). When activation drops below T, the item is due.

How SAT drives scheduling:
- **Queue selection:** at “present time,” compute activation for each card from its stability and elapsed time; pick cards whose activation < T (or the lowest activations first) as the next questions.
- **On review:** adjust stability up or down depending on the outcome. Successful recalls raise stability; failures reduce it. Difficulty can modulate the size of the change.
- **History tracking:** you need `last_review_at`, `stability`, `difficulty` (optional), and a per-profile target threshold `T`. Activation is recalculated on the fly from history rather than stored.

Practical mapping to this project:
- Store per card: `stability`, `difficulty`, `last_review_at`, and optionally `target_recall` (the threshold).
- At dequeue time: compute `activation = exp(-elapsed_days / (stability / f(difficulty)))`; if `activation < target_recall`, the card is due. Sort by activation ascending.
- At grading time: if fail, shrink stability (e.g., multiply by 0.5–0.8); if pass, increase stability proportionally to current stability and grade strength. Update `last_review_at`.
- This keeps selection logic (who is due) and update logic (how S changes) explicit and history-driven, similar in spirit to FSRS but framed around a target threshold.
