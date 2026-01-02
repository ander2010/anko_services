# Examples

Quick examples for invoking the service and following progress.

## Process PDF (ingestion)
1) Submit the request over HTTP:
   - `python examples/process_request_client.py`
   - Uses env defaults: `PROCESS_REQUEST_BASE_URL` (default `http://localhost:8080`), `PROCESS_REQUEST_FILE_PATH`, `PROCESS_REQUEST_DOC_ID`.
   - Returns a `job_id`.
2) Stream progress over websocket:
   - `python examples/ws_progress_client.py --job-id <job_id> --base-url http://localhost:8080`
   - If you disconnect, reconnect to the same job id; the server sends the latest snapshot from Redis.
3) When status is `COMPLETED`, outputs are stored in the DB:
   - `tags` table: tags for the processed document.
   - To inspect results, query `tags` using the `job_id` returned by the request.

## Generate Questions (existing document)
1) Submit the request:
   - `python examples/generate_questions_client.py`
   - Uses env defaults: `PROCESS_REQUEST_BASE_URL`; request includes `doc_id`, optional `query_text`, `tags`, `theme`, `question_format`, etc.
   - Job id is deterministic from `doc_id`, `process`, `theme`, `question_format`, `tags`, `query_text` to avoid duplicates for identical inputs.
2) Stream progress over websocket as above with the returned `job_id`.
3) On completion:
   - New/updated questions in `qa_pairs` (query by `job_id`).
   - Completion metadata also stored in `notifications` keyed by `job_id` for reconnect fallback.

## Chat / Direct Answer
- `python examples/ask_client.py --question "..." [--doc-id ...] [--base-url http://localhost:8080]`
- Sends a question to the `/ask` endpoint (uses RAG context when provided) and returns the answer payload.
- Environment defaults: `ASK_BASE_URL` (fallback to `PROCESS_REQUEST_BASE_URL`), optional `ASK_DOC_ID`, `ASK_SESSION_ID`, and `ASK_USER_ID`.

## Question Variants
- `python examples/generate_question_variants_client.py --question-id <id> [--quantity 10] [--difficulty medium] [--question-format variety]`
- Calls `/questions/{question_id}/variants` to enqueue variant generation. Defaults apply if omitted.

## Websocket + Fallback
- Progress: `/ws/progress/<job_id>` streams live pubsub and heartbeats; reconnecting clients get the latest snapshot from Redis.
  - Message types you’ll see:
    - `{"type": "snapshot", ...}`: sent once on connect with the latest Redis hash (progress/state so far).
    - `{"type": "progress", ...}`: emitted whenever a task publishes an update; includes `status`, `current_step`, `progress`, `step_progress`, and any `extra` fields (e.g., `chunk_index`, `tags`).
    - `{"type": "heartbeat", ...}`: sent periodically when no new updates arrive; payload echoes the current Redis hash so you can display stale-but-known progress.
  - Treat heartbeats as “no change yet” signals; progress messages are the live events to drive UI.
- Fallback: durable completion metadata is saved in `notifications` (by `job_id`) so you can query DB even if the websocket was missed and Redis was cleared.

## Flashcards:
  - HTTP: `POST /study/start` with `document_ids`, `tags`, `quantity`, `difficulty`, `user_id` → returns `job_id`, `token`, `ws_path`.
  - WS: connect to `/ws/flashcards`, send `{"message_type":"subscribe_job", "job_id", "user_id", "request_id", "last_seq", "token"}`.
  - Stream: server sends `card` messages; client replies with `card_feedback` (`rating`: 0=hard, 1=good, 2=easy; `time_to_answer_ms` optional). `-1` in the sample client closes the socket.
  - SRS: new/lapsed cards use Anki-like learning steps (1m/10m). Again=0 returns immediately; Good=1 advances step; Easy=2 graduates to review (days). Latency can downgrade a slow “good/easy”. After learning, SM-2 style day intervals apply.
  - Completion: `ack` per feedback, `idle` when waiting for due cards, `done` when no more due cards and job complete.
  - See `examples/flashcards_ws_client.py` for a full runnable loop (non-blocking input with a timeout).
