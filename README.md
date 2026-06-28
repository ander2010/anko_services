# Pipeline Service

FastAPI + Celery service that ingests PDFs, runs OCR + embedding + QA/tagging, and streams progress over websockets. Persistence supports SQLite or Postgres and publishes progress via Redis for live clients and reconnects.

## Quick Start
- Prereqs: Docker + Docker Compose, `.env` in project root (see `.env` sample values).
- Start services (API optional): `docker compose up -d api worker redis db`
  - To run without the API: `docker compose up -d worker redis db`
- API listens on `http://localhost:8080` (ws: `ws://localhost:8080/ws/progress/<job_id>`).
- Monitoring: Flower at `http://localhost:5555` and Prometheus at `http://localhost:9090` (basic auth `admin` / `anko2025`). Celery metrics are exposed via the `celery-exporter` target scraped by Prometheus.

## API input notes
- `/ask` and `/ws/chat/{session_id}` expect `context` to be an array of integers (document IDs). Non-integer values (strings, bools, nulls) return a validation error; send an empty list when no context is desired.
- Progress payloads continue to echo the document IDs you provided; downstream joins still use stringified IDs internally.
- The example `ask_client` no longer supplies a default context. Set `ASK_CONTEXT` to a comma-separated integer list when you want document-scoped retrieval; omit it to force direct answering.
- Tagging uses an LLM document summary plus an embedding-centroid filter at the end of the pipeline. Hugging Face access is required to pull the model `sentence-transformers/all-MiniLM-L6-v2` on first run; set `HF_HOME`/`HF_HUB_OFFLINE=1` if running offline. Summary limits are tunable via settings: `summary_max_chunks`, `summary_max_chars`, `summary_min_chunk_chars`, `summary_max_words`, `summary_min_tags`, `summary_max_tags`.

## Deployment notes (Nginx + subpaths)
- Flower and Prometheus are configured to live under `/flower/` and `/prometheus/` (see `docker-compose.yml` flags `--url-prefix=/flower` and `--web.external-url=... --web.route-prefix=/prometheus`). If your public host is plain HTTP, keep `http://yourdomain` in `--web.external-url`; if you terminate TLS, use `https://yourdomain`.
- Example Nginx server block (no rewrites; preserves prefixes):
  ```
  upstream flower_app      { server 127.0.0.1:5555; }
  upstream prometheus_app  { server 127.0.0.1:9090; }

  server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location /flower/ {
      proxy_pass http://flower_app;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /prometheus/ {
      proxy_pass http://prometheus_app;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
    }
  }
  ```
- If adding HTTPS, create a 443 server with your certs (e.g., via Certbot `certbot --nginx -d yourdomain.com -d www.yourdomain.com --redirect`) and update `--web.external-url` to use `https://`.

## Architecture
```
          +-------------+        enqueue        +-------------------+
HTTP/WS   |   FastAPI   | --------------------> | Celery (worker)   |
clients   |  service_app|                       | validate/ocr/     |
   |      +------+------+                       | embedding/persist/|
   |             |  ^                           | tagging           |
   |             |  |                           +---------+---------+
   |    progress |  | snapshots                           |
   |             v  |                                     |
   |      +---------+---------+                 +---------v---------+
   |      |    Redis          | <---- pubsub ---| progress emitters |
   |      |  hash + pubsub    |                 | (tasks)           |
   |      +---------+---------+                 +---------+---------+
   |                |                                   |
   |                | durable artifacts                 |
   |                v                                   v
   |         +------+--------+                +---------+---------+
   |         | SQLite /      |  knowledge     | sections/tags     |
   |         | Postgres      |  store (emb/QA)| table             |
   |         +---------------+                +-------------------+
```
- FastAPI accepts process requests, derives deterministic `job_id`s, pushes initial progress, and exposes a websocket that streams pubsub events plus Redis snapshots for reconnects.
- Celery worker runs the pipeline: validate → OCR → embedding → persist → tag. Progress is emitted to Redis throughout.
- Storage can be SQLite or Postgres; embeddings/chunks/qa_pairs are persisted for durability and replay. Tag output is stored in `sections` (title/content = tag, plus job_id).
- Redis is used for live progress (pubsub) and latest snapshot per `job_id`.

## Tables & Relationships
- Documents: `api_document` (Django PK `id` integer). Referenced by chunks/QA/sections/flashcards.
- Chunks: typically `chunks` (or Django `api_chunk` if you mirror), FK to document, unique `(document_id, chunk_index)`, stores text, embedding, metadata, question_ids.
- QA pairs: typically `qa_pairs` (or Django `api_qapair`), FK to document, unique `(document_id, qa_index)`, stores question/answer/context/metadata + job_id + chunk_id/index.
- Sections: `api_section`, FK to document, stores tags as `title`/`content`, plus `job_id` and `order` (no separate `tags` table).
- Flashcards: `api_flashcard` (card_id PK, user_id, job_id, front/back, deck_id BIGINT, notes, source_doc_id, tags JSON, SRS fields); reviews in `api_flashcardreview`.

## Key Components
- `service_app.py`: FastAPI endpoints (`/process-request`, `/batteries/create`, `/ws/progress/{job_id}`, `/flashcards/create`, `/ws/flashcards/{job_id}`), job id derivation, progress snapshots from Redis, and battery finalization callbacks.
- `pipeline/workflow/celery_pipeline.py`: Celery chain: validate → OCR → embedding → persist → tag.
- `pipeline/celery_tasks/*`: Individual Celery tasks (OCR, embedding, tagging, persistence, flashcard generation).
- `pipeline/workflow/progress.py`: Emits progress to Redis hash + pubsub for snapshots/reconnects.
- Storage: `pipeline/db/storage.py` (SQLite/SQLAlchemy) and `pipeline/workflow/postgres_storage.py` (Postgres) manage documents, chunks, QA pairs, and `sections` tags. Flashcards use `flashcards`, `flashcard_reviews`.
- QA: `process_pdf` always skips QA generation; use `/batteries/create` with a `source_bundle` to create multi-document question sets. Flashcards use `/flashcards/create` + `/ws/flashcards/{job_id}` with an Anki-like SRS (learning steps 1m/10m, ratings 0/1/2).

## Examples
- `examples/process_request_client.py`: Submit a process request (uses env defaults like `PROCESS_REQUEST_BASE_URL`, `PROCESS_REQUEST_FILE_PATH`).
- `examples/ws_progress_client.py`: Follow websocket progress for a `job_id`.
- `examples/generate_questions_client.py`: Example battery-generation request using the typed `source_bundle` payload.
- Flashcards: `examples/create_flashcards_job.py` (typed `source_bundle` request), `examples/learn_flashcards_job.py` (learn via websocket, interactive ratings), `examples/flashcards_ws_client.py` (generic WS client).
- `examples/check_db_connection.py`: Verify DB connectivity and list/preview tables.
- `examples/truncate_tables.py`: Truncate `documents`, `chunks`, `qa_pairs` (override via `TRUNCATE_TABLES`).

## Job IDs & Idempotency
- For battery-generation requests, job id is derived deterministically from the normalized source bundle, question settings, and prompt version. A provided `job_id` overrides this.

## Progress
- Live updates: Redis pubsub channel `progress:<job_id>`.
- Snapshots: Redis hash `job:<job_id>` sent on websocket connect/heartbeat.
- Durable markers: `sections` stores final tags per document.

## Environment
Common variables (see `.env`):
- `DB_URL` (or `DB_USER/DB_PASSWORD/DB_HOST/DB_PORT/DB_NAME`) for storage backend.
- `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `PROGRESS_REDIS_URL`.
- Celery broker hardening:
  - `CELERY_VALIDATE_BROKER_ON_STARTUP=1` makes startup fail fast if `CELERY_BROKER_URL` points at a read-only Redis replica.
  - `CELERY_BROKER_STARTUP_CHECK_TIMEOUT` controls that probe timeout in seconds (default `5`).
  - `CELERY_WORKER_CANCEL_LONG_RUNNING_TASKS_ON_CONNECTION_LOSS` controls the Celery 5.x connection-loss behavior ahead of the Celery 6 default switch.
  - `CELERY_WORKER_ENABLE_REMOTE_CONTROL` and `CELERY_WORKER_SEND_TASK_EVENTS` expose Celery control-plane toggles explicitly.
  - `CELERY_REDIS_VISIBILITY_TIMEOUT` is passed through to Redis transport options when you need a custom visibility timeout.
  - `CELERY_WORKER_EXTRA_ARGS` and `CELERY_OCR_WORKER_EXTRA_ARGS` append raw worker CLI flags from compose. If Redis reconnects still trigger `mingle`/pidbox crashes, use `--without-mingle --without-gossip` here as a targeted workaround.
- `OCR_WORKERS`, `QA_WORKERS`, `VECTOR_BATCH_SIZE`, `OPENAI_API_KEY`, `OPENAI_MODEL`.
- Logging controls: `LOG_LEVEL`, `LOG_TO_STDOUT`, `LOG_TO_FILE`, `LOG_PATH` (or `LOG_DIR` + `LOG_FILE_NAME`), `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`.
- Chunk sizing (to control LLM call volume/quality trade-off): `MAX_CHUNK_TOKENS` (default 320), `MIN_CHUNK_TOKENS` (default 40), `CHUNK_OVERLAP` (default 24).
- Optional Supabase mount controls: `SUPABASE_*`, `SUPABASE_MOUNT_ENABLED`.
