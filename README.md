# Pipeline Service

FastAPI + Celery service that ingests PDFs, runs OCR + embedding + QA/tagging, and streams progress over websockets. Persistence supports SQLite or Postgres and publishes progress via Redis for live clients and reconnects.

## Quick Start
- Prereqs: Docker + Docker Compose, `.env` in project root (see `.env` sample values).
- Start services (API optional): `docker compose up -d api worker redis db`
  - To run without the API: `docker compose up -d worker redis db`
- API listens on `http://localhost:8080` (ws: `ws://localhost:8080/ws/progress/<job_id>`).
- Monitoring: Flower at `http://localhost:5555` and Prometheus at `http://localhost:9090` (basic auth `admin` / `anko2025`). Celery metrics are exposed via the `celery-exporter` target scraped by Prometheus.

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
   |         | SQLite /      |  knowledge     | notifications     |
   |         | Postgres      |  store (emb/QA)| & tags tables     |
   |         +---------------+                +-------------------+
```
- FastAPI accepts process requests, derives deterministic `job_id`s, pushes initial progress, and exposes a websocket that streams pubsub events plus Redis snapshots for reconnects.
- Celery worker runs the pipeline: validate → OCR → embedding → persist → tag. Progress is emitted to Redis throughout.
- Storage can be SQLite or Postgres; embeddings/chunks/qa_pairs, tags, and notifications (job-level completion metadata) are persisted for durability and replay.
- Redis is used for live progress (pubsub) and latest snapshot per `job_id`.

## Tables & Relationships
- `documents (document_id PK)`: uploaded source; `chunks`, `qa_pairs`, and `tags` reference this.
- `chunks (document_id FK -> documents, chunk_index PK)`: embedding text + vectors; `chunk_id` unique; `question_ids` tracks generated QA per chunk.
- `qa_pairs (document_id FK -> documents, qa_index PK)`: questions/answers + `job_id` and optional `chunk_id/chunk_index` linkage to `chunks`.
- `tags (document_id FK -> documents, document_id+tag PK)`: final tag set for a document.
- `notifications (job_id PK)`: durable progress snapshots per job (status, step, progress, metadata). Websocket reconnects/DB queries read from here.
- Flashcards:
  - `flashcard_jobs (job_id PK, user_id, requested_new, status, error, created_at/updated_at)`
  - `flashcards (card_id PK, FK -> flashcard_jobs.job_id, user_id, front/back, tags, status, learning_step_index, repetition, interval_days, ease_factor, due_at, timestamps)`
  - `flashcard_reviews (id PK, card_id FK, user_id, job_id, rating, time_to_answer_ms, notes, created_at)`

## Key Components
- `service_app.py`: FastAPI endpoints (`/process-request`, `/ws/progress/{job_id}`, `/flashcards/create`, `/flashcards/learn/{job_id}`, `/ws/flashcards/{job_id}`), job id derivation, progress snapshots from Redis.
- `pipeline/workflow/celery_pipeline.py`: Celery chain: validate → OCR → embedding → persist → tag.
- `pipeline/celery_tasks/*`: Individual Celery tasks (OCR, embedding, tagging, persistence, flashcard generation).
- `pipeline/workflow/progress.py`: Emits progress to Redis hash + pubsub for snapshots/reconnects.
- Storage: `pipeline/db/storage.py` (SQLite/SQLAlchemy) and `pipeline/workflow/postgres_storage.py` (Postgres) manage documents, chunks, QA pairs, plus `notifications` and `tags` tables for durable completion state. Flashcards use `flashcard_jobs`, `flashcards`, `flashcard_reviews`.
- QA: `process_pdf` always skips QA generation; use the `generate_question` process to create questions for an existing document. Flashcards use `/flashcards/create` + `/flashcards/learn/{job_id}` + `/ws/flashcards/{job_id}` with an Anki-like SRS (learning steps 1m/10m, ratings 0/1/2).

## Examples
- `examples/process_request_client.py`: Submit a process request (uses env defaults like `PROCESS_REQUEST_BASE_URL`, `PROCESS_REQUEST_FILE_PATH`).
- `examples/ws_progress_client.py`: Follow websocket progress for a `job_id`.
- `examples/generate_questions_client.py`: Example generate-question request.
- Flashcards: `examples/create_flashcards_job.py` (create 10 Barcelona cards for doc `test`), `examples/learn_flashcards_job.py` (learn via websocket, interactive ratings), `examples/flashcards_ws_client.py` (generic WS client).
- `examples/check_db_connection.py`: Verify DB connectivity and list/preview tables.
- `examples/truncate_tables.py`: Truncate `documents`, `chunks`, `qa_pairs` (override via `TRUNCATE_TABLES`).

## Job IDs & Idempotency
- For generate-question requests, job id is derived deterministically from `doc_id`, `process`, `theme`, `question_format`, `tags`, and `query_text` (sorted) to avoid duplicate work for identical requests. A provided `job_id` overrides this.

## Progress & Notifications
- Live updates: Redis pubsub channel `progress:<job_id>`.
- Snapshots: Redis hash `job:<job_id>` sent on websocket connect/heartbeat.
- Durable markers: `notifications` table stores completion metadata by `job_id`; `tags` table records final tags per document.

## Environment
Common variables (see `.env`):
- `DB_URL` (or `DB_USER/DB_PASSWORD/DB_HOST/DB_PORT/DB_NAME`) for storage backend.
- `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `PROGRESS_REDIS_URL`.
- `OCR_WORKERS`, `QA_WORKERS`, `VECTOR_BATCH_SIZE`, `OPENAI_API_KEY`, `OPENAI_MODEL`.
- Chunk sizing (to control LLM call volume/quality trade-off): `MAX_CHUNK_TOKENS` (default 320), `MIN_CHUNK_TOKENS` (default 40), `CHUNK_OVERLAP` (default 24).
- Optional Supabase mount controls: `SUPABASE_*`, `SUPABASE_MOUNT_ENABLED`.
