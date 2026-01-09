#!/usr/bin/env bash
set -euo pipefail

maybe_mount_supabase() {
  local enabled
  enabled="${SUPABASE_MOUNT_ENABLED:-true}"
  case "${enabled,,}" in
    ""|0|false|no|off) return 0 ;;
  esac

  local bucket="${SUPABASE_S3_BUCKET:-}"
  local endpoint="${SUPABASE_S3_ENDPOINT:-}"
  local access="${SUPABASE_S3_ACCESS_KEY:-}"
  local secret="${SUPABASE_S3_SECRET_KEY:-}"
  local region="${SUPABASE_S3_REGION:-us-west-2}"
  local mount_path="${SUPABASE_MOUNT_PATH:-/mnt/supabase}"

  if [[ -z "$bucket" || -z "$endpoint" || -z "$access" || -z "$secret" ]]; then
    echo "Supabase mount skipped: missing SUPABASE_S3_BUCKET/ENDPOINT/ACCESS_KEY/SECRET_KEY" >&2
    return 0
  fi

  mkdir -p "$mount_path"
  local passwd_file="/tmp/.supabase_s3_passwd"
  echo "${access}:${secret}" > "$passwd_file"
  chmod 600 "$passwd_file"

  # Avoid re-mounting if already mounted
  if mountpoint -q "$mount_path"; then
    echo "Supabase already mounted at $mount_path"
    return 0
  fi

  echo "Mounting Supabase bucket '${bucket}' to ${mount_path}"
  if ! s3fs "$bucket" "$mount_path" \
      -o passwd_file="$passwd_file" \
      -o url="${endpoint}" \
      -o use_path_request_style \
      -o allow_other \
      -o nonempty \
      -o enable_noobj_cache \
      -o multipart_size=64 \
      -o dbglevel=info \
      -o use_cache=/tmp/s3fs-cache \
      -o retries=3 \
      -o endpoint="${region}"; then
    echo "Supabase mount failed; continuing without mount" >&2
  fi
}

ensure_db_exists() {
  python - <<'PY'
import os
import psycopg
from psycopg import sql
from urllib.parse import urlparse

db_url = os.getenv("DB_URL")
if not db_url:
    host = os.getenv("POSTGRES_HOST", "hope-db")
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "")
    dbname = os.getenv("POSTGRES_DB", "anko") or "anko"
    db_url = f"postgresql://{user}:{password}@{host}:5432/{dbname}"

parsed = urlparse(db_url)
target_db = (parsed.path or "/").lstrip("/") or os.getenv("POSTGRES_DB", "anko") or "anko"
admin_db = os.getenv("POSTGRES_DB_ADMIN", "postgres")
admin_url = db_url.rsplit("/", 1)[0] + f"/{admin_db}"

try:
    with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (target_db,))
            if cur.fetchone():
                print(f"Database '{target_db}' already exists; skipping creation.")
            else:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
                print(f"Created database '{target_db}'.")
except Exception as exc:
    print(f"Warning: could not ensure database '{target_db}': {exc}")
PY
}

maybe_mount_supabase
ensure_db_exists
exec "$@"
