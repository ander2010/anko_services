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
  s3fs "$bucket" "$mount_path" \
    -o passwd_file="$passwd_file" \
    -o url="${endpoint}" \
    -o use_path_request_style \
    -o allow_other \
    -o nonempty \
    -o enable_noobj_cache \
    -o max_conns=20 \
    -o parallel_count=16 \
    -o multipart_size=64 \
    -o dbglevel=info \
    -o use_cache=/tmp/s3fs-cache \
    -o retries=3 \
    -o endpoint="${region}"
}

maybe_mount_supabase
exec "$@"
