FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        redis-tools \
        redis-server \
        s3fs \
        fuse3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Allow configuring worker/batch knobs at runtime.
ENV OCR_WORKERS=4 \
    QA_WORKERS=4 \
    VECTOR_BATCH_SIZE=32

COPY . .

# Allow fuse mounts to use allow_other
RUN grep -q "user_allow_other" /etc/fuse.conf || echo "user_allow_other" >> /etc/fuse.conf

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
# Default to serving the API; override CMD for batch runs as needed.
CMD ["uvicorn", "service_app:app", "--host", "0.0.0.0", "--port", "8000"]
