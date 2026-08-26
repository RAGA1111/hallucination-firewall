FROM python:3.12-slim AS builder



ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


FROM python:3.12-slim

ARG BUILD_VERSION=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    BUILD_VERSION=$BUILD_VERSION

WORKDIR /app

COPY --from=builder /install /usr/local

COPY api ./api
COPY core ./core
COPY eval ./eval
COPY pipeline.py .
COPY requirements.txt .

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]