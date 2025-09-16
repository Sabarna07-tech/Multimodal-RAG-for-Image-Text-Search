# Multimodal RAG SaaS Platform

## Overview

This project is a production-ready, multi-tenant Retrieval-Augmented Generation (RAG) platform. It lets each tenant ingest private data (PDFs and YouTube videos) and chat with the resulting knowledge base using Google's Gemini Pro Vision model. The stack centers on FastAPI, Celery, Redis, and ChromaDB with a lightweight HTML frontend for demos.

Key capabilities:

- **Tenant isolation** – every API key maps to its own text/image collections in ChromaDB.
- **Multimodal ingestion** – PDFs are chunked and embedded; YouTube transcripts/frames are processed asynchronously.
- **Cited answers** – chat responses return the retrieved metadata for transparency.
- **Operational hardening** – rate limiting, request logging, idempotent background jobs, configurable storage paths.

---

## Setup

### Requirements

- Python 3.9+
- Redis (for task queue + rate limiting cache)
- Google Gemini API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the example environment file and edit the values:

```bash
cp .env.example .env
```

Important variables:

- `GOOGLE_API_KEY` – Gemini API access.
- `API_KEYS` – JSON object mapping client API keys to tenant identifiers.
- `REDIS_URL` – Redis connection string (defaults to `redis://localhost:6379/0`).
- `CHROMA_DB_PATH`, `CHECKPOINT_DIR`, `INGEST_CACHE_DIR` – override storage paths if desired.

### Running the stack

Start Redis (via Docker or your package manager), then run the FastAPI app:

```bash
python main.py
```

In a separate shell, start the Celery worker for YouTube ingestion:

```bash
celery -A app.tasks worker --loglevel=info
```

The demo UI is served at http://localhost:8000. Supply one of the API keys configured in `.env`, upload PDFs, enqueue YouTube URLs, and chat with the indexed content.

---

## Application Components

```
+---------+      +--------------+      +--------------------+
|  User   | ---> |  FastAPI API | ---> |  Retrieval + LLM   |
+---------+      +--------------+      +--------------------+
      |                |                         |
      |                |                         +--> Gemini Pro Vision
      |                +--> Celery + Redis jobs
      |                +--> ChromaDB (tenant text/image collections)
      +--> Static HTML/JS client
```

- **FastAPI (`app/main.py`)** – authentication, rate limiting, ingestion endpoints, chat API, and static assets.
- **Celery worker (`app/tasks.py`)** – downloads YouTube content, embeds transcript/frames, and writes to the correct tenant collections.
- **Embeddings (`app/embedding/embedder.py`)** – SentenceTransformer + CLIP models for text/image encoding.
- **Retriever (`app/retrieval/retriever.py`)** – dense search with optional reranking via CrossEncoder.
- **Generator (`app/generation/generator.py`)** – wraps Gemini, assembles prompts with citations, and returns answers.
- **Vector storage (`app/vector_store/chroma_store.py`)** – persistent ChromaDB client with per-tenant collections.

---

## API Reference (excerpt)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/healthz` | GET | Liveness probe. |
| `/process-pdf/` | POST | Upload PDF (multipart). Immediately indexes text/images. |
| `/process-youtube/` | POST | Form body with `url`. Enqueues background ingestion job (202 Accepted with job id). |
| `/ingest/status/{job_id}` | GET | Poll Celery job status. |
| `/chat/` | POST | JSON `{"thread_id": str, "message": str}`. Returns answer + citations. |

All endpoints require `X-API-Key` header using a key defined in `API_KEYS`.

---

## Testing

Unit tests use FastAPI's TestClient with extensive mocking to avoid heavy model downloads. Install pytest (not part of runtime deps) and run:

```bash
pip install pytest
python -m pytest
```

---

## Development Notes

- The application falls back to an in-memory cache if Redis is unreachable during local dev, but production deployments should run Redis.
- YouTube ingestion persists sampled frames under `INGEST_CACHE_DIR/frames/<user>/<video_id>` so embeddings remain valid after temporary directories are cleaned up.
- The HTML demo polls ingestion status and surfaces citation metadata with each chat response. Integrations can call the same REST API directly.