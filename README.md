# Multimodal RAG SaaS Platform

> Interview-ready brief for a production RAG stack (FastAPI + Celery + LanceDB + Gemini Vision Pro).

---

## Architecture at a Glance

```
┌──────────┐    REST/WS    ┌────────────┐    tasks    ┌──────────────┐
│  Client  │ ────────────► │  FastAPI   │ ──────────► │   Celery     │
└──────────┘     │         └─────┬──────┘             └─────┬────────┘
                 │  rate-limit        │ schedules            │ writes
                 ▼                    ▼                      ▼
           Redis (keys)         LanceDB (text/img)     Notes SQLite
                 │                    │                      │
                 └────────────► Gemini Vision Pro ◄──────────┘
                                 (grounded answers)
```

### Pipeline Sequence

```
User → /process-pdf or /ingest-youtube → ingest.{pdf|youtube}
      → LanceDB index_text_nodes/index_image_nodes
      → retrieve.retrieve_text/retrieve_images (MiniLM + CLIP)
      → cross-encoder rerank + score fusion
      → ml.generate.generate_response (Gemini Vision Pro, citations)
      → JSON answer with traceable metadata
```

**Why this matters**

- **Bi-encoder recall**: MiniLM + CLIP embeddings provide wide, low-latency candidate recall.
- **Cross-encoder precision**: ms-marco MiniLM reranker sharpens top-k text evidence before fusion.
- **Transcript-first ingestion**: `app/ingest/youtube.py` prefers official transcripts, falling back to Whisper only when needed to cut latency/cost.
- **Scene-aware frames**: `app/ingest/frames.py` uses ffmpeg scene detection + phash dedup to capture salient visuals without overwhelming the index.
- **LanceDB vector store**: columnar, on-disk store with per-user collections and versioning; simple to ship, fast to query.
- **Gemini Vision Pro**: final grounding layer that reasons over both text snippets and referenced frames, with inline citations and abstention when similarity is low.

---

## Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/healthz` | GET | Liveness probe. |
| `/process-pdf/` | POST (multipart) | Ingest a PDF immediately (text + image indexing). |
| `/process-youtube/` | POST (form) | Enqueue YouTube ingestion; returns `202` with `job_id`. |
| `/ingest-youtube` | POST (JSON) | Same as above but JSON payload. |
| `/yt_status/{job_id}` | GET | Poll Celery task state + `% progress`. |
| `/videos/` | GET | List processed videos for the current API key (tenant). |
| `/videos/{video_id}/notes` | GET | Retrieve auto-generated notes. |
| `/videos/{video_id}/quiz` | GET | Retrieve generated quiz payload. |
| `/chat_pro` | POST | `{"message": "...", "video_id": Optional[str]}` → grounded answer + citations. |

Headers: `X-API-Key` must match a key defined in `API_KEYS`.

---

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `APP_NAME` | Display name for FastAPI docs/logging. |
| `API_KEYS` | JSON map `{api_key: tenant_id}`; drives rate limiting + storage isolation. |
| `MODEL_TEXT` | Text encoder (default `sentence-transformers/all-MiniLM-L6-v2`). |
| `MODEL_CLIP` | CLIP checkpoint for image/query embeddings. |
| `RERANKER_MODEL` | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). |
| `GEMINI_API_KEY` | Google API key for Gemini Vision Pro. |
| `GEMINI_MODEL` | Defaults to `gemini-pro-vision`. |
| `LANCEDB_DIR` | Persistent vector store directory. |
| `MEDIA_DIR`, `THUMBS_DIR` | Storage for extracted assets/thumbnails. |
| `INGEST_CACHE_DIR` | Temp space for transcripts, frames, uploads. |
| `RATE_LIMIT_PER_MIN` | Per-key throttling. |
| `INDEX_TOPK_TEXT`, `INDEX_TOPK_IMG`, `RERANK_TOPK`, `FINAL_N` | Retrieval hyperparameters. |
| `CONFIDENCE_TAU` | Minimum similarity required before answering. |
| `YT_FRAME_SCENE_THRESH`, `YT_MAX_FRAMES`, `YT_FRAME_DEDUP_DELTA` | Scene-aware frame sampling controls. |
| `YT_FRAME_EXTRACTOR` | e.g., `ffmpeg`. |
| `NOTE_CONTEXT_CHARS`, `QUIZ_QUESTION_COUNT` | Notes/quizzes generation knobs. |

See `.env.example` for all supported keys.

---

## Quick Start

1. Copy the environment template and review the variables:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the FastAPI server:
   ```bash
   python main.py
   ```
4. In another shell, start the Celery worker for background ingestion:
   ```bash
   celery -A app.tasks worker --loglevel=info
   ```
5. Hit the endpoints (examples below) with an `X-API-Key` defined in your `.env`.

## Demo Commands

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Run API (reload off for interviews)
python main.py

# 3. Start background worker
celery -A app.tasks worker --loglevel=info

# 4. Upload a PDF
curl -X POST http://localhost:8000/process-pdf/ \
  -H "X-API-Key: test-key" \
  -F "file=@docs/intro.pdf"

# 5. Enqueue a YouTube ingestion
curl -X POST http://localhost:8000/ingest-youtube \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtu.be/dQw4w9WgXcQ"}'

# 6. Poll status
curl -H "X-API-Key: test-key" http://localhost:8000/yt_status/<job_id>

# 7. Chat
curl -X POST http://localhost:8000/chat_pro \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the main theorem.", "video_id": "abc123"}'
```

---

## Why This Architecture?

- **Recall vs. precision trade-off**: MiniLM/CLIP bi-encoders give fast recall, while the ms-marco cross-encoder reranker injects precision on the most relevant snippets before fusion.
- **Transcript-first ingestion**: Skips expensive Whisper runs when official transcripts are available; reduces latency by minutes per video.
- **Scene-aware frame sampling**: ffmpeg `select='gt(scene, SCENE_THRESH)' -vsync vfr` plus perceptual hash dedup keeps only meaningful frames, capping LanceDB storage.
- **LanceDB simplicity**: Lightweight, on-disk vector store shared across users with per-user tables and index versioning for cache invalidation.
- **Gemini Vision Pro grounding**: Combines text and referenced images, citing `[doc:...]` or `[ts:start-end]` inline; abstains when confidence is below `CONFIDENCE_TAU`.

This provides a balanced, interview-ready story: scalable ingestion, efficient retrieval, precision reranking, and grounded, explainable answers at the final Gemini layer.

---

## Testing & Tooling

```bash
pip install pytest
pytest
```

Key tests cover metadata persistence, embedding consistency, retrieval fusion, caching, and generation gating. Each can be mocked to avoid GPU/model downloads during interviews.

---

Happy building—and good luck with the interview!
