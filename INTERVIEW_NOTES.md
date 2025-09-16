# Multimodal RAG Interview Notes

## High-Impact Questions & Answers

1. **What problem does this platform solve?**  
   It delivers a multi-tenant study companion for YouTube/PDF content. Learners enqueue lectures, the backend ingests transcripts/frames, embeds them into tenant-isolated ChromaDB collections, generates structured notes and quizzes, then exposes a chat endpoint that cites sources while answering questions scoped to a specific video.

2. **Walk me through the ingestion flow for a new YouTube lecture.**  
   1) FastAPI authenticates the tenant key and enqueues the request via Celery.  
   2) The worker downloads lightweight media with `yt-dlp`, captures metadata, and extracts transcripts plus optional frames.  
   3) Transcript chunks are embedded with SentenceTransformers; frames go through CLIP. Both are written to tenant-specific Chroma collections.  
   4) Gemini summarizes the lecture into notes (summary, key points, timeline) and quiz prompts, persisted in SQLite.  
   5) The worker returns status so the UI refreshes the study library.

3. **How is tenant isolation enforced?**  
   - API key authentication maps directly to tenant IDs.  
   - Chroma collection names are namespaced by tenant.  
   - Notes/quizzes store keys on `(user_id, video_id)` in SQLite.  
   - Rate limiting uses the same API key so tenants cannot starve others.

4. **What happens when a learner chats about a specific video?**  
   The client sends `thread_id`, `message`, and `video_id`. Retrieval narrows Chroma queries with `where={"video_id": ...}` so only that lecture’s embeddings are considered. Gemini receives the question plus retrieved context (text snippets, optional images) and returns a cited answer.

5. **How do you handle failures or very long videos?**  
   - Download timeouts and retry counts are capped.  
   - The worker rejects videos longer than the configured minute threshold.  
   - Transcript-less videos still receive fallback notes.  
   - Failures push the Celery job to `FAILURE` with rich metadata so the status endpoint and UI can surface clear errors.

6. **Which trade-offs were made in the design?**  
   - SQLite is simple but single-writer; fine for demos, but production should migrate to Postgres.  
   - Gemini adds latency/cost, so heuristic notes act as a fallback.  
   - Retrieval defaults to dense embeddings; sparse or reranking can be enabled later but were omitted to keep resource usage low.

7. **How would you scale this for production?**  
   Deploy FastAPI behind a gateway with autoscaled workers, move notes to a managed relational DB, push frames to object storage, add observability (metrics/logging), introduce retry queues, and manage secrets via a vault.

8. **How is the system tested without calling external services?**  
   Pytest fixtures monkeypatch the embedder, Gemini generator, Redis client, and Celery results. Tests seed the note store, assert PDF ingestion indexes content, validate `/videos/*` routes, and ensure chat returns mocked answers scoped to a provided `video_id`.

## Whiteboard Architecture Diagram

```
┌───────────┐    HTTPS     ┌────────────┐     RPC/Redis      ┌─────────────┐
│  Browser  │ ───────────▶ │  FastAPI   │ ─────────────────▶ │   Celery     │
│  (HTML UI)│              │  (app.main)│                    │   Worker     │
└───────────┘              │ - Auth     │◀─Redis cache──────▶│ - yt_dlp     │
                           │ - REST API │                    │ - Embedding  │
                           │ - Notes API│                    │ - Gemini     │
                           └─────┬──────┘                    └─────┬───────┘
                                 │                               │
                Tenant-scoped embeddings (ChromaDB)              │
                                 │                               │
                          ┌──────▼────────┐                 ┌────▼────────┐
                          │  ChromaDB     │                 │ SQLite Notes│
                          │ (text/images) │                 │ per video   │
                          └───────────────┘                 └─────────────┘
```

Explain left-to-right: authenticated requests hit FastAPI, ingestion jobs are queued to Celery via Redis, embeddings land in tenant-specific Chroma collections, Gemini-generated notes/quizzes are stored in SQLite, and chat reads from these stores to deliver cited responses.
