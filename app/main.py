from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from loguru import logger
except Exception:  # pragma: no cover - fallback logging
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("app")
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.routes import router, limiter
from app.settings import settings

app = FastAPI(title=settings.api.app_name)
app.include_router(router)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    logger.info(f"{request_id} {request.method} {request.url.path} start")
    try:
        response = await call_next(request)
        logger.info(f"{request_id} {request.method} {request.url.path} done {response.status_code}")
        return response
    except Exception:
        logger.exception(f"{request_id} unhandled error")
        raise


@app.get("/", include_in_schema=False)
def ui_index():
    return FileResponse("app/static/index.html")


app.mount("/static", StaticFiles(directory="app/static"), name="static")
