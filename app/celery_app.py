from celery import Celery
from app.settings import settings

# [MIGRATE] Celery now sources Redis configuration from the centralized settings loader.
celery_app = Celery("mrag", broker=settings.api.redis_url, backend=settings.api.redis_url)
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
    worker_hijack_root_logger=False,
)
