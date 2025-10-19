from redis import Redis
from app.settings import settings

_redis = None

def get_redis() -> Redis:
    global _redis
    if _redis is None:
        # [MIGRATE] Redis configuration now flows through the centralized settings loader.
        _redis = Redis.from_url(settings.api.redis_url, decode_responses=True)
    return _redis
