"""
cache.py — Upstash Redis query cache.

Result cache — caches the full reranked results for a (question, index_id) pair.
  Key:   result:{hash(question)}:{index_id}
  Value: JSON serialized results

TTL from config (default 1 hour).
On cache hit, stage1 + stage2 latency are reported as 0ms
and a 'cached: true' flag is added to the response.

Why Upstash:
  - Serverless Redis — no server to manage, works over HTTP
  - Free tier: 10k requests/day, 256MB storage
  - Works identically across local dev and production
"""

import json
import hashlib
from typing import Optional

from upstash_redis import Redis

from app.config import (
    UPSTASH_REDIS_REST_URL,
    UPSTASH_REDIS_REST_TOKEN,
    CACHE_TTL_SECONDS,
    CACHE_ENABLED,
)

#  Redis client (None if credentials missing) 
_redis: Optional[Redis] = None


def get_redis() -> Optional[Redis]:
    global _redis
    if _redis is None and CACHE_ENABLED:
        try:
            _redis = Redis(
                url=UPSTASH_REDIS_REST_URL,
                token=UPSTASH_REDIS_REST_TOKEN,
            )
            # Verify connection
            _redis.ping()
            print("✓ Upstash Redis connected — query caching enabled")
        except Exception as e:
            print(f"⚠ Redis connection failed: {e} — caching disabled")
            _redis = None
    return _redis


def _hash(text: str) -> str:
    """Short deterministic hash for cache keys."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


#  Result cache 

def get_cached_results(question: str, index_id: str) -> Optional[dict]:
    """Return cached query results or None."""
    r = get_redis()
    if not r:
        return None
    try:
        key = f"result:{_hash(question)}:{index_id}"
        val = r.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


def set_cached_results(question: str, index_id: str, payload: dict):
    """Cache full result payload with TTL."""
    r = get_redis()
    if not r:
        return
    try:
        key = f"result:{_hash(question)}:{index_id}"
        r.setex(key, CACHE_TTL_SECONDS, json.dumps(payload))
    except Exception:
        pass


def cache_stats() -> dict:
    """Return basic cache info for health endpoint."""
    r = get_redis()
    if not r:
        return {"enabled": False}
    try:
        info = r.info()
        return {
            "enabled":     True,
            "used_memory": info.get("used_memory_human", "n/a"),
            "ttl_seconds": CACHE_TTL_SECONDS,
        }
    except Exception:
        return {"enabled": True, "status": "error"}