"""
redis_cache.py

Production Redis caching wrapper for ProteinSynergyDock API.
Generates deterministic SHA256 cache keys based on normalized drug pair SMILES
and protein UniProt target. Serializes responses using msgpack with a 24-hour TTL.
Handles connection failures gracefully with silent fallback to compute mode.
"""

import os
import hashlib
import logging
from typing import Optional, Dict, Any
import msgpack
import redis

logger = logging.getLogger("proteinsynergydock.cache")

# Environment variable for Redis URL (defaults to localhost for dev/docker)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class RedisCache:
    def __init__(self, redis_url: str = REDIS_URL, default_ttl: int = 86400):
        self.default_ttl = default_ttl
        self._client: Optional[redis.Redis] = None
        try:
            self._client = redis.Redis.from_url(redis_url, socket_timeout=2.0)
            # Ping to test connection
            self._client.ping()
            logger.info("Connected to Redis cache at %s", redis_url)
        except Exception as e:
            logger.warning("Redis connection failed (%s). Caching disabled.", e)
            self._client = None

    def is_available(self) -> bool:
        """Check if Redis client is connected and responsive."""
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    @staticmethod
    def generate_cache_key(drug_a_smiles: str, drug_b_smiles: str, protein_uniprot: str) -> str:
        """
        Generate a deterministic SHA256 cache key.
        Sorts SMILES to ensure drug_a + drug_b and drug_b + drug_a produce identical keys.
        """
        norm_a = drug_a_smiles.strip()
        norm_b = drug_b_smiles.strip()
        pair_smiles = sorted([norm_a, norm_b])
        norm_protein = protein_uniprot.strip().upper()
        
        raw_key = f"{pair_smiles[0]}:{pair_smiles[1]}:{norm_protein}"
        hash_digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"psd:predict:v3:{hash_digest}"

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve and msgpack-deserialize cached prediction."""
        if not self._client:
            return None
        try:
            data = self._client.get(cache_key)
            if data:
                unpacked = msgpack.unpackb(data, raw=False)
                if isinstance(unpacked, dict):
                    unpacked["cached"] = True
                    return unpacked
        except Exception as e:
            logger.warning("Redis GET error for key %s: %s", cache_key, e)
        return None

    def set(self, cache_key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Serialize payload with msgpack and store in Redis with TTL (default 24h)."""
        if not self._client:
            return False
        try:
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            # Ensure value is dict copy without non-serializable elements
            clean_value = {k: v for k, v in value.items() if k != "cached"}
            packed = msgpack.packb(clean_value, use_bin_type=True)
            self._client.setex(name=cache_key, time=ttl_seconds, value=packed)
            return True
        except Exception as e:
            logger.warning("Redis SET error for key %s: %s", cache_key, e)
            return False

# Global Singleton Instance
cache = RedisCache()
