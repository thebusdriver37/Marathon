"""Backend-neutral conversation cache contract.

Marathon's router owns the cache *policy* for every model: which conversation
is live, when a new session gets the starter prefix, when an idle conversation
is checkpointed, what happens on a conversation switch or unload, and how
cached tokens are reported. An adapter only performs the engine-specific
*mechanics* (save, restore, prefill a shared prefix, probe) and declares what
the engine does by itself, so the router can skip work the engine already does.

Adapters:

- ``llamacpp-slots``: llama.cpp's /slots save/restore/erase with router-managed
  snapshot files (and the content-addressed slot store). This is the existing
  Qwen path; its methods delegate to the router's unchanged implementations.
- ``maya-prefix``: Inkling on Maya with the native durable cache. Snapshots are
  content-addressed by exact token prefix inside the engine, restored
  automatically by the next matching prompt and persisted on eviction/unload.
  Marathon adds explicit checkpoints and the starter prefix through the
  runtime's /slots and /cache/prefill endpoints (slot_api.py).
- ``engine-prefix``: an engine with only its own in-memory prefix cache and no
  control API (DeepSeek on mlx-serve, Spark, generic OpenAI servers). Marathon
  still keeps prompts prefix-stable and reports cached tokens it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

@dataclass(frozen=True)
class CacheCapabilities:
    name: str
    # Router-managed snapshot files restored/saved by filename (llama.cpp slots).
    slot_files: bool
    # The engine can persist the hot conversation now, on request.
    explicit_save: bool
    # The engine restores a stored prefix by itself on the next matching prompt.
    restores_by_prefix: bool
    # The engine persists the outgoing conversation itself before another one
    # replaces it, and when its worker unloads gracefully.
    persists_on_switch: bool
    persists_on_unload: bool
    # The engine can prefill (and persist) a shared prefix without generating.
    starter_prefill: bool


LLAMACPP_SLOTS = CacheCapabilities(
    name="llamacpp-slots", slot_files=True, explicit_save=True,
    restores_by_prefix=False, persists_on_switch=False, persists_on_unload=False,
    starter_prefill=True,
)
MAYA_PREFIX = CacheCapabilities(
    name="maya-prefix", slot_files=False, explicit_save=True,
    restores_by_prefix=True, persists_on_switch=True, persists_on_unload=True,
    starter_prefill=True,
)
ENGINE_PREFIX = CacheCapabilities(
    name="engine-prefix", slot_files=False, explicit_save=False,
    restores_by_prefix=False, persists_on_switch=False, persists_on_unload=False,
    starter_prefill=False,
)

CACHE_BACKENDS = ("llamacpp", "maya", "engine")


def capabilities_for(cache_backend: str, supports_slots: bool) -> CacheCapabilities:
    """The adapter a profile uses. llama.cpp's slot flag keeps its meaning."""

    if supports_slots:
        return LLAMACPP_SLOTS
    if cache_backend == "maya":
        return MAYA_PREFIX
    return ENGINE_PREFIX


class CacheAdapter:
    capabilities = ENGINE_PREFIX

    async def probe(self, router: Any, profile: Any) -> dict[str, Any]:
        return {"backend": self.capabilities.name, "supported": False}

    async def save(self, router: Any, profile: Any, filename: str | None = None) -> dict[str, Any]:
        return {"status": "unsupported"}

    async def prefill(self, router: Any, profile: Any, bodies: list[dict[str, Any]],
                      *, save: bool) -> dict[str, Any]:
        return {"status": "unsupported"}


class MayaPrefixAdapter(CacheAdapter):
    """Mechanics for runtime/inkling-maya-*/slot_api.py."""

    capabilities = MAYA_PREFIX

    async def probe(self, router: Any, profile: Any) -> dict[str, Any]:
        slots = await router._request_json(profile, "GET", "/slots", retry_connection_error=True)
        slot = slots[0] if isinstance(slots, list) and slots and isinstance(slots[0], dict) else {}
        cache = slot.get("cache") if isinstance(slot.get("cache"), dict) else {}
        return {
            "backend": self.capabilities.name,
            "supported": bool(cache.get("enabled")),
            "busy": bool(slot.get("is_processing")),
            "cached_tokens": int(slot.get("n_prompt_tokens") or 0),
            "dirty": bool(cache.get("dirty")),
            "store_files": cache.get("store_files"),
            "store_bytes": cache.get("store_bytes"),
            "engine_write_bytes": slot.get("engine_write_bytes"),
        }

    async def save(self, router: Any, profile: Any, filename: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        result = await router._request_json(
            profile, "POST", "/slots/0?action=save", {"filename": filename} if filename else {},
            retry_connection_error=True,
        )
        if not isinstance(result, dict):
            raise RuntimeError("Maya returned an invalid slot save result")
        status = result.get("status")
        if status not in ("saved", "clean", "queued", "empty", "short"):
            raise RuntimeError(f"Maya did not save the conversation ({status})")
        return {
            "status": status,
            "context_tokens": int(result.get("n_saved") or 0),
            "size_bytes": int(result.get("n_written") or 0),
            "engine_write_bytes_delta": result.get("engine_write_bytes_delta"),
            "duration_ms": (time.perf_counter() - started) * 1000.0,
        }

    async def prefill(self, router: Any, profile: Any, bodies: list[dict[str, Any]],
                      *, save: bool) -> dict[str, Any]:
        result = await router._request_json(
            profile, "POST", "/cache/prefill", {"requests": bodies, "save": save},
            retry_connection_error=True,
        )
        if not isinstance(result, dict) or not isinstance(result.get("n_tokens"), int):
            raise RuntimeError("Maya returned an invalid prefill result")
        return result


class LlamaCppSlotAdapter(CacheAdapter):
    """llama.cpp /slots. A facade over the router's existing, tuned methods, so
    the Qwen path performs exactly the same backend calls as before."""

    capabilities = LLAMACPP_SLOTS

    async def probe(self, router: Any, profile: Any) -> dict[str, Any]:
        cached = await router._slot_has_cached_prompt(profile)
        return {"backend": self.capabilities.name, "supported": True, "has_cached_prompt": cached}

    async def save(self, router: Any, profile: Any, filename: str | None = None) -> dict[str, Any]:
        if not filename:
            raise ValueError("llama.cpp slot saves need a snapshot filename")
        return await router.save_slot(profile, filename)

    async def prefill(self, router: Any, profile: Any, bodies: list[dict[str, Any]],
                      *, save: bool) -> dict[str, Any]:
        # The starter flow (render, common prefix, prefill, save, prune) is the
        # router's prepare_starter_cache; it takes the original request.
        return await router.prepare_starter_cache(profile, bodies[0])


ENGINE_ADAPTER = CacheAdapter()
MAYA_ADAPTER = MayaPrefixAdapter()
LLAMACPP_ADAPTER = LlamaCppSlotAdapter()


def adapter_for(capabilities: CacheCapabilities) -> CacheAdapter:
    if capabilities is LLAMACPP_SLOTS:
        return LLAMACPP_ADAPTER
    return MAYA_ADAPTER if capabilities is MAYA_PREFIX else ENGINE_ADAPTER
