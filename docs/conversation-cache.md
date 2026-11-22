# Conversation cache: one Marathon flow, per-engine adapters

Marathon owns the cache policy for every model. The router decides which
conversation is live, when a new session gets the starter prefix, when an idle
conversation is checkpointed, what happens on a switch or unload, and reports
cached tokens through the same accounting for every backend. An adapter in
`scripts/routers/marathon_cache_backends.py` does only the engine mechanics and
declares what the engine already does by itself.

| Adapter | Selected by | Save | Restore | Starter | Engine does itself |
| --- | --- | --- | --- | --- | --- |
| `llamacpp-slots` | `supports_slots = true` | `/slots/0?action=save` file | file by name | template, tokenize, prefill, save | nothing |
| `maya-prefix` | external model `cache_backend = "maya"` | `/slots/0?action=save` (engine store) | automatic on exact prefix | `/cache/prefill` with save | persist on switch and on unload |
| `engine-prefix` | default otherwise | none | none | none | in-memory prefix cache only |

The llama.cpp adapter is a facade over the router's existing methods, so the
Qwen path makes exactly the same backend calls as before (`ModelProfile.cache`
maps `supports_slots = true` to it regardless of `cache_backend`).

For `maya-prefix` the router:

- prefills the starter (leading context plus tools and instructions, rendered by
  the engine from the exact forwarded Responses body) on a new conversation and
  on frontend warmup, unless an unsaved conversation is hot;
- schedules the same idle checkpoint as llama.cpp (`slot_snapshot_*` settings)
  and asks the engine to save; an already stored state answers `clean` with no
  write;
- does nothing on normal turns or on resume: the engine restores the longest
  stored prefix itself, on any card that shares the store.

Enable for Inkling in the user catalog, on each Inkling `[[external_models]]`
entry: `cache_backend = "maya"`, with workers running a runtime image that has
`slot_api.py` (gpu-control `inkling/slot-api`). Without that image the adapter's
calls fail softly (starter `status = "error"`, checkpoint logged as failed) and
the native lazy cache keeps working as before.

Probe: `scripts/evals/unified_cache_probe.py` runs one yardstick (new session,
turns, switch, idle checkpoint, resume after restart, cold control) through an
isolated router for any engine.
