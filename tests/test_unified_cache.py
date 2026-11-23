"""Backend-neutral cache contract: Marathon policy, per-engine adapters."""
import asyncio
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/routers"))
import codex_local_router as router
import marathon_cache_backends as cache
from marathon_app import catalog

TOOLS = [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]
CONTEXT = {"type": "message", "role": "user", "content": [{
    "type": "input_text",
    "text": '<marathon_context role="developer">\nsynthetic rules\n</marathon_context>'}]}


def profile(**changes):
    base = router.ModelProfile(
        slug="inkling", alias="inkling", display_name="Inkling", description="Inkling",
        model_paths=(), target="http://127.0.0.1:18493", context_window=262144,
        auto_compact_token_limit=235929, truncation_limit=235929, supports_slots=False,
        external=True, reasoning_transport="responses", cache_backend="maya",
    )
    return replace(base, **changes)


def request(**changes):
    body = {"model": "inkling", "instructions": "synthetic policy", "tools": TOOLS,
            "prompt_cache_key": "conversation-a", "stream": True, "reasoning": {"effort": "medium"},
            "input": [CONTEXT, {"type": "message", "role": "user",
                                "content": [{"type": "input_text", "text": "first synthetic question"}]}]}
    body.update(changes)
    return body


class FakeBackend:
    def __init__(self, replies):
        self.calls = []
        self.replies = replies

    async def __call__(self, profile, method, path, payload=None, **_kwargs):
        self.calls.append((method, path, payload))
        reply = self.replies[path]
        return reply(payload) if callable(reply) else reply


def maya_state(replies, **attributes):
    state = object.__new__(router.RouterState)
    state.starter_cache_enabled = True
    state.backend_cache_id = "synthetic-cache-id"
    state.pool_profile_backends = {}
    state.pool_slugs = set()
    state.pool_model = None
    state.last_response_by_model = {}
    state._generation_requests = 0
    state.checkpoint_tasks = {}
    state.slot_snapshot_min_token_growth = 4096
    state._request_json = FakeBackend(replies)
    for key, value in attributes.items():
        setattr(state, key, value)
    return state


class CapabilityTests(unittest.TestCase):
    def test_llamacpp_slot_flag_keeps_its_meaning(self):
        self.assertIs(profile(supports_slots=True).cache, cache.LLAMACPP_SLOTS)
        self.assertIs(profile(supports_slots=True, cache_backend="engine").cache, cache.LLAMACPP_SLOTS)
        self.assertIs(profile().cache, cache.MAYA_PREFIX)
        self.assertIs(profile(cache_backend="engine").cache, cache.ENGINE_PREFIX)
        default = router.ModelProfile(
            slug="x", alias="x", display_name="x", description="x", model_paths=(),
            target="http://127.0.0.1:1", context_window=1, auto_compact_token_limit=1,
            truncation_limit=1)
        self.assertIs(default.cache, cache.LLAMACPP_SLOTS)   # router default: supports_slots=True

    def test_llamacpp_capabilities_match_the_old_slot_gates(self):
        caps = cache.LLAMACPP_SLOTS
        self.assertTrue(caps.slot_files and caps.explicit_save and caps.starter_prefill)
        self.assertFalse(caps.restores_by_prefix or caps.persists_on_switch)
        self.assertIsInstance(cache.adapter_for(caps), cache.LlamaCppSlotAdapter)
        self.assertIsInstance(cache.adapter_for(cache.MAYA_PREFIX), cache.MayaPrefixAdapter)
        self.assertFalse(cache.ENGINE_PREFIX.explicit_save or cache.ENGINE_PREFIX.starter_prefill)

    def test_catalog_cache_backend_is_validated(self):
        base = ('[[external_models]]\nid = "inkling-2-mini"\nmodel = "inkling-2-mini"\n'
                'display_name = "Inkling"\ndescription = "synthetic"\n'
                'base_url = "http://127.0.0.1:9292/v1"\ncontext = 262144\n'
                'auto_compact_token_limit = 235929\ntruncation_limit = 235929\n'
                'reasoning_transport = "responses"\n')
        for line, expected in (("", "engine"), ('cache_backend = "maya"\n', "maya")):
            with self.subTest(line=line):
                parsed = catalog.external_models(catalog.tomllib.loads(base + line))
                self.assertEqual(parsed[0].cache_backend, expected)
        with self.assertRaises(ValueError):
            catalog.external_models(catalog.tomllib.loads(base + 'cache_backend = "slots"\n'))


class StarterBodyTests(unittest.TestCase):
    def test_bodies_keep_the_forwarded_scaffold_and_differ_only_in_the_sentinel(self):
        first, second = router._starter_responses_bodies(request())
        for body in (first, second):
            self.assertEqual(body["tools"], TOOLS)
            self.assertEqual(body["instructions"], "synthetic policy")
            self.assertEqual(body["reasoning"], {"effort": "medium"})
            self.assertNotIn("stream", body)
            self.assertEqual(body["input"][0], CONTEXT)
            self.assertEqual(len(body["input"]), 2)
        self.assertNotEqual(first["input"][-1], second["input"][-1])
        self.assertEqual({k: v for k, v in first.items() if k != "input"},
                         {k: v for k, v in second.items() if k != "input"})
        self.assertNotIn("first synthetic question", str(first))

    def test_a_warmup_request_still_gives_prefillable_bodies(self):
        # Codex warmups carry generate: false; Maya's /cache/prefill rejects that parameter.
        for body in router._starter_responses_bodies(request(generate=False, store=False)):
            self.assertNotIn("generate", body)
            self.assertNotIn("store", body)

    def test_first_turn_detection(self):
        self.assertTrue(router._is_first_turn_input(request()["input"]))
        self.assertFalse(router._is_first_turn_input([]))
        self.assertFalse(router._is_first_turn_input(request()["input"] + [
            {"type": "function_call", "call_id": "c", "name": "exec_command", "arguments": "{}"}]))
        self.assertFalse(router._is_first_turn_input(request()["input"] + [
            {"type": "message", "role": "assistant", "content": []}]))


class MayaAdapterTests(unittest.TestCase):
    def test_starter_built_then_restored(self):
        replies = {"/cache/prefill": lambda payload: {
            "n_tokens": 10437, "n_cached": 0, "prompt_ms": 4096.6,
            "save": {"status": "saved", "saved_bytes": 293127988}}}
        state = maya_state(replies)
        built = asyncio.run(state.prepare_engine_starter(profile(), request()))
        self.assertEqual((built["status"], built["mode"]), ("built", "build-starter-cache"))
        self.assertEqual((built["starter_tokens"], built["saved_bytes"]), (10437, 293127988))
        method, path, payload = state._request_json.calls[0]
        self.assertEqual((method, path, payload["save"]), ("POST", "/cache/prefill", True))
        self.assertEqual(len(payload["requests"]), 2)
        replies["/cache/prefill"] = {"n_tokens": 10437, "n_cached": 10437, "prompt_ms": 0.0,
                                     "save": {"status": "clean"}}
        restored = asyncio.run(state.prepare_engine_starter(profile(), request()))
        self.assertEqual((restored["status"], restored["saved_bytes"]), ("restored", 0))

    def test_starter_needs_a_reserved_pool_worker_and_never_raises(self):
        state = maya_state({}, pool_slugs={"inkling"}, pool_model=None)
        self.assertEqual(asyncio.run(state.prepare_engine_starter(profile(), request()))["status"], "skipped")
        state = maya_state({"/cache/prefill": lambda _p: (_ for _ in ()).throw(RuntimeError("down"))})
        result = asyncio.run(state.prepare_engine_starter(profile(), request()))
        self.assertEqual((result["status"], result["mode"]), ("error", "starter-cache-fallback"))
        self.assertEqual(
            asyncio.run(maya_state({}).prepare_engine_starter(profile(cache_backend="engine"), request()))["status"],
            "skipped")

    def candidate(self, response_id="resp-1"):
        return router.ConversationCheckpointCandidate(
            profile_slug="inkling", profile_alias="inkling", prompt_cache_key="conversation-a",
            response_id=response_id, scaffold_fingerprint="fp", context_tokens=150000,
            post_compaction=False, checkpoint_request=None)

    def test_idle_checkpoint_saves_through_the_engine(self):
        state = maya_state({"/slots/0?action=save": {
            "status": "saved", "n_saved": 149000, "n_written": 2611378536}},
            last_response_by_model={"inkling": "resp-1"}, slot_snapshot_min_token_growth=4096)
        result = asyncio.run(state._save_engine_checkpoint_locked(profile(), self.candidate()))
        self.assertEqual((result["status"], result["size_bytes"]), ("saved", 2611378536))
        # Same growth rule as llama.cpp: a small append is not rewritten at idle ...
        skipped = asyncio.run(state._save_engine_checkpoint_locked(profile(), self.candidate()))
        self.assertEqual(skipped["reason"], "conversation has not grown enough since the last checkpoint")
        self.assertEqual(len(state._request_json.calls), 1)
        # ... but a forced flush (before unload) asks the engine, which writes nothing when clean.
        state._request_json.replies["/slots/0?action=save"] = {"status": "clean", "n_saved": 149000, "n_written": 0}
        clean = asyncio.run(state._save_engine_checkpoint_locked(profile(), self.candidate(), force=True))
        self.assertEqual((clean["status"], clean["size_bytes"]), ("skipped", 0))

    def test_idle_checkpoint_skips_an_advanced_conversation(self):
        state = maya_state({}, last_response_by_model={"inkling": "resp-2"})
        result = asyncio.run(state._save_engine_checkpoint_locked(profile(), self.candidate("resp-1")))
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(state._request_json.calls, [])

    def test_engine_save_failure_is_an_error(self):
        state = maya_state({"/slots/0?action=save": {"status": "failed"}},
                           last_response_by_model={"inkling": "resp-1"})
        with self.assertRaises(RuntimeError):
            asyncio.run(state._save_engine_checkpoint_locked(profile(), self.candidate()))

    def test_checkpoint_scheduling_follows_the_adapter(self):
        state = maya_state({})
        state.slot_snapshots_enabled = False
        for prof, reason in ((profile(), "rolling checkpoints disabled"),
                             (profile(cache_backend="engine"), "backend has no cache save API")):
            self.assertEqual(state.schedule_conversation_checkpoint(prof, request(), "r", {}, [])["reason"], reason)
        state.slot_snapshots_enabled = True
        self.assertEqual(
            state.schedule_conversation_checkpoint(profile(cache_backend="engine"), request(), "r", {}, [])["reason"],
            "backend has no cache save API")


if __name__ == "__main__":
    unittest.main()


class WarmupTests(unittest.TestCase):
    """A warmup loads the worker for every model without slots; maya also warms its starter."""

    def state(self, starter_enabled=True):
        state = object.__new__(router.RouterState)
        state._closing = False
        state.starter_cache_enabled = starter_enabled
        state.live_slot_by_model = {}
        state.starter_prewarm_tasks = {}
        state.ensure_model_async = mock.AsyncMock()
        state._schedule_engine_starter_prewarm = mock.Mock(return_value={"status": "scheduled"})
        state.telemetry = mock.Mock()
        return state

    def run_warmup(self, state, prof):
        async def go():
            result = state.schedule_starter_cache_prewarm(prof, request(), "warm", "conversation-a")
            task = state.starter_prewarm_tasks.get(prof.slug)
            if task is not None:
                await task
            return result
        return asyncio.run(go())

    def test_maya_warmup_schedules_the_engine_starter(self):
        state = self.state()
        self.assertEqual(self.run_warmup(state, profile()), {"status": "scheduled"})
        state._schedule_engine_starter_prewarm.assert_called_once()
        state.ensure_model_async.assert_not_awaited()

    def test_without_starter_or_cache_backend_the_warmup_still_loads_the_worker(self):
        for prof, enabled in ((profile(cache_backend="engine"), True), (profile(), False)):
            state = self.state(enabled)
            self.assertEqual(self.run_warmup(state, prof), {"status": "scheduled", "kind": "model-load"})
            state._schedule_engine_starter_prewarm.assert_not_called()
            state.ensure_model_async.assert_awaited_once_with(prof.slug)
