"""Local reservations must follow the requested model, not terminal count."""

import asyncio
import fcntl
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from marathon_app.catalog import Backend
from marathon_app.pool import acquire_pool_worker
from marathon_app import runtime
from test_router_context import fixture_profile, router_module


class PoolRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backend = Backend("test-pool", "Pool", None, kind="llama_swap_pool",
                               proxy="http://127.0.0.1:9292", model_alias="local",
                               pool_models=("worker-1", "worker-2", "worker-3"))
        local = replace(fixture_profile(), slug="local", alias="local",
                        target=self.backend.proxy)
        remote = replace(local, slug="spark", alias="remote", target="http://127.0.0.1:18088",
                         external=True, supports_slots=False)
        alias = replace(local, slug="local-selector", supports_slots=False, external=True)
        for patcher in (
            mock.patch.dict(os.environ, {"MARATHON_LAZY_POOL_BACKEND": "test-pool",
                "MARATHON_MODEL_SLUG": "local", "MARATHON_POOL_RUNTIME_DIR": str(self.root)}),
            mock.patch.object(router_module, "backends", return_value={"test-pool": self.backend}),
            mock.patch.object(router_module, "_available_profiles",
                              return_value={p.slug: p for p in (local, remote, alias)}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.state = router_module.RouterState("local", self.root / "state", self.root / "logs")
        self.state.flush_conversation_checkpoints = mock.AsyncMock(return_value=[])
        self.addCleanup(self.state._release_pool_worker)

    def reserve(self, instance):
        handle, worker = acquire_pool_worker(self.backend, self.root, instance)
        self.addCleanup(handle.close)
        return handle, worker

    async def test_discovery_warmup_and_remote_work_when_local_pool_is_full(self):
        for i in range(3):
            self.reserve(str(i))
        self.state.resolve_model("local")
        async with self.state.pool_request("local", generate=False):
            self.assertIsNone(self.state.pool_handle)
            self.assertEqual(self.state.schedule_starter_cache_prewarm(
                self.state.resolve_model("local"), {}, "warm", "test")["status"], "skipped")
        async with self.state.pool_request("spark"):
            self.assertIsNone(self.state.pool_handle)
            self.assertEqual(self.state.resolve_model("spark").target, "http://127.0.0.1:18088")
        with self.assertRaisesRegex(RuntimeError, "No free local worker"):
            async with self.state.pool_request("local"):
                self.fail("local inference cannot exceed physical capacity")

    async def test_switch_releases_local_worker_and_can_reacquire_another(self):
        async with self.state.pool_request("local"):
            self.assertTrue(self.state.resolve_model("local").target.endswith("/upstream/worker-1"))
        self.state.live_slot_by_model["local"] = "old-response"
        async with self.state.pool_request("spark"):
            self.assertIsNone(self.state.pool_handle)
            self.assertNotIn("local", self.state.live_slot_by_model)
        self.assertEqual(self.reserve("another-session")[1], "worker-1")
        async with self.state.pool_request("local"):
            self.assertEqual(self.state.pool_model, "worker-2")
            self.assertTrue(self.state.resolve_model("local").target.endswith("/upstream/worker-2"))

    async def test_frontend_warmup_keeps_default_local_preload_when_capacity_is_free(self):
        async with self.state.pool_request("local", generate=False):
            self.assertEqual(self.state.pool_model, "worker-1")
        async with self.state.pool_request("local"):
            self.assertEqual(self.state.pool_model, "worker-1")
        async with self.state.pool_request("spark", generate=False):
            self.assertIsNone(self.state.pool_handle)

    async def test_local_selector_alias_cannot_bypass_reservations(self):
        async with self.state.pool_request("local-selector"):
            self.assertEqual(self.state.pool_model, "worker-1")
            self.assertTrue(self.state.resolve_model("local-selector").target.endswith("/upstream/worker-1"))
        async with self.state.pool_request("local"):
            self.assertEqual(self.state.pool_model, "worker-1")

    async def test_remote_request_waits_for_active_local_request(self):
        entered = asyncio.Event()

        async def remote():
            async with self.state.pool_request("spark"):
                entered.set()

        async with self.state.pool_request("local"):
            task = asyncio.create_task(remote())
            await asyncio.sleep(0)
            self.assertFalse(entered.is_set())
            self.assertIsNotNone(self.state.pool_handle)
        await task
        self.assertTrue(entered.is_set())
        self.assertIsNone(self.state.pool_handle)

    def test_automatic_names_continue_past_three_and_reuse_free_names(self):
        with mock.patch.object(runtime, "RUNTIME_DIR", self.root):
            handles = []
            try:
                for name in (None, "second", "third", "instance-4"):
                    path = runtime.runtime_paths(name).lock_file
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handle = path.open("w")
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    handles.append(handle)
                self.assertEqual(runtime.automatic_launch_instance(), "instance-5")
                handles[1].close()
                self.assertEqual(runtime.automatic_launch_instance(), "second")
            finally:
                for handle in handles:
                    handle.close()
