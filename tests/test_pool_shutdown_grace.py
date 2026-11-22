"""Shared-card router exit must let native cache persistence finish."""
import asyncio
from dataclasses import replace
import os
import signal
from types import SimpleNamespace
import unittest
from unittest import mock

# Import-time runtime constants must not consult the installed user catalog.
with mock.patch.dict(os.environ, {'MARATHON_USER_CATALOG': '/nonexistent-synthetic-user-catalog.toml'}):
    from marathon_app import runtime
from marathon_app.catalog import Backend


class PoolShutdownGraceTests(unittest.TestCase):
    def test_grace_covers_switch_into_handoff_pool_only(self):
        qwen = Backend('qwen', 'Qwen', None, kind='llama_swap_pool')
        inkling = replace(qwen, id='inkling', pool_lease_backend='qwen')
        other = replace(qwen, id='unrelated')
        for selected in (qwen, inkling):
            self.assertEqual(runtime._router_shutdown_timeout(selected, (qwen, inkling), 20), 150)
            self.assertEqual(runtime._router_shutdown_timeout(selected, (qwen, inkling), 180), 180)
        self.assertEqual(runtime._router_shutdown_timeout(other, (qwen, inkling), 20), 20)
        self.assertEqual(runtime._router_shutdown_timeout(qwen, (qwen,), 20), 20)
        self.assertEqual(runtime._router_shutdown_timeout(replace(qwen, kind='local'), (inkling,), 20), 20)

    def test_non_pool_router_does_not_acquire_a_pool_on_model_change(self):
        from test_router_context import router_module
        state = object.__new__(router_module.RouterState)
        state.pool_backend = None
        state._acquire_pool_worker = mock.AsyncMock()
        state.resolve_model = mock.Mock()
        async def switch():
            async with state.pool_request("inkling-2-mini"):
                pass
        asyncio.run(switch())
        state._acquire_pool_worker.assert_not_awaited()
        state.resolve_model.assert_not_called()

    def test_long_unload_finishes_without_killing_router(self):
        for name, expected_forced in [('router', False), ('llama', True)]:
            with self.subTest(process=name):
                state = object.__new__(runtime.Runtime)
                state.config = SimpleNamespace(stop_timeout=20)
                state._router_stop_timeout = 150
                state.record = mock.Mock()
                clock = [0.0]
                process = mock.Mock(pid=123456, returncode=0)
                process.poll.side_effect = lambda: 0 if clock[0] >= 60 else None
                def advance(seconds):
                    clock[0] += seconds
                with mock.patch.object(runtime.time, 'monotonic', side_effect=lambda: clock[0]), \
                     mock.patch.object(runtime.time, 'sleep', side_effect=advance), \
                     mock.patch.object(runtime.os, 'killpg') as kill:
                    state._terminate(process, name)
                self.assertEqual(state.record.call_args.args[1]['forced'], expected_forced)
                self.assertEqual(kill.call_args_list[0].args, (123456, signal.SIGTERM))
                self.assertEqual(len(kill.call_args_list), 2 if expected_forced else 1)
