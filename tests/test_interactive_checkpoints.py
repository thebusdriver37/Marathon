"""Interactive requests take priority at safe idle-checkpoint boundaries."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_router_context import fixture_profile, router_module as router


class InteractiveCheckpointTests(unittest.IsolatedAsyncioTestCase):
    def make_state(self, directory):
        profile = fixture_profile()
        state = object.__new__(router.RouterState)
        state.available_profiles = {profile.slug: profile}
        state.backend_lock = asyncio.Lock()
        state._generation_requests = 0
        state.backend_cache_id = 'synthetic-backend'
        state.slot_id = 0
        state.slot_snapshot_idle_seconds = 0
        state.slot_snapshot_min_token_growth = 4096
        state.pending_checkpoints = {}
        state.checkpoint_tasks = {}
        state.live_slot_by_model = {profile.slug: 'synthetic-response'}
        state.live_prompt_cache_key_by_model = {profile.slug: 'synthetic-conversation'}
        state.slot_checkpoint_store = router.RollingCheckpointStore(
            Path(directory), Path(directory), max_count=2, max_bytes=4096, ttl_seconds=3600)
        state.telemetry = mock.Mock()
        candidate = router.ConversationCheckpointCandidate(
            profile_slug=profile.slug, profile_alias=profile.alias,
            prompt_cache_key='synthetic-conversation', response_id='synthetic-response',
            scaffold_fingerprint='synthetic-scaffold', context_tokens=20000,
            checkpoint_request={'input': [{'type': 'message', 'role': 'user', 'content': 'synthetic'}]})
        state.pending_checkpoints[profile.slug] = candidate
        return state, profile, candidate

    async def test_incoming_reply_defers_preparation_and_idle_retry_still_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            state, profile, candidate = self.make_state(directory)
            preparing, finish_prepare = asyncio.Event(), asyncio.Event()
            requested, entered, finish_generation = asyncio.Event(), asyncio.Event(), asyncio.Event()
            calls = []

            async def request_json(_profile, method, path, body, **kwargs):
                calls.append(path)
                if path == '/apply-template':
                    preparing.set()
                    await finish_prepare.wait()
                    return {'prompt': 'synthetic prompt'}
                if path == '/tokenize':
                    return {'tokens': [*range(20), len(calls)]}
                return {}

            async def save_slot(_profile, filename):
                folder = state.slot_checkpoint_store.profile_dir(profile.slug)
                (folder / filename).write_bytes(b'synthetic snapshot')
                return {'status': 'saved'}

            async def generate(*args, **kwargs):
                requested.set()
                async with state.backend_lock:
                    entered.set()
                    await finish_generation.wait()
                return {'response_id': 'next-response'}

            state._request_json = mock.AsyncMock(side_effect=request_json)
            state.save_slot = mock.AsyncMock(side_effect=save_slot)
            state._process_translated_create = generate
            checkpoint = asyncio.create_task(state._save_checkpoint_after_idle(profile.slug))
            state.checkpoint_tasks[profile.slug] = checkpoint
            try:
                await asyncio.wait_for(preparing.wait(), 2)
                generation = asyncio.create_task(state._process_websocket_create({'input': []}))
                await asyncio.wait_for(requested.wait(), 2)
                finish_prepare.set()
                await asyncio.wait_for(entered.wait(), 2)
                self.assertNotIn('/completion', calls)
                state.save_slot.assert_not_awaited()
                self.assertIs(state.pending_checkpoints[profile.slug], candidate)
                finish_generation.set()
                await asyncio.wait_for(generation, 2)
                await asyncio.wait_for(checkpoint, 2)
                self.assertEqual(state._generation_requests, 0)
                state.save_slot.assert_awaited_once()
                self.assertEqual(state.pending_checkpoints, {})
                record = state.slot_checkpoint_store.find(
                    profile_slug=profile.slug, profile_alias=profile.alias,
                    prompt_cache_key=candidate.prompt_cache_key,
                    backend_cache_id=state.backend_cache_id,
                    scaffold_fingerprint=candidate.scaffold_fingerprint)
                self.assertIsNotNone(record)
            finally:
                finish_generation.set()
                checkpoint.cancel()
                await asyncio.gather(checkpoint, return_exceptions=True)

    async def test_reply_during_prefill_skips_disk_save(self):
        with tempfile.TemporaryDirectory() as directory:
            state, profile, candidate = self.make_state(directory)
            prefilled, finish_prefill = asyncio.Event(), asyncio.Event()
            async def prefill(*args):
                prefilled.set()
                await finish_prefill.wait()
                return {'context_tokens': 20000, 'conversation_item_count': 1,
                        'conversation_prefix_hash': 'a' * 64}
            state._prefill_conversation_checkpoint_boundary = prefill
            state.save_slot = mock.AsyncMock()
            task = asyncio.create_task(state._save_conversation_checkpoint(candidate, force=False))
            state.checkpoint_tasks[profile.slug] = task
            await asyncio.wait_for(prefilled.wait(), 2)
            state._generation_requests = 1
            finish_prefill.set()
            with self.assertRaises(router._CheckpointDeferred):
                await asyncio.wait_for(task, 2)
            state.save_slot.assert_not_awaited()
            self.assertFalse(state.backend_lock.locked())

    async def test_started_slot_write_finishes_and_commits_before_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            state, profile, candidate = self.make_state(directory)
            writing, finish_write = asyncio.Event(), asyncio.Event()
            state._prefill_conversation_checkpoint_boundary = mock.AsyncMock(return_value={
                'context_tokens': 20000, 'conversation_item_count': 1,
                'conversation_prefix_hash': 'a' * 64})
            async def save_slot(_profile, filename):
                writing.set()
                await finish_write.wait()
                folder = state.slot_checkpoint_store.profile_dir(profile.slug)
                (folder / filename).write_bytes(b'complete snapshot')
            state.save_slot = save_slot
            checkpoint = asyncio.create_task(state._save_conversation_checkpoint(candidate, force=False))
            state.checkpoint_tasks[profile.slug] = checkpoint
            await asyncio.wait_for(writing.wait(), 2)
            state._generation_requests = 1
            waiting = asyncio.create_task(state.backend_lock.acquire())
            await asyncio.sleep(0)
            self.assertFalse(waiting.done())
            finish_write.set()
            result = await asyncio.wait_for(checkpoint, 2)
            self.assertEqual(result['status'], 'saved')
            await asyncio.wait_for(waiting, 2)
            self.assertEqual(list(Path(directory).rglob('*.pending.bin')), [])
            self.assertEqual(len(list(Path(directory).rglob('conversation__*.bin'))), 1)
            state.backend_lock.release()

    async def test_reply_while_waiting_for_cache_io_prevents_slot_write(self):
        with tempfile.TemporaryDirectory() as directory:
            state, profile, _ = self.make_state(directory)
            state._request_json = mock.AsyncMock()
            async with state.slot_checkpoint_store.lock_io():
                checkpoint = asyncio.create_task(state.save_slot(profile, 'synthetic.bin'))
                state.checkpoint_tasks[profile.slug] = checkpoint
                await asyncio.sleep(0)
                self.assertFalse(checkpoint.done())
                state._generation_requests = 1
            with self.assertRaises(router._CheckpointDeferred):
                await asyncio.wait_for(checkpoint, 2)
            state._request_json.assert_not_awaited()
            async with state.slot_checkpoint_store.lock_io():
                pass

    async def test_generation_priority_is_released_on_cancel_and_warmup_does_not_claim_it(self):
        state = object.__new__(router.RouterState)
        started = asyncio.Event()
        async def generate(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()
        state._process_translated_create = generate
        task = asyncio.create_task(state._process_websocket_create({'input': []}))
        await started.wait()
        self.assertEqual(state._generation_requests, 1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(state._generation_requests, 0)
        state._process_translated_create = mock.AsyncMock(return_value={})
        await state._process_websocket_create({'input': [], 'generate': False})
        self.assertEqual(state._generation_requests, 0)
