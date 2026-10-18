"""Nonempty warmups must preserve valid live prompt caches."""
import asyncio
import copy
import threading
import unittest
from unittest import mock

from test_router_context import fixture_profile, router_module as router


def message(text, role="user"):
    return {"type": "message", "role": role,
            "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}]}


class WarmupCacheTests(unittest.IsolatedAsyncioTestCase):
    async def scenario(self, *, fork=False, slot_valid=True, changed_scaffold=False,
                       disk_checkpoint=False, delta_only=False, base_only=False):
        profile = fixture_profile()
        state = object.__new__(router.RouterState)
        state.resolve_model = mock.Mock(return_value=profile)
        state.ensure_model_async = mock.AsyncMock(return_value=profile)
        state.lineage_lock = asyncio.Lock()
        state.backend_lock = asyncio.Lock()
        state.lock = threading.Lock()
        state.lineage = {}
        state.last_response_by_model = {}
        state.live_slot_by_model = {}
        state.live_prompt_cache_key_by_model = {}
        state.starter_slot_models = set()
        state.experimental_delta_only = delta_only
        state.slot_id = 0
        state._trace_seq = state._response_id_seq = 0
        state.debug = False
        state.web_search = None
        state.telemetry = mock.Mock()
        state.trace_request = mock.Mock()
        state._slot_has_cached_prompt = mock.AsyncMock(return_value=slot_valid)
        state.prepare_starter_cache = mock.AsyncMock(return_value={"mode": "build-starter-cache", "status": "built"})
        state.prepare_conversation_checkpoint = mock.AsyncMock(return_value={"status": "skipped"})
        state.erase_slot = mock.AsyncMock()
        state.schedule_conversation_checkpoint = mock.Mock(return_value={"status": "skipped"})
        state.schedule_starter_cache_prewarm = mock.Mock(return_value={"status": "skipped"})
        answer = message("remembered fact", "assistant")
        state._run_responses_loop = mock.AsyncMock(side_effect=[
            ({"id": "resp_seed"}, [answer], 0), ({"id": "resp_reply"}, [answer], 0),
        ])
        scaffold = {"model": profile.slug, "instructions": "stable instructions", "tools": []}
        history = [message("original user facts")]
        await state.process_websocket_create({**scaffold, "prompt_cache_key": "original", "input": history})
        history = history + [answer]
        key = "fork" if fork else "original"
        prefix_count = 1 if base_only else len(history)
        warm = await state.process_websocket_create({**scaffold, "prompt_cache_key": key,
                                                     "input": history[:prefix_count], "generate": False})
        self.assertEqual(state.live_slot_by_model[profile.slug], "resp_seed")
        for action in (state.prepare_starter_cache, state.prepare_conversation_checkpoint, state.erase_slot):
            action.reset_mock()
        if disk_checkpoint:
            state.prepare_conversation_checkpoint.return_value = {
                "mode": "restore-conversation-checkpoint", "status": "restored"}
        delta = [message("new user question")]
        await state.process_websocket_create({**scaffold, "prompt_cache_key": key,
            "input": history[prefix_count:] + delta,
            "previous_response_id": warm["response_id"],
            **({"instructions": "changed instructions"} if changed_scaffold else {})})
        forwarded = state._run_responses_loop.await_args.kwargs["forward_request"]
        if not delta_only:
            self.assertEqual(forwarded["input"], history + delta)
            self.assertTrue(forwarded["cache_prompt"])
        completions = [call.args[1] for call in state.telemetry.emit.call_args_list
                       if call.args[0] == "router.response.completed"]
        return state, completions[-1]["slot"]["prepare_mode"]

    async def test_same_conversation_warmup_keeps_live_cache_and_skips_disk_restore(self):
        state, mode = await self.scenario(disk_checkpoint=True)
        self.assertEqual(mode, "reuse-live-reconnect-root")
        state.prepare_conversation_checkpoint.assert_not_awaited()
        state.prepare_starter_cache.assert_not_awaited()
        state.erase_slot.assert_not_awaited()

    async def test_fork_warmup_keeps_common_prefix_after_checkpoint_miss(self):
        state, mode = await self.scenario(fork=True)
        self.assertEqual(mode, "reuse-live-cross-conversation-root")
        state.prepare_conversation_checkpoint.assert_awaited_once()
        state.prepare_starter_cache.assert_not_awaited()
        state.erase_slot.assert_not_awaited()

    async def test_native_base_warmup_preserves_full_followup_history(self):
        for fork, mode in ((False, "reuse-live-reconnect-root"),
                           (True, "reuse-live-cross-conversation-root")):
            with self.subTest(fork=fork):
                state, actual = await self.scenario(fork=fork, base_only=True)
                self.assertEqual(actual, mode)
                state.erase_slot.assert_not_awaited()

    async def test_lost_slot_without_checkpoint_uses_starter_fallback(self):
        state, mode = await self.scenario(slot_valid=False)
        self.assertEqual(mode, "build-starter-cache")
        state.prepare_conversation_checkpoint.assert_awaited_once()
        state.prepare_starter_cache.assert_awaited_once()

    async def test_fork_prefers_its_own_disk_checkpoint(self):
        state, mode = await self.scenario(fork=True, disk_checkpoint=True)
        self.assertEqual(mode, "restore-conversation-checkpoint")
        state.prepare_conversation_checkpoint.assert_awaited_once()

    async def test_lost_slot_uses_existing_disk_fallback(self):
        state, mode = await self.scenario(slot_valid=False, disk_checkpoint=True)
        self.assertEqual(mode, "restore-conversation-checkpoint")
        state.prepare_conversation_checkpoint.assert_awaited_once()

    async def test_changed_scaffold_does_not_reuse_unverified_live_cache(self):
        state, mode = await self.scenario(changed_scaffold=True)
        self.assertNotIn("reuse-live", mode)
        state.prepare_starter_cache.assert_awaited_once()

    async def test_delta_only_mode_does_not_use_full_prompt_reuse(self):
        state, mode = await self.scenario(delta_only=True)
        self.assertNotIn("reuse-live", mode)
        state.erase_slot.assert_awaited_once()

    async def branch_scenario(self, *, slot_valid=True, key="original",
                              changed_scaffold=False, changed_live_scaffold=False,
                              checkpoint="skipped", parent_file=False,
                              delta_only=False, starter=False):
        state, _ = await self.scenario()
        state._slot_has_cached_prompt.return_value = slot_valid
        state.experimental_delta_only = delta_only
        state.prepare_conversation_checkpoint.reset_mock()
        state.prepare_conversation_checkpoint.return_value = {
            "status": checkpoint, "mode": "restore-conversation-checkpoint"}
        state.restore_slot = mock.AsyncMock()
        state._run_responses_loop.side_effect = None
        state._run_responses_loop.return_value = (
            {"id": "resp_branch"}, [message("branch answer", "assistant")], 0)
        if parent_file:
            state.lineage["resp_seed"].snapshot_filename = "synthetic-parent.bin"
        if changed_live_scaffold:
            state.lineage["resp_reply"].scaffold_fingerprint = "other scaffold"
        if starter:
            state.starter_slot_models.add(fixture_profile().slug)
        delta = [message("question from the earlier response")]
        await state.process_websocket_create({
            "model": fixture_profile().slug, "tools": [], "prompt_cache_key": key,
            "instructions": "changed instructions" if changed_scaffold else "stable instructions",
            "previous_response_id": "resp_seed", "input": delta,
        })
        forwarded = state._run_responses_loop.await_args.kwargs["forward_request"]
        expected = delta if delta_only else [
            message("original user facts"), message("remembered fact", "assistant"), *delta]
        self.assertEqual(forwarded["input"], expected)
        self.assertNotIn("previous_response_id", forwarded)
        self.assertTrue(forwarded["cache_prompt"])
        completions = [c.args[1] for c in state.telemetry.emit.call_args_list
                       if c.args[0] == "router.response.completed"]
        return state, completions[-1]["slot"]["prepare_mode"]

    async def test_branch_without_snapshot_keeps_verified_shared_prefix(self):
        state, mode = await self.branch_scenario()
        self.assertEqual(mode, "reuse-live-branch-prefix")
        state.prepare_conversation_checkpoint.assert_awaited_once()
        state.erase_slot.assert_not_awaited()
        state.restore_slot.assert_not_awaited()

    async def test_branch_reuse_requires_valid_matching_full_prompt_state(self):
        cases = ({"slot_valid": False}, {"slot_valid": None}, {"key": "other"},
                 {"key": ""}, {"changed_scaffold": True}, {"changed_live_scaffold": True},
                 {"delta_only": True}, {"starter": True}, {"checkpoint": "error"})
        for options in cases:
            with self.subTest(**options):
                state, mode = await self.branch_scenario(**options)
                self.assertNotIn("reuse-live", mode)
                state.erase_slot.assert_awaited_once()

    async def test_branch_prefers_available_disk_checkpoints(self):
        for options, expected in (({"checkpoint": "restored"}, "restore-conversation-checkpoint"),
                                  ({"parent_file": True}, "restore-parent")):
            with self.subTest(**options):
                state, mode = await self.branch_scenario(**options)
                self.assertEqual(mode, expected)
                state.erase_slot.assert_not_awaited()
                if options.get("parent_file"):
                    state.restore_slot.assert_awaited_once()

    def test_generated_output_is_never_a_warmup(self):
        snapshot = router.ResponseSnapshot(
            response_id="warm_nonempty", profile_slug="test", parent_response_id=None,
            input_items=[message("history")], output_items=[], conversation_item_count=1,
            snapshot_filename="", instructions_text="", base_instructions_hash="",
            instructions_hash="", tools_hash="", prompt_cache_key="test", created_at=0,
        )
        self.assertTrue(router._is_warmup_root(snapshot))
        generated = copy.deepcopy(snapshot)
        generated.output_items = [message("answer", "assistant")]
        self.assertFalse(router._is_warmup_root(generated))
