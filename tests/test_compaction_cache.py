"""Compaction must reuse only its own prompt and must never run tools."""
import asyncio
import copy
import json
import sys
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/routers"))
import codex_local_router as router


META = {"x-codex-turn-metadata": json.dumps({"request_kind": "compaction"})}
TOOLS = [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]


def profile():
    return router.ModelProfile(
        slug="test", alias="test", display_name="Test", description="Test",
        model_paths=(), target="http://127.0.0.1:18000", context_window=196000,
        auto_compact_token_limit=170000, truncation_limit=160000,
    )


def snapshot():
    return router.ResponseSnapshot(
        response_id="previous", profile_slug="test", parent_response_id=None,
        input_items=[], output_items=[], conversation_item_count=0, snapshot_filename="",
        instructions_text="policy", base_instructions_hash="base", instructions_hash="instructions",
        tools_hash="tools", prompt_cache_key="conversation", created_at=0,
        tool_scaffold={"tools": copy.deepcopy(TOOLS), "parallel_tool_calls": True},
    )


def request():
    return {"model": "test", "instructions": "policy", "tools": [],
            "prompt_cache_key": "conversation", "client_metadata": META,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Summarize the conversation"}]}]}


class CompactionCacheTests(unittest.TestCase):
    def test_scaffold_selection_requires_same_conversation_model_and_instructions(self):
        for change in ({}, {"prompt_cache_key": "different"}, {"prompt_cache_key": ""},
                       {"instructions": "new policy"}, {"tools": TOOLS},
                       {"client_metadata": {}}, {"previous_response_id": "previous"},
                       {"generate": False}):
            with self.subTest(change=change), mock.patch.dict(
                router.os.environ, {"MARATHON_COMPACTION_PREFIX_CACHE": "1"}
            ):
                state = object.__new__(router.RouterState)
                source = snapshot()
                source.tool_scaffold["instructions"] = "must not replace the current policy"
                state.lineage = {"previous": source}
                state.live_slot_by_model = {"test": "previous"}
                body = request() | copy.deepcopy(change)
                before = copy.deepcopy(body)
                selected = state._compaction_tool_scaffold(profile(), body)
                if change:
                    self.assertIsNone(selected)
                    self.assertEqual(body, before)
                else:
                    self.assertIs(selected, source)
                    self.assertEqual(body["tool_choice"], "none")
                    self.assertEqual(body["tools"], TOOLS)
                    self.assertTrue(body["parallel_tool_calls"])
                    self.assertEqual(body["instructions"], "policy")
                    self.assertEqual(body["input"], before["input"])
                    body["tools"][0]["name"] = "changed"
                    self.assertEqual(source.tool_scaffold["tools"], TOOLS)

    def test_unavailable_or_disabled_cache_falls_back(self):
        for flag, supports_slots, source in (
            ("0", True, snapshot()), ("1", False, snapshot()),
            ("1", True, replace(snapshot(), profile_slug="other")),
            ("1", True, replace(snapshot(), tool_scaffold=None)), ("1", True, None),
        ):
            with self.subTest(flag=flag, source=source), mock.patch.dict(
                router.os.environ, {"MARATHON_COMPACTION_PREFIX_CACHE": flag}
            ):
                state = object.__new__(router.RouterState)
                state.lineage = {"previous": source} if source else {}
                state.live_slot_by_model = {"test": "previous"}
                body = request()
                before = copy.deepcopy(body)
                self.assertIsNone(state._compaction_tool_scaffold(
                    replace(profile(), supports_slots=supports_slots), body))
                self.assertEqual(body, before)

    def test_compaction_uses_live_slot_but_does_not_trust_a_lost_slot(self):
        for slot_valid, switched in ((True, False), (False, False), (True, True)):
            with self.subTest(slot_valid=slot_valid, switched=switched), mock.patch.dict(
                router.os.environ, {"MARATHON_COMPACTION_PREFIX_CACHE": "1"}
            ):
                state = object.__new__(router.RouterState)
                state.ensure_model_async = mock.AsyncMock(return_value=profile())
                state.lineage_lock = asyncio.Lock()
                state.lineage = {"previous": snapshot()}
                state.last_response_by_model = {"test": "previous"}
                state.live_slot_by_model = {"test": "previous"}
                state.live_prompt_cache_key_by_model = {"test": "conversation"}
                state.starter_slot_models = set()
                state.experimental_delta_only = False
                state.slot_id = 0
                state.backend_lock = asyncio.Lock()
                state._slot_has_cached_prompt = mock.AsyncMock(return_value=slot_valid)
                state._checkpoint_before_conversation_switch_locked = mock.AsyncMock(return_value=None)
                if switched:
                    async def switch_slot(*_args):
                        state.live_slot_by_model["test"] = "other-response"
                        state.live_prompt_cache_key_by_model["test"] = "other-conversation"
                    state._checkpoint_before_conversation_switch_locked.side_effect = switch_slot
                state.prepare_starter_cache = mock.AsyncMock(return_value={"mode": "build-starter-cache", "status": "built"})
                state.prepare_conversation_checkpoint = mock.AsyncMock(return_value={"status": "skipped"})
                state.erase_slot = mock.AsyncMock()
                state._run_responses_loop = mock.AsyncMock(return_value=(
                    {"id": "summary", "usage": {"input_tokens": 100000, "output_tokens": 100}}, [], 0))
                state.schedule_conversation_checkpoint = mock.Mock(return_value={"status": "skipped"})
                state.telemetry = mock.Mock()
                state.trace_request = mock.Mock()
                state.lock = threading.Lock()
                state._trace_seq = 0
                state.debug = False
                state.web_search = None
                asyncio.run(state.process_websocket_create(request()))
                forwarded = state._run_responses_loop.await_args.kwargs["forward_request"]
                self.assertEqual(forwarded["tools"], TOOLS)
                self.assertEqual(forwarded["tool_choice"], "none")
                self.assertEqual(forwarded["input"], request()["input"])
                if slot_valid and not switched:
                    state.prepare_starter_cache.assert_not_awaited()
                    state.prepare_conversation_checkpoint.assert_not_awaited()
                else:
                    state.prepare_starter_cache.assert_awaited_once()
                state.erase_slot.assert_not_awaited()

    def test_compaction_rejects_local_and_managed_tools_before_execution(self):
        for name in ("exec_command", "web_search", "web_fetch"):
            with self.subTest(name=name):
                state = object.__new__(router.RouterState)
                state.web_search_settings = SimpleNamespace(max_iterations=3)
                state.telemetry = mock.Mock()
                state._request_json = mock.AsyncMock(return_value={"output": [
                    {"type": "function_call", "name": name, "call_id": "call", "arguments": "{}"}],
                    "usage": {"input_tokens": 10000, "output_tokens": 10}})
                with self.assertRaisesRegex(RuntimeError, "tools are disabled during compaction"):
                    asyncio.run(state._run_responses_loop(
                        profile=profile(), forward_request=request() | {"tools": TOOLS, "tool_choice": "none"},
                        web_search_enabled=False))
                state._request_json.assert_awaited_once()

    def test_stream_does_not_forward_compaction_tool_events(self):
        events = [
            {"type": "response.output_item.added", "item": {"type": "function_call", "name": "exec_command"}},
            {"type": "response.function_call_arguments.delta", "delta": "{}"},
        ]
        for event in events:
            class Content:
                async def iter_chunked(self, _size):
                    yield f"data: {json.dumps(event)}\n\n".encode()

            class Response:
                status = 200
                content = Content()

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return False

            state = object.__new__(router.RouterState)
            state.http_client = SimpleNamespace(post=lambda *_args, **_kwargs: Response())
            sink = mock.AsyncMock(return_value=True)
            with self.subTest(event=event), self.assertRaisesRegex(RuntimeError, "tools are disabled during compaction"):
                asyncio.run(state._request_responses_stream(profile(), request(), event_sink=sink))
            sink.assert_not_awaited()

    def test_stalled_compaction_never_enables_tools_on_recovery(self):
        state = object.__new__(router.RouterState)
        state.web_search_settings = SimpleNamespace(max_iterations=3)
        state.telemetry = mock.Mock()
        state._request_json = mock.AsyncMock(side_effect=[
            {"output": [{"type": "reasoning"}], "usage": {"output_tokens": 8192}},
            {"output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Valid context summary"}]}], "usage": {"output_tokens": 20}},
        ])
        asyncio.run(state._run_responses_loop(
            profile=profile(), forward_request=request() | {"tools": TOOLS, "tool_choice": "none"},
            web_search_enabled=False))
        recovery = state._request_json.await_args_list[1].args[3]
        self.assertEqual(recovery["tool_choice"], "none")
        self.assertIn("context summary", recovery["input"][-1]["content"][0]["text"])
