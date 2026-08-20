from __future__ import annotations

import asyncio
import json
import sys
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
ROUTER_DIR = ROOT_DIR / "scripts" / "routers"
sys.path.insert(0, str(ROUTER_DIR))

import codex_local_router as router_module


def fixture_profile(context_window: int = 262_144) -> router_module.ModelProfile:
    return router_module.ModelProfile(
        slug="dynamic-model",
        alias="dynamic-model",
        display_name="Dynamic model",
        description="Test model",
        launcher="/bin/true",
        model_paths=("/tmp/model.gguf",),
        target="http://127.0.0.1:18000",
        context_window=context_window,
        auto_compact_token_limit=context_window * 9 // 10,
        truncation_limit=context_window * 85 // 100,
    )


class RouterContextTests(unittest.TestCase):
    def test_router_state_reports_empty_profile_catalog(self) -> None:
        with mock.patch.object(router_module, "_available_profiles", return_value={}):
            with self.assertRaisesRegex(
                RuntimeError, "no available local model profiles found"
            ):
                router_module.RouterState(
                    "missing-model",
                    Path("/tmp/marathon-test-state"),
                    Path("/tmp/marathon-test-logs"),
                )

    def test_closed_client_transport_is_not_a_backend_error(self) -> None:
        self.assertTrue(
            router_module._is_client_disconnect(
                ConnectionResetError("Cannot write to closing transport")
            )
        )
        self.assertFalse(router_module._is_client_disconnect(RuntimeError("kernel failed")))

    def test_custom_supervised_backend_can_use_native_alias_without_slots(self) -> None:
        environment = {
            "MARATHON_MODEL_PATH": "/tmp/model.gguf",
            "MARATHON_MODEL_SLUG": "public-model-id",
            "MARATHON_BACKEND_MODEL_ID": "deepseek-v4-flash",
            "MARATHON_BACKEND_SLOT_API": "0",
            "MARATHON_MODEL_SUPERVISED": "1",
            "MARATHON_MODEL_TEMPERATURE": "0",
        }
        with mock.patch.dict(router_module.os.environ, environment, clear=True):
            profile = router_module._custom_model_profile(ROOT_DIR)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.slug, "public-model-id")
        self.assertEqual(profile.alias, "deepseek-v4-flash")
        self.assertFalse(profile.supports_slots)
        self.assertTrue(profile.supervised)
        self.assertEqual(profile.temperature, 0.0)

    def test_custom_profile_loads_reasoning_capabilities(self) -> None:
        environment = {
            "MARATHON_MODEL_PATH": "/tmp/model.gguf",
            "MARATHON_MODEL_DEFAULT_REASONING_LEVEL": "xhigh",
            "MARATHON_MODEL_REASONING_LEVELS": json.dumps(
                [
                    {"effort": "low", "description": "Fast"},
                    {"effort": "medium", "description": "Balanced"},
                    {"effort": "xhigh", "description": "Deep"},
                ]
            ),
        }
        with mock.patch.dict(router_module.os.environ, environment, clear=True):
            profile = router_module._custom_model_profile(ROOT_DIR)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.default_reasoning_level, "xhigh")
        self.assertEqual(
            profile.supported_reasoning_levels,
            (("low", "Fast"), ("medium", "Balanced"), ("xhigh", "Deep")),
        )

    def test_custom_profile_loads_image_capability(self) -> None:
        environment = {
            "MARATHON_MODEL_PATH": "/tmp/model.gguf",
            "MARATHON_MODEL_INPUT_MODALITIES": "text,image",
        }
        with mock.patch.dict(router_module.os.environ, environment, clear=True):
            profile = router_module._custom_model_profile(ROOT_DIR)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.input_modalities, ("text", "image"))

    def test_catalog_advertises_full_dynamic_context_window(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.available_profiles = {profile.slug: profile}
        state._refresh_profiles = lambda: state.available_profiles

        with mock.patch.object(router_module, "_base_instructions", return_value="prompt"):
            catalog = state.model_catalog()
            model = catalog["models"][0]
            openai_model = catalog["data"][0]

        self.assertEqual(model["context_window"], 262_144)
        self.assertEqual(model["max_context_window"], 262_144)
        self.assertFalse(model["supports_parallel_tool_calls"])
        self.assertEqual(model["auto_compact_token_limit"], 235_929)
        self.assertEqual(model["effective_context_window_percent"], 100)
        self.assertEqual(openai_model["context_length"], 262_144)
        self.assertEqual(openai_model["max_model_len"], 262_144)

    def test_catalog_advertises_profile_parallel_tool_capability(self) -> None:
        profile = replace(fixture_profile(), supports_parallel_tool_calls=True)
        state = object.__new__(router_module.RouterState)
        state.available_profiles = {profile.slug: profile}
        state._refresh_profiles = lambda: state.available_profiles

        with mock.patch.object(router_module, "_base_instructions", return_value="prompt"):
            model = state.model_catalog()["models"][0]

        self.assertTrue(model["supports_parallel_tool_calls"])

    def test_catalog_advertises_profile_image_capability(self) -> None:
        profile = replace(
            fixture_profile(), input_modalities=("text", "image")
        )
        state = object.__new__(router_module.RouterState)
        state.available_profiles = {profile.slug: profile}
        state._refresh_profiles = lambda: state.available_profiles

        with mock.patch.object(router_module, "_base_instructions", return_value="prompt"):
            model = state.model_catalog()["models"][0]

        self.assertEqual(model["input_modalities"], ["text", "image"])

    def test_catalog_advertises_profile_reasoning_levels(self) -> None:
        profile = replace(
            fixture_profile(),
            default_reasoning_level="xhigh",
            supported_reasoning_levels=(
                ("low", "Fast"),
                ("medium", "Balanced"),
                ("xhigh", "Deep"),
            ),
        )
        state = object.__new__(router_module.RouterState)
        state.available_profiles = {profile.slug: profile}
        state._refresh_profiles = lambda: state.available_profiles

        with mock.patch.object(router_module, "_base_instructions", return_value="prompt"):
            model = state.model_catalog()["models"][0]

        self.assertEqual(model["default_reasoning_level"], "xhigh")
        self.assertEqual(
            model["supported_reasoning_levels"],
            [
                {"effort": "low", "description": "Fast"},
                {"effort": "medium", "description": "Balanced"},
                {"effort": "xhigh", "description": "Deep"},
            ],
        )

    def test_native_reasoning_effort_reaches_llama_template(self) -> None:
        profile = replace(
            fixture_profile(),
            default_reasoning_level="xhigh",
            supported_reasoning_levels=(
                ("low", "Fast"),
                ("medium", "Balanced"),
                ("xhigh", "Deep"),
            ),
        )
        request = {
            "input": [],
            "reasoning": {"effort": "low"},
            "chat_template_kwargs": {
                "preserve_reasoning": True,
                "enable_thinking": False,
            },
        }

        normalized = router_module.normalize_responses_request(request, profile)

        self.assertEqual(
            normalized["chat_template_kwargs"],
            {
                "preserve_reasoning": True,
                "enable_thinking": True,
                "reasoning_effort": "low",
            },
        )

    def test_image_tool_output_becomes_backend_image_message(self) -> None:
        image_url = "data:image/png;base64,aGVsbG8="
        request = {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_view",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "high",
                        }
                    ],
                }
            ]
        }

        normalized = router_module.normalize_responses_request(request)

        self.assertEqual(
            normalized["input"],
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_view",
                    "output": "Image attached in the following user message.",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "high",
                        }
                    ],
                },
            ],
        )

    def test_no_reasoning_disables_thinking_and_clears_effort(self) -> None:
        profile = replace(
            fixture_profile(),
            default_reasoning_level="xhigh",
            supported_reasoning_levels=(
                ("none", "Direct"),
                ("low", "Fast"),
                ("xhigh", "Deep"),
            ),
        )
        request = {
            "input": [],
            "reasoning": {"effort": "none"},
            "chat_template_kwargs": {
                "preserve_reasoning": True,
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            },
        }

        normalized = router_module.normalize_responses_request(request, profile)

        self.assertEqual(
            normalized["chat_template_kwargs"],
            {"preserve_reasoning": True, "enable_thinking": False},
        )

    def test_unsupported_reasoning_effort_is_rejected(self) -> None:
        profile = replace(
            fixture_profile(),
            default_reasoning_level="xhigh",
            supported_reasoning_levels=(("low", "Fast"), ("xhigh", "Deep")),
        )

        with self.assertRaisesRegex(ValueError, "choose one of: low, xhigh"):
            router_module.normalize_responses_request(
                {"input": [], "reasoning": {"effort": "ultra"}}, profile
            )

    def test_http_rejects_unsupported_reasoning_effort_as_bad_request(self) -> None:
        profile = replace(
            fixture_profile(),
            default_reasoning_level="xhigh",
            supported_reasoning_levels=(("low", "Fast"), ("xhigh", "Deep")),
        )
        state = object.__new__(router_module.RouterState)
        state.debug = False
        state.telemetry = mock.Mock()
        state.ensure_model_async = mock.AsyncMock(return_value=profile)

        class Request:
            app = {"state": state}
            path = "/v1/responses"
            method = "POST"
            headers: dict[str, str] = {}

            async def read(self) -> bytes:
                return json.dumps(
                    {
                        "model": profile.slug,
                        "input": [],
                        "reasoning": {"effort": "ultra"},
                    }
                ).encode()

        response = asyncio.run(router_module.handle_http_proxy(Request()))
        payload = json.loads(response.body)

        self.assertEqual(response.status, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("choose one of: low, xhigh", payload["error"]["message"])

    def test_managed_tool_loop_reports_final_context_not_cumulative_usage(self) -> None:
        profile = fixture_profile(131_072)
        state = object.__new__(router_module.RouterState)
        state.web_search_settings = SimpleNamespace(max_iterations=3)
        first = {
            "output": [
                {
                    "type": "function_call",
                    "name": "web_search",
                    "call_id": "call_1",
                    "arguments": '{"query":"local inference"}',
                }
            ],
            "usage": {
                "input_tokens": 10_000,
                "output_tokens": 50,
                "total_tokens": 10_050,
                "input_tokens_details": {"cached_tokens": 9_000},
            },
        }
        final = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {
                "input_tokens": 12_000,
                "output_tokens": 75,
                "total_tokens": 12_075,
                "input_tokens_details": {"cached_tokens": 10_000},
            },
        }
        state._request_json = mock.AsyncMock(side_effect=[first, final])
        state._execute_managed_call = mock.AsyncMock(
            return_value={
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result",
            }
        )

        response, _items, iterations = asyncio.run(
            state._run_responses_loop(
                profile=profile,
                forward_request={"input": [], "tools": []},
                web_search_enabled=True,
            )
        )

        self.assertEqual(iterations, 1)
        self.assertEqual(response["usage"]["total_tokens"], 12_075)
        self.assertEqual(
            response["usage"]["input_tokens_details"]["cached_tokens"], 10_000
        )

    def test_tool_outputs_are_bounded_with_head_and_tail_preserved(self) -> None:
        payload = {
            "instructions": "base",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "HEAD" + "x" * 200 + "TAIL",
                }
            ],
        }
        with mock.patch.dict(
            "os.environ", {"MARATHON_TOOL_OUTPUT_MAX_CHARS": "80"}, clear=False
        ):
            normalized = router_module.normalize_responses_request(payload)

        output = normalized["input"][0]["output"]
        self.assertLessEqual(len(output), 80)
        self.assertTrue(output.startswith("HEAD"))
        self.assertTrue(output.endswith("TAIL"))
        self.assertIn("Marathon truncated", output)
        self.assertEqual(normalized["_marathon_tool_output_truncations"], 1)

    def test_patch_adapter_removes_unsupported_numeric_hunk_headers(self) -> None:
        patch = """*** Begin Patch
*** Update File: example.py
-1,3 +1,4 @@
-old
+new
*** End Patch"""
        arguments = json.dumps({"input": patch})

        normalized = router_module._apply_patch_input_from_arguments(arguments)

        self.assertIn("\n@@\n-old\n+new\n", normalized)
        self.assertNotIn("-1,3 +1,4 @@", normalized)

    def test_structured_patch_operations_compile_to_native_patch_grammar(self) -> None:
        arguments = {
            "operations": [
                {
                    "action": "replace",
                    "path": "src/example.js",
                    "old_text": "const oldValue = 1;\n",
                    "new_text": "const newValue = 2;\n",
                },
                {"action": "add", "path": "notes.txt", "content": "one\ntwo\n"},
                {"action": "delete", "path": "obsolete.txt"},
            ]
        }

        compiled = router_module._apply_patch_input_from_arguments(arguments)

        self.assertEqual(
            compiled,
            "*** Begin Patch\n"
            "*** Update File: src/example.js\n"
            "@@\n"
            "-const oldValue = 1;\n"
            "+const newValue = 2;\n"
            "*** Add File: notes.txt\n"
            "+one\n+two\n"
            "*** Delete File: obsolete.txt\n"
            "*** End Patch",
        )

    def test_patch_schema_requires_action_specific_fields(self) -> None:
        tool = router_module._apply_patch_function_tool()
        variants = tool["parameters"]["properties"]["operations"]["items"]["oneOf"]

        self.assertTrue(tool["strict"])
        self.assertEqual(
            [variant["required"] for variant in variants],
            [
                ["action", "path", "content"],
                ["action", "path", "old_text", "new_text"],
                ["action", "path"],
            ],
        )

    def test_structured_patch_rejects_embedded_patch_envelopes(self) -> None:
        compiled = router_module._structured_patch_to_input(
            [
                {
                    "action": "add",
                    "path": "example.c",
                    "content": "int main(void) { return 0; }\n*** End Patch\n",
                }
            ]
        )

        self.assertEqual(compiled, "")

    def test_tool_argument_guard_rejects_loops_invalid_json_and_oversize(self) -> None:
        repeated = "</tool_call>" * 200
        self.assertTrue(router_module._has_runaway_repetition(repeated))
        self.assertFalse(
            router_module._has_runaway_repetition(
                "int parse_json(const char *input) {\n    return input != NULL;\n}\n"
                * 20
            )
        )
        self.assertIn(
            "structured JSON",
            router_module._apply_patch_protocol_error(
                {
                    "type": "function_call",
                    "name": "apply_patch",
                    "arguments": '{"operations":[{"action":"add","path":"x.c",',
                },
                24_576,
            ),
        )
        self.assertIn(
            "exceeded",
            router_module._partial_tool_argument_error("x" * 101, 100),
        )

    def test_patch_lineage_and_output_replay_stay_backend_native(self) -> None:
        custom_call = {
            "type": "custom_tool_call",
            "name": "apply_patch",
            "call_id": "patch_1",
            "input": "*** Begin Patch\n*** End Patch",
        }
        backend_call = router_module._backend_lineage_item(custom_call)
        self.assertEqual(backend_call["type"], "function_call")
        self.assertIn("input", json.loads(backend_call["arguments"]))

        structured_arguments = json.dumps(
            {
                "operations": [
                    {
                        "action": "add",
                        "path": "example.txt",
                        "content": "hello",
                    }
                ]
            }
        )
        custom_call["input"] = router_module._apply_patch_input_from_arguments(
            structured_arguments
        )
        custom_call[router_module._BACKEND_ARGUMENTS_KEY] = structured_arguments
        backend_call = router_module._backend_lineage_item(custom_call)
        self.assertEqual(backend_call["arguments"], structured_arguments)
        self.assertNotIn(router_module._BACKEND_ARGUMENTS_KEY, backend_call)

        custom_call[router_module._BACKEND_ARGUMENTS_KEY] = '{"operations":['
        backend_call = router_module._backend_lineage_item(custom_call)
        self.assertNotEqual(backend_call["arguments"], '{"operations":[')
        self.assertEqual(
            json.loads(backend_call["arguments"])["input"],
            custom_call["input"],
        )

        normalized = router_module.normalize_responses_request(
            {
                "input": [
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "patch_1",
                        "output": "Done!",
                    }
                ]
            }
        )
        self.assertEqual(normalized["input"][0]["type"], "function_call_output")

    def test_malformed_replayed_tool_pair_is_omitted(self) -> None:
        malformed_call = {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "broken_call",
            "arguments": '{"cmd":"unterminated',
        }
        malformed_output = {
            "type": "function_call_output",
            "call_id": "broken_call",
            "output": "failed to parse function arguments",
        }
        valid_call = {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "valid_call",
            "arguments": '{"cmd":"pwd"}',
        }
        valid_output = {
            "type": "function_call_output",
            "call_id": "valid_call",
            "output": "/workspace",
        }

        normalized = router_module.normalize_responses_request(
            {
                "input": [
                    malformed_call,
                    malformed_output,
                    valid_call,
                    valid_output,
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "continue"}],
                    },
                ]
            }
        )

        self.assertEqual(
            [item.get("call_id") for item in normalized["input"][:2]],
            ["valid_call", "valid_call"],
        )
        self.assertEqual(normalized["_marathon_malformed_tool_replay_drops"], 2)

    def test_completed_message_stays_commentary_when_tool_precedes_it(self) -> None:
        items = [
            {"type": "function_call", "name": "exec_command"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will test it."}],
            },
        ]

        self.assertEqual(router_module._completed_message_phase(items), "commentary")
        self.assertEqual(
            router_module._completed_message_phase([items[1]]), "final_answer"
        )

    def test_lifted_instructions_persist_until_base_instructions_change(self) -> None:
        parent = router_module.ResponseSnapshot(
            response_id="resp_1",
            profile_slug="dynamic-model",
            conversation_items=[],
            snapshot_filename="",
            instructions_text="base\n\ndeveloper policy",
            base_instructions_hash="base-hash",
            instructions_hash="effective-hash",
            tools_hash="tools-hash",
            prompt_cache_key="session",
            created_at=0,
        )

        retained = router_module._effective_instructions_for_request(
            parent, "base", "base-hash", 0
        )
        changed = router_module._effective_instructions_for_request(
            parent, "new base", "new-base-hash", 0
        )

        self.assertEqual(retained, "base\n\ndeveloper policy")
        self.assertEqual(changed, "new base")

    def test_reconnect_root_reuses_only_same_live_prompt_cache_session(self) -> None:
        self.assertTrue(
            router_module._can_reuse_reconnect_root(
                "model", "session-a", {"model": "resp"}, {"model": "session-a"}
            )
        )
        self.assertFalse(
            router_module._can_reuse_reconnect_root(
                "model", "session-b", {"model": "resp"}, {"model": "session-a"}
            )
        )

    def test_new_conversation_preserves_token_exact_root_prefix(self) -> None:
        live_slots = {"model": "resp"}
        live_cache_keys = {"model": "session-a"}

        self.assertEqual(
            router_module._root_prompt_cache_mode(
                "model", "session-a", live_slots, live_cache_keys
            ),
            "reuse-live-reconnect-root",
        )
        self.assertEqual(
            router_module._root_prompt_cache_mode(
                "model", "session-b", live_slots, live_cache_keys
            ),
            "reuse-live-cross-conversation-root",
        )
        self.assertEqual(
            router_module._root_prompt_cache_mode(
                "model", "session-b", {}, {}
            ),
            "reuse-backend-root-prefix",
        )

    def test_new_conversation_does_not_erase_live_llama_slot(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.ensure_model_async = mock.AsyncMock(return_value=profile)
        state.lineage_lock = asyncio.Lock()
        state.lineage = {}
        state.last_response_by_model = {profile.slug: "resp_old"}
        state.live_slot_by_model = {profile.slug: "resp_old"}
        state.live_prompt_cache_key_by_model = {profile.slug: "session-a"}
        state.experimental_delta_only = False
        state.slot_id = 0
        state.backend_lock = asyncio.Lock()
        state.slot_snapshots_enabled = False
        state.erase_slot = mock.AsyncMock()
        state._run_responses_loop = mock.AsyncMock(
            return_value=(
                {
                    "id": "resp_new",
                    "usage": {
                        "input_tokens": 10_500,
                        "input_tokens_details": {"cached_tokens": 10_000},
                    },
                },
                [],
                0,
            )
        )
        state.telemetry = mock.Mock()
        state.trace_request = mock.Mock()
        state.lock = threading.Lock()
        state._trace_seq = 0
        state._response_id_seq = 0
        state.debug = False

        result = asyncio.run(
            state.process_websocket_create(
                {
                    "model": profile.slug,
                    "prompt_cache_key": "session-b",
                    "instructions": "stable system prompt",
                    "tools": [],
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "new conversation"}
                            ],
                        }
                    ],
                }
            )
        )

        state.erase_slot.assert_not_awaited()
        forwarded = state._run_responses_loop.await_args.kwargs["forward_request"]
        self.assertTrue(forwarded["cache_prompt"])
        self.assertEqual(forwarded["id_slot"], 0)
        completed = [
            call.args[1]
            for call in state.telemetry.emit.call_args_list
            if call.args[0] == "router.response.completed"
        ][0]
        self.assertEqual(
            completed["slot"]["prepare_mode"],
            "reuse-live-cross-conversation-root",
        )
        self.assertEqual(result["usage"]["input_tokens_details"]["cached_tokens"], 10_000)

    def test_ws_scope_supersedes_only_the_same_generating_session(self) -> None:
        request = {
            "model": "model-a",
            "prompt_cache_key": "session-a",
        }
        self.assertEqual(
            router_module._active_ws_request_scope(request),
            "model-a\0session-a",
        )
        self.assertIsNone(
            router_module._active_ws_request_scope({**request, "generate": False})
        )
        self.assertIsNone(router_module._active_ws_request_scope({"model": "model-a"}))

        async def scenario() -> None:
            state = object.__new__(router_module.RouterState)
            state.active_ws_tasks = {}
            blocker = asyncio.Event()
            first = asyncio.create_task(blocker.wait())
            second = asyncio.create_task(asyncio.sleep(0))
            scope = router_module._active_ws_request_scope(request)
            self.assertIsNone(state.replace_active_ws_task(scope, first))
            self.assertIs(state.replace_active_ws_task(scope, second), first)
            first.cancel()
            await asyncio.gather(first, second, return_exceptions=True)
            await asyncio.sleep(0)
            self.assertEqual(state.active_ws_tasks, {})

        asyncio.run(scenario())

    def test_sse_comments_are_ignored(self) -> None:
        class Content:
            async def iter_chunked(self, _size: int):
                yield b": decode\n\n"
                yield b'data: {"type":"response.created"}\n\n'

        async def collect() -> list[dict[str, object]]:
            state = object.__new__(router_module.RouterState)
            response = SimpleNamespace(content=Content())
            return [event async for event in state._iter_sse_json(response)]

        events = asyncio.run(collect())
        self.assertEqual(events, [{"type": "response.created"}])

    def test_slow_live_backend_stream_is_not_canceled(self) -> None:
        profile = fixture_profile(65_536)

        class Content:
            async def iter_chunked(self, _size: int):
                await asyncio.sleep(0.02)
                yield (
                    b'data: {"type":"response.completed","response":'
                    b'{"id":"resp_slow","usage":{"output_tokens":1}}}\n\n'
                )

        class Response:
            status = 200
            content = Content()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class Client:
            def post(self, _url: str, *, json: dict[str, object]):
                self.request = json
                return Response()

        state = object.__new__(router_module.RouterState)
        state.http_client = Client()
        # This was the old fatal watchdog value. Keeping it on this lightweight
        # fixture makes the regression fail if that cancellation path returns.
        state.stream_idle_timeout_seconds = 0.001

        async def sink(_event: dict[str, object]) -> bool:
            return True

        response = asyncio.run(
            state._request_responses_stream(
                profile,
                {"input": [], "tools": []},
                event_sink=sink,
            )
        )
        self.assertEqual(response["id"], "resp_slow")
        self.assertEqual(response["usage"]["output_tokens"], 1)

    def test_output_budget_scales_and_is_profile_overrideable(self) -> None:
        self.assertEqual(router_module._max_output_tokens(fixture_profile(32_768)), 4_096)
        self.assertEqual(router_module._max_output_tokens(fixture_profile(65_536)), 8_192)
        self.assertEqual(router_module._max_output_tokens(fixture_profile(262_144)), 8_192)
        with mock.patch.dict("os.environ", {"MARATHON_MAX_OUTPUT_TOKENS": "6000"}):
            self.assertEqual(router_module._max_output_tokens(fixture_profile()), 6_000)

    def test_native_thinking_budget_only_applies_after_tool_output(self) -> None:
        request = {
            "tools": [{"type": "function", "name": "exec_command"}],
            "max_output_tokens": 8_192,
        }
        tool_output = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "tests failed",
            }
        ]

        with mock.patch.dict(
            "os.environ",
            {"MARATHON_MODEL_TOOL_THINKING_BUDGET_TOKENS": "1024"},
            clear=False,
        ):
            self.assertEqual(
                router_module._tool_thinking_budget_for_turn(request, tool_output),
                1_024,
            )
            self.assertIsNone(
                router_module._tool_thinking_budget_for_turn(
                    request,
                    [{"type": "message", "role": "user", "content": []}],
                )
            )

    def test_output_budget_stall_requires_no_actionable_output(self) -> None:
        stalled = {
            "usage": {"output_tokens": 8192},
            "output": [{"type": "reasoning"}],
        }
        self.assertTrue(
            router_module._response_stalled_at_output_limit(
                stalled, stalled["output"], 8192
            )
        )
        self.assertTrue(
            router_module._response_stalled_at_output_limit(
                stalled,
                [{"type": "message", "role": "assistant", "content": []}],
                8192,
            )
        )
        self.assertFalse(
            router_module._response_stalled_at_output_limit(
                stalled,
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "A complete answer."}
                        ],
                    }
                ],
                8192,
            )
        )
        self.assertFalse(
            router_module._response_stalled_at_output_limit(
                stalled,
                [{"type": "function_call", "name": "exec_command"}],
                8192,
            )
        )

    def test_backend_malformed_tool_call_http_error_is_recoverable(self) -> None:
        profile = fixture_profile(65_536)

        class Response:
            status = 500

            async def text(self) -> str:
                return (
                    '{"error":{"message":"Failed to parse tool call arguments '
                    'as JSON: missing closing quote"}}'
                )

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        class Client:
            def post(self, *_args, **_kwargs):
                return Response()

        state = object.__new__(router_module.RouterState)
        state.http_client = Client()

        with self.assertRaisesRegex(
            router_module.ToolProtocolError,
            "malformed or truncated tool-call arguments",
        ):
            asyncio.run(
                state._request_responses_stream(
                    profile,
                    {"input": [], "tools": []},
                )
            )

    def test_stalled_response_recovers_with_required_tool_action(self) -> None:
        profile = fixture_profile(65_536)
        state = object.__new__(router_module.RouterState)
        state.web_search_settings = SimpleNamespace(max_iterations=3)
        state.telemetry = mock.Mock()
        stalled = {
            "output": [{"type": "reasoning"}],
            "usage": {"input_tokens": 10_000, "output_tokens": 8_192},
        }
        recovered = {
            "output": [
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call_1",
                    "arguments": "{}",
                }
            ],
            "usage": {"input_tokens": 10_500, "output_tokens": 80},
        }
        state._request_json = mock.AsyncMock(side_effect=[stalled, recovered])

        response, items, _iterations = asyncio.run(
            state._run_responses_loop(
                profile=profile,
                forward_request={
                    "input": [],
                    "tools": [{"type": "function", "name": "exec_command"}],
                    "max_output_tokens": 8_192,
                },
                web_search_enabled=False,
            )
        )

        self.assertEqual(response["output"], recovered["output"])
        self.assertEqual(response["usage"]["output_tokens"], 80)
        second_request = state._request_json.await_args_list[1].args[3]
        self.assertEqual(second_request["tool_choice"], "required")
        self.assertEqual(second_request["input"][-1]["role"], "user")
        self.assertEqual(items[-1]["name"], "exec_command")
        state.telemetry.emit.assert_called_once_with(
            "router.response.stalled_recovery",
            mock.ANY,
            level="warning",
        )

    def test_tool_protocol_failure_retries_once_with_smaller_budget(self) -> None:
        profile = fixture_profile(65_536)
        state = object.__new__(router_module.RouterState)
        state.web_search_settings = SimpleNamespace(max_iterations=3)
        state.telemetry = mock.Mock()
        recovered = {
            "output": [
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call_1",
                    "arguments": '{}',
                }
            ],
            "usage": {"input_tokens": 10_500, "output_tokens": 80},
        }
        state._request_responses_stream = mock.AsyncMock(
            side_effect=[
                router_module.ToolProtocolError(
                    "apply_patch arguments were not valid structured JSON"
                ),
                recovered,
            ]
        )

        async def sink(_event: dict[str, object]) -> bool:
            return True

        response, items, _iterations = asyncio.run(
            state._run_responses_loop(
                profile=profile,
                forward_request={
                    "input": [],
                    "tools": [{"type": "function", "name": "apply_patch"}],
                    "max_output_tokens": 8_192,
                },
                web_search_enabled=False,
                event_sink=sink,
            )
        )

        self.assertEqual(response["output"], recovered["output"])
        self.assertEqual(response["usage"]["output_tokens"], 80)
        self.assertEqual(items[-1]["name"], "exec_command")
        retry = state._request_responses_stream.await_args_list[1].args[1]
        self.assertEqual(retry["tool_choice"], "required")
        self.assertEqual(retry["max_output_tokens"], 4_096)
        self.assertIn("split", retry["input"][-1]["content"][0]["text"].lower())
        state.telemetry.emit.assert_called_once_with(
            "router.response.tool_protocol_recovery",
            mock.ANY,
            level="warning",
        )

    def test_web_actions_are_replayed_from_turn_cache_after_reconnect(self) -> None:
        state = object.__new__(router_module.RouterState)
        state.web_tool_cache = router_module.OrderedDict()
        state.web_tool_cache_max_entries = 10
        state.telemetry = mock.Mock()
        state._execute_managed_call = mock.AsyncMock(
            return_value={
                "type": "function_call_output",
                "call_id": "first",
                "output": "search result",
            }
        )
        first = {
            "type": "function_call",
            "name": "web_search",
            "call_id": "first",
            "arguments": '{"query":"  Local   Inference "}',
        }
        retry = {
            **first,
            "call_id": "retry",
            "arguments": '{"query":"local inference"}',
        }

        async def run_calls():
            a = await state._execute_managed_call_cached(first, 0, "turn")
            b = await state._execute_managed_call_cached(retry, 0, "turn")
            return a, b

        original, replayed = asyncio.run(run_calls())

        state._execute_managed_call.assert_awaited_once()
        self.assertEqual(original["output"], replayed["output"])
        self.assertEqual(replayed["call_id"], "retry")
        state.telemetry.emit.assert_any_call(
            "router.web_tool.cache_hit",
            mock.ANY,
        )

    def test_managed_web_progress_survives_stream_disconnect(self) -> None:
        profile = fixture_profile(131_072)
        state = object.__new__(router_module.RouterState)
        state.web_search_settings = SimpleNamespace(max_iterations=5)
        state.web_tool_cache = router_module.OrderedDict()
        state.web_tool_cache_max_entries = 10
        state.telemetry = mock.Mock()
        state._execute_managed_call = mock.AsyncMock(
            return_value={
                "type": "function_call_output",
                "call_id": "search_1",
                "output": "durable search result",
            }
        )
        search_call = {
            "type": "function_call",
            "name": "web_search",
            "call_id": "search_1",
            "arguments": '{"query":"durable websocket state"}',
        }
        final_message = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "finished"}],
        }
        state._request_responses_stream = mock.AsyncMock(
            side_effect=[
                {"output": [search_call], "usage": {"output_tokens": 20}},
                {"output": [final_message], "usage": {"output_tokens": 10}},
            ]
        )
        request = {
            "model": profile.alias,
            "prompt_cache_key": "session-a",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "research this"}],
                }
            ],
            "tools": [router_module.web_search_function_tool()],
            "max_output_tokens": 8_192,
        }

        async def disconnected_sink(_event: dict[str, object]) -> bool:
            return False

        async def connected_sink(_event: dict[str, object]) -> bool:
            return True

        async def scenario():
            with self.assertRaisesRegex(ConnectionError, "client disconnected"):
                await state._run_responses_loop(
                    profile=profile,
                    forward_request=request,
                    web_search_enabled=True,
                    event_sink=disconnected_sink,
                )
            return await state._run_responses_loop(
                profile=profile,
                forward_request=request,
                web_search_enabled=True,
                event_sink=connected_sink,
            )

        response, items, iterations = asyncio.run(scenario())

        resumed_request = state._request_responses_stream.await_args_list[1].args[1]
        resumed_types = [item.get("type") for item in resumed_request["input"]]
        self.assertIn("function_call", resumed_types)
        self.assertIn("function_call_output", resumed_types)
        self.assertEqual(iterations, 1)
        self.assertEqual(items[-1]["content"], final_message["content"])
        self.assertEqual(items[-1]["phase"], "final_answer")
        state._execute_managed_call.assert_awaited_once()
        state.telemetry.emit.assert_any_call(
            "router.web_turn.resumed",
            mock.ANY,
        )

    def test_repeated_managed_web_call_forces_final_and_replays_completion(self) -> None:
        profile = fixture_profile(131_072)
        state = object.__new__(router_module.RouterState)
        state.web_search_settings = SimpleNamespace(max_iterations=5)
        state.web_tool_cache = router_module.OrderedDict()
        state.web_tool_cache_max_entries = 10
        state.telemetry = mock.Mock()
        state._execute_managed_call = mock.AsyncMock(
            return_value={
                "type": "function_call_output",
                "call_id": "search_1",
                "output": "one network result",
            }
        )

        def search(call_id: str) -> dict[str, object]:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "web_search",
                        "call_id": call_id,
                        "arguments": '{"query":"same query"}',
                    }
                ],
                "usage": {"output_tokens": 20},
            }

        final = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "final answer"}],
                }
            ],
            "usage": {"output_tokens": 10},
        }
        state._request_responses_stream = mock.AsyncMock(
            side_effect=[search("search_1"), search("search_2"), final]
        )
        request = {
            "model": profile.alias,
            "prompt_cache_key": "session-repeat",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "research this"}],
                }
            ],
            "tools": [router_module.web_search_function_tool()],
            "max_output_tokens": 8_192,
        }

        async def sink(_event: dict[str, object]) -> bool:
            return True

        async def scenario():
            first = await state._run_responses_loop(
                profile=profile,
                forward_request=request,
                web_search_enabled=True,
                event_sink=sink,
            )
            replayed = await state._run_responses_loop(
                profile=profile,
                forward_request=request,
                web_search_enabled=True,
                event_sink=sink,
            )
            return first, replayed

        first, replayed = asyncio.run(scenario())

        self.assertEqual(first[2], 2)
        self.assertEqual(state._request_responses_stream.await_count, 3)
        state._execute_managed_call.assert_awaited_once()
        final_request = state._request_responses_stream.await_args_list[2].args[1]
        final_tool_names = {
            tool.get("name") for tool in final_request.get("tools", [])
        }
        self.assertNotIn("web_search", final_tool_names)
        self.assertTrue(
            replayed[0].get(router_module._WEB_REPLAYED_COMPLETION_KEY)
        )
        self.assertEqual(replayed[1], first[1])
        state.telemetry.emit.assert_any_call(
            "router.web_tool.repeat_guard",
            mock.ANY,
            level="warning",
        )

    def test_web_turn_progress_is_bounded_and_expires(self) -> None:
        state = object.__new__(router_module.RouterState)
        state.web_turn_progress = router_module.OrderedDict()
        state.web_turn_progress_max_entries = 2
        state.web_turn_progress_ttl_seconds = 10

        def progress(updated_at: float) -> router_module.ManagedWebTurnProgress:
            return router_module.ManagedWebTurnProgress(
                request_suffix=[],
                cumulative_items=[],
                iterations=0,
                seen_signatures=set(),
                finalizing=False,
                completed_response=None,
                updated_at=updated_at,
            )

        now = router_module.time.time()
        state.web_turn_progress["expired"] = progress(now - 11)
        state.web_turn_progress["first"] = progress(now)
        state.save_web_turn_progress("second", progress(now))
        state.save_web_turn_progress("third", progress(now))

        self.assertEqual(list(state.web_turn_progress), ["second", "third"])
        self.assertIsNone(state.load_web_turn_progress("expired"))


if __name__ == "__main__":
    unittest.main()
