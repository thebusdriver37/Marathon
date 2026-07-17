from __future__ import annotations

import asyncio
import json
import sys
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
    def test_catalog_advertises_full_dynamic_context_window(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.available_profiles = {profile.slug: profile}
        state._refresh_profiles = lambda: state.available_profiles

        with mock.patch.object(router_module, "_base_instructions", return_value="prompt"):
            model = state.model_catalog()["models"][0]

        self.assertEqual(model["context_window"], 262_144)
        self.assertEqual(model["max_context_window"], 262_144)
        self.assertFalse(model["supports_parallel_tool_calls"])
        self.assertEqual(model["auto_compact_token_limit"], 235_929)
        self.assertEqual(model["effective_context_window_percent"], 100)

    def test_catalog_advertises_profile_parallel_tool_capability(self) -> None:
        profile = replace(fixture_profile(), supports_parallel_tool_calls=True)
        state = object.__new__(router_module.RouterState)
        state.available_profiles = {profile.slug: profile}
        state._refresh_profiles = lambda: state.available_profiles

        with mock.patch.object(router_module, "_base_instructions", return_value="prompt"):
            model = state.model_catalog()["models"][0]

        self.assertTrue(model["supports_parallel_tool_calls"])

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
        self.assertFalse(
            router_module._response_stalled_at_output_limit(
                stalled,
                [{"type": "function_call", "name": "exec_command"}],
                8192,
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


if __name__ == "__main__":
    unittest.main()
