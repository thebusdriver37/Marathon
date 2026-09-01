from __future__ import annotations

import asyncio
import copy
import json
import sys
import tempfile
import threading
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
ROUTER_DIR = ROOT_DIR / "scripts" / "routers"
sys.path.insert(0, str(ROUTER_DIR))

import codex_local_router as router_module
from marathon_web_search import SearchOutcome
from marathon_web_search import SearchResult


def fixture_profile(context_window: int = 262_144) -> router_module.ModelProfile:
    return router_module.ModelProfile(
        slug="dynamic-model",
        alias="dynamic-model",
        display_name="Dynamic model",
        description="Test model",
        model_paths=("/tmp/model.gguf",),
        target="http://127.0.0.1:18000",
        context_window=context_window,
        auto_compact_token_limit=context_window * 9 // 10,
        truncation_limit=context_window * 85 // 100,
    )


class RouterContextTests(unittest.TestCase):
    def test_base_instructions_include_marathon_runtime_safety(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.md"
            prompt_path.write_text("Upstream Codex prompt.\n", encoding="utf-8")
            with mock.patch.dict(
                router_module.os.environ,
                {"MARATHON_PROMPT_FILE": str(prompt_path)},
                clear=False,
            ):
                instructions = router_module._base_instructions()

        self.assertEqual(
            instructions,
            "Upstream Codex prompt.\n\n"
            + router_module.MARATHON_RUNTIME_INSTRUCTIONS,
        )

    def test_web_search_cache_signature_normalizes_but_distinguishes_time_range(self) -> None:
        def call(query: str, time_range: str) -> dict[str, object]:
            return {
                "type": "function_call",
                "name": "web_search",
                "arguments": json.dumps({"query": query, "time_range": time_range}),
            }

        week_a = router_module._managed_call_signature(
            call(" Recent   release ", "Week")
        )
        week_b = router_module._managed_call_signature(call("recent release", "week"))
        month = router_module._managed_call_signature(call("recent release", "month"))

        self.assertEqual(week_a, week_b)
        self.assertNotEqual(week_a, month)

    def test_web_search_forwards_time_range_and_includes_it_in_tool_output(self) -> None:
        state = object.__new__(router_module.RouterState)
        state.web_search = mock.Mock()
        state.web_search.search_with_diagnostics = mock.AsyncMock(
            return_value=SearchOutcome(
                results=[
                    SearchResult(
                        title="Recent result",
                        url="https://example.test/recent",
                        snippet="Published recently.",
                        engine="google cse",
                    )
                ]
            )
        )
        state.web_search_settings = SimpleNamespace(base_url="http://search.test")
        state.log_dir = Path("/tmp")
        item = {
            "type": "function_call",
            "name": "web_search",
            "call_id": "search_1",
            "arguments": json.dumps(
                {"query": "recent release", "max_results": 4, "time_range": "week"}
            ),
        }

        output = asyncio.run(state._execute_web_search_call(item, 0))

        state.web_search.search_with_diagnostics.assert_awaited_once_with(
            "recent release",
            max_results=4,
            time_range="week",
        )
        self.assertIn("time range: week", output["output"])

    def test_web_search_rejects_invalid_time_range_before_execution(self) -> None:
        state = object.__new__(router_module.RouterState)
        state.web_search = mock.Mock()
        state.web_search.search_with_diagnostics = mock.AsyncMock()
        item = {
            "type": "function_call",
            "name": "web_search",
            "call_id": "search_1",
            "arguments": '{"query":"recent release","time_range":"forever"}',
        }

        output = asyncio.run(state._execute_web_search_call(item, 0))

        state.web_search.search_with_diagnostics.assert_not_awaited()
        self.assertIn("day, week, month, year", output["output"])

    def test_external_model_loads_from_the_existing_user_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.toml"
            catalog_path.write_text(
                """
[[external_models]]
id = "deepseek-spark"
model = "spark/deepseek-v4-flash-0731-spark"
display_name = "DeepSeek V4 Flash on Spark"
description = "Remote coding model"
base_url = "http://127.0.0.1:9292/v1"
api_key_env = "SPARK_API_KEY"
context = 262144
auto_compact_token_limit = 229376
truncation_limit = 221184
temperature = 0.0
""".strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                router_module.os.environ,
                {
                    "MARATHON_USER_CATALOG": str(catalog_path),
                    "SPARK_API_KEY": "test-secret",
                },
                clear=True,
            ):
                profiles = router_module._external_model_profiles()
                available = router_module._available_profiles()
                headers = profiles["deepseek-spark"].upstream_headers()

        profile = profiles["deepseek-spark"]
        self.assertEqual(profile.alias, "spark/deepseek-v4-flash-0731-spark")
        self.assertEqual(profile.target, "http://127.0.0.1:9292")
        self.assertEqual(profile.context_window, 262_144)
        self.assertEqual(profile.auto_compact_token_limit, 229_376)
        self.assertEqual(profile.truncation_limit, 221_184)
        self.assertEqual(profile.temperature, 0.0)
        self.assertTrue(profile.external)
        self.assertFalse(profile.supports_slots)
        self.assertIn("deepseek-spark", available)
        self.assertEqual(headers, {"Authorization": "Bearer test-secret"})

    def test_external_model_reports_a_missing_api_key_when_selected(self) -> None:
        profile = replace(
            fixture_profile(),
            external=True,
            supports_slots=False,
            api_key_env="MISSING_EXTERNAL_KEY",
        )
        with mock.patch.dict(router_module.os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "MISSING_EXTERNAL_KEY",
            ):
                profile.upstream_headers()

    def test_external_model_can_read_one_named_key_from_an_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / ".env"
            key_file.write_text(
                "UNRELATED=ignore-me\nSPARK_API_KEY=file-secret\n",
                encoding="utf-8",
            )
            profile = replace(
                fixture_profile(),
                external=True,
                supports_slots=False,
                api_key_env="SPARK_API_KEY",
                api_key_file=str(key_file),
            )
            with mock.patch.dict(router_module.os.environ, {}, clear=True):
                headers = profile.upstream_headers()

        self.assertEqual(headers, {"Authorization": "Bearer file-secret"})

    def test_external_proxy_does_not_forward_client_credentials(self) -> None:
        profile = replace(
            fixture_profile(),
            external=True,
            api_key_env="EXTERNAL_API_KEY",
        )
        with mock.patch.dict(
            router_module.os.environ,
            {"EXTERNAL_API_KEY": "external-secret"},
            clear=True,
        ):
            headers = router_module._proxy_request_headers(
                profile,
                {
                    "Authorization": "Bearer client-secret",
                    "Cookie": "session=client-secret",
                    "Host": "127.0.0.1:8080",
                    "X-Request-ID": "trace-1",
                },
            )

        self.assertEqual(headers["Authorization"], "Bearer external-secret")
        self.assertNotIn("Cookie", headers)
        self.assertNotIn("Host", headers)
        self.assertEqual(headers["X-Request-ID"], "trace-1")

    def test_external_model_rejects_a_local_model_id_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.toml"
            catalog_path.write_text(
                """
[[external_models]]
id = "custom"
model = "remote-model"
display_name = "Collision"
base_url = "http://127.0.0.1:9292/v1"
context = 32768
""".strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                router_module.os.environ,
                {
                    "MARATHON_MODEL_PATH": "/tmp/local-model.gguf",
                    "MARATHON_USER_CATALOG": str(catalog_path),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "conflicts with a local model"):
                    router_module._profiles()

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

    def test_custom_backend_can_use_native_alias_without_slots(self) -> None:
        environment = {
            "MARATHON_MODEL_PATH": "/tmp/model.gguf",
            "MARATHON_MODEL_SLUG": "public-model-id",
            "MARATHON_BACKEND_MODEL_ID": "deepseek-v4-flash",
            "MARATHON_BACKEND_SLOT_API": "0",
            "MARATHON_MODEL_TEMPERATURE": "0",
        }
        with mock.patch.dict(router_module.os.environ, environment, clear=True):
            profile = router_module._custom_model_profile(ROOT_DIR)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.slug, "public-model-id")
        self.assertEqual(profile.alias, "deepseek-v4-flash")
        self.assertFalse(profile.supports_slots)
        self.assertEqual(profile.temperature, 0.0)

    def test_custom_backend_can_use_authenticated_external_slot_api(self) -> None:
        environment = {
            "MARATHON_MODEL_PATH": "/tmp/model.gguf",
            "MARATHON_MODEL_SLUG": "pooled-model",
            "MARATHON_BACKEND_MODEL_ID": "upstream-model",
            "MARATHON_BACKEND_SLOT_API": "1",
            "MARATHON_MODEL_EXTERNAL": "1",
            "MARATHON_MODEL_API_KEY_ENV": "POOL_API_KEY",
            "MARATHON_MODEL_API_KEY_FILE": "/tmp/pool.env",
        }
        with mock.patch.dict(router_module.os.environ, environment, clear=True):
            profile = router_module._custom_model_profile(ROOT_DIR)

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertTrue(profile.external)
        self.assertTrue(profile.supports_slots)
        self.assertEqual(profile.api_key_env, "POOL_API_KEY")
        self.assertEqual(profile.api_key_file, "/tmp/pool.env")

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

    def test_catalog_enables_codex_skill_and_plugin_guidance(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.available_profiles = {profile.slug: profile}
        state._refresh_profiles = lambda: state.available_profiles

        with mock.patch.object(router_module, "_base_instructions", return_value="prompt"):
            model = state.model_catalog()["models"][0]

        self.assertTrue(model["include_skills_usage_instructions"])
        self.assertTrue(model["include_plugin_usage_instructions"])

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

    def test_trace_retains_only_bounded_input_fingerprints(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.lock = threading.Lock()
        state._trace_seq = 0
        state._last_trace_by_model = {}
        state.telemetry = mock.Mock()
        state.debug = False
        request = {
            "input": [
                {"role": "user", "content": f"private-{index}"}
                for index in range(5)
            ],
            "instructions": "policy",
            "tools": [],
        }

        with mock.patch.dict(
            router_module.os.environ,
            {"MARATHON_TRACE_INPUT_FINGERPRINTS": "2"},
            clear=False,
        ):
            state.trace_request(
                requested_model=profile.slug,
                profile=profile,
                raw_request=request,
                normalized_request=request,
                path="/v1/responses",
                method="POST",
                raw_body_bytes=123,
                normalized_body_bytes=100,
            )

        retained = state._last_trace_by_model[profile.slug]["input_summary"]
        emitted = state.telemetry.emit.call_args.args[1]
        self.assertEqual(len(retained["fingerprints"]), 2)
        self.assertNotIn("private-", json.dumps(state._last_trace_by_model))
        self.assertEqual(emitted["raw"]["body_bytes"], 123)
        self.assertEqual(emitted["normalized"]["body_bytes"], 100)

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

    def test_local_reasoning_survives_session_resume_for_slot_reuse(self) -> None:
        state = object.__new__(router_module.RouterState)
        backend_item = {
            "id": "reasoning-one",
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [
                {"type": "reasoning_text", "text": "synthetic reasoning"}
            ],
            "encrypted_content": "",
        }

        persisted_item = state.sanitize_output_item(
            backend_item,
            replayable_reasoning=True,
        )
        serialized_item = copy.deepcopy(persisted_item)
        serialized_item.pop("content")
        resumed = router_module.normalize_responses_request(
            {
                "input": [
                    serialized_item,
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "synthetic follow-up"}
                        ],
                    },
                ]
            }
        )

        self.assertEqual(persisted_item["content"][0]["type"], "text")
        self.assertTrue(
            persisted_item["encrypted_content"].startswith(
                router_module.MARATHON_LOCAL_REASONING_CAPSULE_PREFIX
            )
        )
        self.assertNotIn("synthetic reasoning", persisted_item["encrypted_content"])
        self.assertNotIn("status", persisted_item)
        self.assertEqual(resumed["input"][0], persisted_item)

    def test_local_reasoning_capsule_preserves_checkpoint_prefix(self) -> None:
        state = object.__new__(router_module.RouterState)
        persisted_reasoning = state.sanitize_output_item(
            {
                "type": "reasoning",
                "summary": [],
                "content": [
                    {"type": "reasoning_text", "text": "synthetic reasoning"}
                ],
            },
            replayable_reasoning=True,
        )
        saved_request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
                persisted_reasoning,
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ],
        }
        serialized_reasoning = copy.deepcopy(persisted_reasoning)
        serialized_reasoning.pop("content")
        resumed_request = router_module.normalize_responses_request(
            {
                "instructions": "synthetic system prompt",
                "tools": [],
                "input": [
                    saved_request["input"][0],
                    serialized_reasoning,
                    saved_request["input"][2],
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "follow up"}
                        ],
                    },
                ],
            }
        )

        self.assertEqual(
            router_module._conversation_checkpoint_prefix_hash(saved_request, 3),
            router_module._conversation_checkpoint_prefix_hash(resumed_request, 3),
        )

    def test_invalid_local_reasoning_capsule_is_not_replayed(self) -> None:
        normalized = router_module.normalize_responses_request(
            {
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [],
                        "encrypted_content": (
                            router_module.MARATHON_LOCAL_REASONING_CAPSULE_PREFIX
                            + "not-valid-base64"
                        ),
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "synthetic follow-up",
                    },
                ]
            }
        )

        self.assertEqual(len(normalized["input"]), 1)
        self.assertEqual(normalized["input"][0]["role"], "user")

    def test_unmarked_plaintext_reasoning_is_not_replayed(self) -> None:
        historical = {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "text", "text": "historical reasoning"}],
            "encrypted_content": "",
        }

        normalized = router_module.normalize_responses_request(
            {
                "input": [
                    historical,
                    {
                        "type": "message",
                        "role": "user",
                        "content": "synthetic follow-up",
                    },
                ]
            }
        )

        self.assertEqual(len(normalized["input"]), 1)
        self.assertEqual(normalized["input"][0]["type"], "message")

    def test_opaque_reasoning_is_still_dropped_before_llama(self) -> None:
        opaque = {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "opaque-upstream-payload",
        }

        normalized = router_module.normalize_responses_request(
            {
                "input": [
                    opaque,
                    {
                        "type": "message",
                        "role": "user",
                        "content": "synthetic follow-up",
                    },
                ]
            }
        )

        self.assertEqual(len(normalized["input"]), 1)
        self.assertEqual(normalized["input"][0]["type"], "message")

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
            parent_response_id=None,
            input_items=[],
            output_items=[],
            conversation_item_count=0,
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

    def test_empty_non_generating_warmup_is_a_starter_root(self) -> None:
        warmup = router_module.ResponseSnapshot(
            response_id="warm_123",
            profile_slug="dynamic-model",
            parent_response_id=None,
            input_items=[],
            output_items=[],
            conversation_item_count=0,
            snapshot_filename="",
            instructions_text="warmup",
            base_instructions_hash="base",
            instructions_hash="instructions",
            tools_hash="tools",
            prompt_cache_key="session",
            created_at=0,
        )
        generated = replace(warmup, response_id="resp_123")

        self.assertTrue(router_module._is_warmup_root(warmup))
        self.assertFalse(router_module._is_warmup_root(generated))

    def test_lineage_uses_parent_chains_and_rebases_at_the_lru_limit(self) -> None:
        state = object.__new__(router_module.RouterState)
        state.lineage = router_module.OrderedDict()
        state.lineage_max_entries_per_model = 2
        state.last_response_by_model = {}

        def snapshot(
            response_id: str,
            parent_response_id: str | None,
            input_text: str,
            output_text: str,
            count: int,
        ) -> router_module.ResponseSnapshot:
            return router_module.ResponseSnapshot(
                response_id=response_id,
                profile_slug="model",
                parent_response_id=parent_response_id,
                input_items=[{"type": "input", "text": input_text}],
                output_items=[{"type": "output", "text": output_text}],
                conversation_item_count=count,
                snapshot_filename="",
                instructions_text="",
                base_instructions_hash="",
                instructions_hash="",
                tools_hash="",
                prompt_cache_key="",
                created_at=0,
            )

        state._store_response_snapshot(snapshot("one", None, "u1", "a1", 2))
        state._store_response_snapshot(snapshot("two", "one", "u2", "a2", 4))
        state._store_response_snapshot(snapshot("three", "two", "u3", "a3", 6))

        self.assertNotIn("one", state.lineage)
        self.assertIsNone(state.lineage["two"].parent_response_id)
        self.assertEqual(
            [item["text"] for item in state._materialize_conversation(state.lineage["three"])],
            ["u1", "a1", "u2", "a2", "u3", "a3"],
        )

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

    def test_cold_root_restores_rolling_checkpoint_before_starter_cache(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.ensure_model_async = mock.AsyncMock(return_value=profile)
        state.lineage_lock = asyncio.Lock()
        state.lineage = {}
        state.last_response_by_model = {}
        state.live_slot_by_model = {}
        state.live_prompt_cache_key_by_model = {}
        state.experimental_delta_only = False
        state.slot_id = 0
        state.backend_lock = asyncio.Lock()
        state.prepare_conversation_checkpoint = mock.AsyncMock(
            return_value={
                "mode": "restore-conversation-checkpoint",
                "status": "restored",
                "restore_result": {"status": "restored"},
            }
        )
        state.prepare_starter_cache = mock.AsyncMock()
        state._run_responses_loop = mock.AsyncMock(
            return_value=(
                {
                    "id": "resp_new",
                    "usage": {
                        "input_tokens": 100_100,
                        "input_tokens_details": {"cached_tokens": 100_000},
                        "output_tokens": 100,
                        "total_tokens": 100_200,
                    },
                },
                [],
                0,
            )
        )
        state.schedule_conversation_checkpoint = mock.Mock(
            return_value={"status": "scheduled"}
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
                    "prompt_cache_key": "synthetic-session",
                    "instructions": "synthetic system prompt",
                    "tools": [],
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "synthetic input"}
                            ],
                        }
                    ],
                }
            )
        )

        state.prepare_conversation_checkpoint.assert_awaited_once()
        state.prepare_starter_cache.assert_not_awaited()
        self.assertEqual(result["usage"]["input_tokens_details"]["cached_tokens"], 100_000)
        completed = [
            call.args[1]
            for call in state.telemetry.emit.call_args_list
            if call.args[0] == "router.response.completed"
        ][0]
        self.assertEqual(
            completed["slot"]["prepare_mode"],
            "restore-conversation-checkpoint",
        )

    def test_starter_scaffold_matches_responses_tool_conversion(self) -> None:
        scaffold = router_module._starter_scaffold_chat_body(
            {
                "instructions": "stable system prompt",
                "input": [{"role": "user", "content": "not cached"}],
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "description": "Read one file",
                        "parameters": {"type": "object"},
                    },
                    {"type": "web_search"},
                ],
                "parallel_tool_calls": False,
            }
        )

        self.assertEqual(
            scaffold["messages"],
            [
                {"role": "system", "content": "stable system prompt"},
                {
                    "role": "user",
                    "content": router_module.STARTER_CACHE_SENTINEL,
                },
            ],
        )
        self.assertNotEqual(
            scaffold["messages"][-1]["content"],
            "not cached",
        )
        self.assertFalse(scaffold["add_generation_prompt"])
        self.assertEqual(len(scaffold["tools"]), 1)
        self.assertTrue(scaffold["tools"][0]["function"]["strict"])
        self.assertNotIn("input", scaffold)

    def test_starter_cache_survives_router_restart(self) -> None:
        profile = fixture_profile()
        request = {
            "instructions": "stable system prompt",
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "object"},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temporary:
            slot_root = Path(temporary)

            def make_state() -> router_module.RouterState:
                state = object.__new__(router_module.RouterState)
                state.starter_cache_enabled = True
                state.starter_cache_max_count = 8
                state.starter_cache_max_bytes = 1024 * 1024
                state.backend_cache_id = "backend-v1"
                state.slot_save_root = slot_root
                state.slot_id = 0
                state.erase_slot = mock.AsyncMock(return_value={"status": "erased"})
                state.restore_slot = mock.AsyncMock(return_value={"status": "restored"})

                async def qwen_compatible_request(
                    _profile: router_module.ModelProfile,
                    method: str,
                    path: str,
                    payload: dict[str, object],
                    **_kwargs: object,
                ) -> dict[str, object]:
                    self.assertEqual(method, "POST")
                    if path == "/apply-template":
                        messages = payload.get("messages")
                        self.assertIsInstance(messages, list)
                        roles = [message.get("role") for message in messages]
                        if "user" not in roles:
                            raise RuntimeError("No user query found in messages.")
                        user_message = next(
                            message for message in messages if message.get("role") == "user"
                        )
                        return {
                            "prompt": f"stable starter prefix{user_message['content']}"
                        }
                    if path == "/tokenize":
                        content = payload.get("content")
                        self.assertIsInstance(content, str)
                        self.assertTrue(payload.get("add_special"))
                        self.assertTrue(payload.get("parse_special"))
                        suffix = (
                            10
                            if router_module.STARTER_CACHE_SENTINEL in content
                            else 11
                        )
                        return {"tokens": [1, 2, 3, 4, 5, 6, 7, suffix]}
                    if path == "/completion":
                        self.assertEqual(payload.get("prompt"), [1, 2, 3])
                        return {"timings": {"prompt_n": 10_000}}
                    raise AssertionError(f"unexpected backend path: {path}")

                state._request_json = mock.AsyncMock(
                    side_effect=qwen_compatible_request
                )

                async def save_slot(
                    saved_profile: router_module.ModelProfile,
                    filename: str,
                ) -> dict[str, object]:
                    directory = state._slot_save_dir(saved_profile)
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / filename).write_bytes(b"persistent slot state")
                    return {"status": "saved"}

                state.save_slot = mock.AsyncMock(side_effect=save_slot)
                return state

            first = make_state()
            built = asyncio.run(first.prepare_starter_cache(profile, request))

            self.assertEqual(built["mode"], "build-starter-cache")
            self.assertEqual(first._request_json.await_count, 5)
            completion = first._request_json.await_args_list[4].args[3]
            self.assertEqual(completion["n_predict"], 0)
            self.assertEqual(completion["prompt"], [1, 2, 3])

            restarted = make_state()
            restored = asyncio.run(restarted.prepare_starter_cache(profile, request))

            self.assertEqual(restored["mode"], "restore-starter-cache")
            restarted.restore_slot.assert_awaited_once_with(
                profile,
                built["snapshot_filename"],
            )
            restarted._request_json.assert_not_awaited()
            restarted.erase_slot.assert_not_awaited()

    def test_conversation_checkpoint_body_ends_before_next_user(self) -> None:
        body = router_module._conversation_checkpoint_chat_body(
            {
                "instructions": "synthetic system prompt",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "first question"}
                        ],
                    },
                    {
                        "type": "reasoning",
                        "summary": [],
                        "content": [{"type": "text", "text": "brief thought"}],
                    },
                    {
                        "type": "function_call",
                        "name": "read_file",
                        "call_id": "call_1",
                        "arguments": '{"path":"synthetic"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "synthetic result",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "finished"}
                        ],
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        )

        self.assertFalse(body["add_generation_prompt"])
        self.assertEqual(body["messages"][-1], {
            "role": "user",
            "content": router_module.STARTER_CACHE_SENTINEL,
        })
        self.assertEqual(body["messages"][2]["reasoning_content"], "brief thought")
        self.assertEqual(
            body["messages"][2]["tool_calls"][0]["function"]["name"],
            "read_file",
        )
        self.assertEqual(body["messages"][3]["role"], "tool")
        self.assertEqual(body["messages"][4]["content"][0]["text"], "finished")
        self.assertTrue(body["tools"][0]["function"]["strict"])

    def test_compaction_response_defers_obsolete_checkpoint(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.slot_snapshots_enabled = True
        state._closing = False
        state.slot_snapshot_min_tokens = 16_384
        state.slot_snapshot_idle_seconds = 60
        state.backend_cache_id = "backend-v1"
        state.pending_checkpoints = {}
        state.checkpoint_tasks = {}
        state.awaiting_post_compaction_checkpoints = set()
        state.telemetry = mock.Mock()
        state.slot_checkpoint_store = mock.Mock()
        stale_checkpoint = SimpleNamespace(size_bytes=550_000_000)
        state.slot_checkpoint_store.find_any.return_value = stale_checkpoint
        request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "prompt_cache_key": "synthetic-session",
            "client_metadata": {
                "x-codex-turn-metadata": json.dumps(
                    {
                        "request_kind": "compaction",
                        "window_id": "synthetic-window",
                    }
                )
            },
        }

        async def scenario() -> dict[str, object]:
            result = state.schedule_conversation_checkpoint(
                profile,
                request,
                "compaction-response",
                {"usage": {"total_tokens": 200_000}},
                [],
            )
            tasks = list(state.checkpoint_tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "compaction state is not canonical")
        self.assertEqual(state.pending_checkpoints, {})
        self.assertEqual(state.checkpoint_tasks, {})
        self.assertIn(
            (profile.slug, "synthetic-session"),
            state.awaiting_post_compaction_checkpoints,
        )
        state.slot_checkpoint_store.discard.assert_called_once_with(stale_checkpoint)

    def test_first_turn_after_compaction_gets_priority_checkpoint(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.slot_snapshots_enabled = True
        state._closing = False
        state.slot_snapshot_min_tokens = 16_384
        state.slot_snapshot_idle_seconds = 0
        state.backend_cache_id = "backend-v1"
        state.pending_checkpoints = {}
        state.checkpoint_tasks = {}
        state.awaiting_post_compaction_checkpoints = {
            (profile.slug, "synthetic-session")
        }
        state.telemetry = mock.Mock()
        state._save_conversation_checkpoint = mock.AsyncMock(
            return_value={"status": "saved", "context_tokens": 8_000}
        )
        request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "prompt_cache_key": "synthetic-session",
            "client_metadata": {
                "x-codex-turn-metadata": json.dumps(
                    {
                        "request_kind": "turn",
                        "window_id": "synthetic-window:1",
                    }
                )
            },
        }

        async def scenario() -> dict[str, object]:
            result = state.schedule_conversation_checkpoint(
                profile,
                request,
                "post-compaction-response",
                {"usage": {"total_tokens": 8_000}},
                [],
            )
            task = state.checkpoint_tasks[profile.slug]
            await task
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "scheduled")
        self.assertEqual(result["idle_seconds"], 0)
        self.assertEqual(result["priority"], "post-compaction")
        candidate = state._save_conversation_checkpoint.await_args.args[0]
        self.assertTrue(candidate.post_compaction)
        self.assertTrue(
            state._save_conversation_checkpoint.await_args.kwargs["force"]
        )
        self.assertEqual(state.pending_checkpoints, {})
        self.assertEqual(state.awaiting_post_compaction_checkpoints, set())

    def test_rolling_checkpoint_skips_small_growth_at_large_context(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.available_profiles = {profile.slug: profile}
        state.backend_cache_id = "backend-v1"
        state.slot_snapshot_min_token_growth = (
            router_module.DEFAULT_SLOT_SNAPSHOT_MIN_TOKEN_GROWTH
        )
        state.backend_lock = asyncio.Lock()
        state.live_slot_by_model = {profile.slug: "response-two"}
        state.live_prompt_cache_key_by_model = {
            profile.slug: "synthetic-session"
        }
        metadata = SimpleNamespace(
            response_id_hash="different-response",
            profile_slug=profile.slug,
            backend_cache_id="backend-v1",
            scaffold_fingerprint="synthetic-scaffold",
            context_tokens=200_000,
        )
        state.slot_checkpoint_store = mock.Mock()
        state.slot_checkpoint_store.find_any.return_value = SimpleNamespace(
            metadata=metadata
        )
        state.slot_checkpoint_store.response_id_hash.return_value = (
            "new-response"
        )
        state.slot_checkpoint_store.pending_filename.return_value = (
            "pending-checkpoint.bin"
        )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state.slot_checkpoint_store.profile_dir.return_value = Path(temporary.name)
        state.slot_checkpoint_store.commit.return_value = {"status": "saved"}
        state.save_slot = mock.AsyncMock(return_value={"status": "saved"})
        candidate = router_module.ConversationCheckpointCandidate(
            profile_slug=profile.slug,
            profile_alias=profile.alias,
            prompt_cache_key="synthetic-session",
            response_id="response-two",
            scaffold_fingerprint="synthetic-scaffold",
            context_tokens=216_000,
        )

        result = asyncio.run(
            state._save_conversation_checkpoint(candidate, force=False)
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            result["reason"],
            "conversation has not grown enough since the last checkpoint",
        )
        self.assertEqual(result["required_token_growth"], 20_000)
        state.save_slot.assert_not_awaited()

    def test_checkpoint_schedule_reuses_completed_conversation_input(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.slot_snapshots_enabled = True
        state._closing = False
        state.slot_snapshot_min_tokens = 16_384
        state.slot_snapshot_idle_seconds = 60
        state.backend_cache_id = "backend-v1"
        state.pending_checkpoints = {}
        state.checkpoint_tasks = {}
        state.awaiting_post_compaction_checkpoints = set()
        conversation_input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "synthetic"}],
            }
        ]
        request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "prompt_cache_key": "synthetic-session",
        }

        async def scenario() -> router_module.ConversationCheckpointCandidate:
            state.schedule_conversation_checkpoint(
                profile,
                request,
                "response-one",
                {"usage": {"total_tokens": 20_000}},
                conversation_input,
            )
            candidate = state.pending_checkpoints[profile.slug]
            task = state.checkpoint_tasks[profile.slug]
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return candidate

        candidate = asyncio.run(scenario())

        self.assertIsNotNone(candidate.checkpoint_request)
        self.assertIs(candidate.checkpoint_request["input"], conversation_input)

    def test_rolling_conversation_checkpoint_survives_router_restart(self) -> None:
        profile = fixture_profile()
        request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "prompt_cache_key": "synthetic-session",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temporary:
            slot_root = Path(temporary)

            def make_state() -> router_module.RouterState:
                state = object.__new__(router_module.RouterState)
                state.available_profiles = {profile.slug: profile}
                state.backend_cache_id = "backend-v1"
                state.slot_save_root = slot_root
                state.slot_snapshots_enabled = True
                state.slot_snapshot_min_token_growth = 4_096
                state.slot_checkpoint_store = router_module.RollingCheckpointStore(
                    slot_root,
                    slot_root,
                    max_count=2,
                    max_bytes=32 * 1024**3,
                    ttl_seconds=172_800,
                )
                state.backend_lock = asyncio.Lock()
                state.live_slot_by_model = {}
                state.live_prompt_cache_key_by_model = {}
                state.telemetry = mock.Mock()
                state.restore_slot = mock.AsyncMock(return_value={"status": "restored"})
                state._prefill_conversation_checkpoint_boundary = mock.AsyncMock(
                    side_effect=lambda _profile, checkpoint_request: {
                        "context_tokens": 100_000,
                        "conversation_item_count": len(checkpoint_request["input"]),
                        "conversation_prefix_hash": (
                            router_module._conversation_checkpoint_prefix_hash(
                                checkpoint_request,
                                len(checkpoint_request["input"]),
                            )
                        ),
                    }
                )

                async def save_slot(
                    saved_profile: router_module.ModelProfile,
                    filename: str,
                ) -> dict[str, object]:
                    directory = state._slot_save_dir(saved_profile)
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / filename).write_bytes(b"synthetic slot checkpoint")
                    return {"status": "saved"}

                state.save_slot = mock.AsyncMock(side_effect=save_slot)
                return state

            first = make_state()
            candidate = router_module.ConversationCheckpointCandidate(
                profile_slug=profile.slug,
                profile_alias=profile.alias,
                prompt_cache_key="synthetic-session",
                response_id="response-one",
                scaffold_fingerprint=first._conversation_scaffold_fingerprint(
                    profile,
                    request,
                ),
                context_tokens=100_000,
                checkpoint_request=request,
            )
            first.live_slot_by_model[profile.slug] = candidate.response_id
            first.live_prompt_cache_key_by_model[profile.slug] = (
                candidate.prompt_cache_key
            )

            saved = asyncio.run(
                first._save_conversation_checkpoint(candidate, force=False)
            )
            self.assertEqual(saved["status"], "saved")

            restarted = make_state()
            resumed_request = dict(request)
            resumed_request["input"] = list(request["input"]) + [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "follow up"}],
                }
            ]
            restored = asyncio.run(
                restarted.prepare_conversation_checkpoint(
                    profile,
                    resumed_request,
                    "synthetic-session",
                )
            )

            self.assertEqual(restored["mode"], "restore-conversation-checkpoint")
            self.assertEqual(restored["context_tokens"], 100_000)
            restarted.restore_slot.assert_awaited_once_with(
                profile,
                restored["snapshot_filename"],
            )

    def test_rolling_checkpoint_rejects_resumed_rollback_before_restore(self) -> None:
        profile = fixture_profile()
        saved_input = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "question"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
            {
                "type": "function_call",
                "name": "synthetic_tool",
                "call_id": "call-one",
                "arguments": "{}",
            },
        ]
        saved_request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "input": saved_input,
        }
        metadata = SimpleNamespace(
            conversation_item_count=len(saved_input),
            conversation_prefix_hash=(
                router_module._conversation_checkpoint_prefix_hash(
                    saved_request,
                    len(saved_input),
                )
            ),
            context_tokens=100_000,
        )
        record = SimpleNamespace(metadata=metadata)
        state = object.__new__(router_module.RouterState)
        state.slot_snapshots_enabled = True
        state.backend_cache_id = "backend-v1"
        state.awaiting_post_compaction_checkpoints = set()
        state.slot_checkpoint_store = mock.Mock()
        state.slot_checkpoint_store.find.return_value = record
        state.restore_slot = mock.AsyncMock(return_value={"status": "restored"})
        resumed_request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "input": saved_input[:2],
        }

        result = asyncio.run(
            state.prepare_conversation_checkpoint(
                profile,
                resumed_request,
                "synthetic-session",
            )
        )

        self.assertEqual(result["mode"], "conversation-checkpoint-miss")
        self.assertEqual(
            result["reason"],
            "checkpoint is not a prefix of the resumed conversation",
        )
        self.assertEqual(result["checkpoint_items"], 3)
        self.assertEqual(result["current_items"], 2)
        state.slot_checkpoint_store.discard.assert_called_once_with(record)
        state.restore_slot.assert_not_awaited()

    def test_rolling_checkpoint_ignores_non_prompt_item_metadata(self) -> None:
        saved_request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "input": [
                {
                    "type": "message",
                    "id": "backend-message-id",
                    "role": "assistant",
                    "status": "completed",
                    "phase": "final_answer",
                    "internal_chat_message_metadata_passthrough": {"opaque": True},
                    "content": [{"type": "output_text", "text": "answer"}],
                }
            ],
        }
        resumed_request = copy.deepcopy(saved_request)
        resumed_request["input"][0].pop("status")
        resumed_request["input"][0]["id"] = "reserialized-message-id"
        resumed_request["input"][0]["phase"] = "commentary"
        resumed_request["input"][0][
            "internal_chat_message_metadata_passthrough"
        ] = {"opaque": False}

        self.assertNotEqual(
            router_module._input_fingerprint_summary(saved_request["input"], 0)[
                "hash"
            ],
            router_module._input_fingerprint_summary(resumed_request["input"], 0)[
                "hash"
            ],
        )
        self.assertEqual(
            router_module._conversation_checkpoint_prefix_hash(saved_request, 1),
            router_module._conversation_checkpoint_prefix_hash(resumed_request, 1),
        )

    def test_rolling_checkpoint_detects_model_visible_history_change(self) -> None:
        saved_request = {
            "instructions": "synthetic system prompt",
            "tools": [],
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                }
            ],
        }
        changed_request = copy.deepcopy(saved_request)
        changed_request["input"][0]["content"][0]["text"] = "different answer"

        self.assertNotEqual(
            router_module._conversation_checkpoint_prefix_hash(saved_request, 1),
            router_module._conversation_checkpoint_prefix_hash(changed_request, 1),
        )

    def test_rolling_checkpoint_saves_a_stable_prefilled_boundary(self) -> None:
        profile = fixture_profile()
        with tempfile.TemporaryDirectory() as temporary:
            slot_root = Path(temporary)
            state = object.__new__(router_module.RouterState)
            state.available_profiles = {profile.slug: profile}
            state.backend_cache_id = "backend-v1"
            state.slot_save_root = slot_root
            state.slot_id = 0
            state.slot_snapshot_min_token_growth = 4_096
            state.slot_checkpoint_store = router_module.RollingCheckpointStore(
                slot_root,
                slot_root,
                max_count=2,
                max_bytes=32 * 1024**3,
                ttl_seconds=172_800,
            )
            state.backend_lock = asyncio.Lock()
            state.live_slot_by_model = {profile.slug: "response-one"}
            state.live_prompt_cache_key_by_model = {
                profile.slug: "synthetic-session"
            }

            async def backend_request(
                _profile: router_module.ModelProfile,
                method: str,
                path: str,
                payload: dict[str, object],
                **_kwargs: object,
            ) -> dict[str, object]:
                self.assertEqual(method, "POST")
                if path == "/apply-template":
                    messages = payload["messages"]
                    self.assertIsInstance(messages, list)
                    return {"prompt": str(messages[-1]["content"])}
                if path == "/tokenize":
                    suffix = (
                        10
                        if router_module.STARTER_CACHE_SENTINEL
                        in str(payload["content"])
                        else 11
                    )
                    return {"tokens": [1, 2, 3, 4, 5, 6, 7, 8, suffix]}
                if path == "/completion":
                    self.assertEqual(payload["prompt"], [1, 2, 3, 4])
                    self.assertEqual(payload["n_predict"], 0)
                    self.assertTrue(payload["cache_prompt"])
                    return {"timings": {"prompt_n": 4}}
                raise AssertionError(f"unexpected backend path: {path}")

            state._request_json = mock.AsyncMock(side_effect=backend_request)

            async def save_slot(
                saved_profile: router_module.ModelProfile,
                filename: str,
            ) -> dict[str, object]:
                directory = state._slot_save_dir(saved_profile)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / filename).write_bytes(b"synthetic slot checkpoint")
                return {"status": "saved"}

            state.save_slot = mock.AsyncMock(side_effect=save_slot)
            checkpoint_request = {
                "instructions": "synthetic system prompt",
                "tools": [],
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "question"}
                        ],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "answer"}
                        ],
                    },
                ],
            }
            candidate = router_module.ConversationCheckpointCandidate(
                profile_slug=profile.slug,
                profile_alias=profile.alias,
                prompt_cache_key="synthetic-session",
                response_id="response-one",
                scaffold_fingerprint=state._conversation_scaffold_fingerprint(
                    profile,
                    checkpoint_request,
                ),
                context_tokens=20_000,
                checkpoint_request=checkpoint_request,
            )

            result = asyncio.run(
                state._save_conversation_checkpoint(candidate, force=False)
            )

            self.assertEqual(result["status"], "saved")
            self.assertEqual(result["context_tokens"], 4)
            self.assertEqual(result["boundary"]["context_tokens"], 4)
            self.assertEqual(state._request_json.await_count, 5)
            state.save_slot.assert_awaited_once()

    def test_rolling_checkpoint_rejects_changed_scaffold(self) -> None:
        profile = fixture_profile()
        with tempfile.TemporaryDirectory() as temporary:
            slot_root = Path(temporary)
            state = object.__new__(router_module.RouterState)
            state.available_profiles = {profile.slug: profile}
            state.backend_cache_id = "backend-v1"
            state.slot_save_root = slot_root
            state.slot_snapshots_enabled = True
            state.slot_snapshot_min_token_growth = 4_096
            state.slot_checkpoint_store = router_module.RollingCheckpointStore(
                slot_root,
                slot_root,
                max_count=2,
                max_bytes=32 * 1024**3,
                ttl_seconds=172_800,
            )
            state.backend_lock = asyncio.Lock()
            state.live_slot_by_model = {profile.slug: "response-one"}
            state.live_prompt_cache_key_by_model = {
                profile.slug: "synthetic-session"
            }
            state.telemetry = mock.Mock()
            state._prefill_conversation_checkpoint_boundary = mock.AsyncMock(
                side_effect=lambda _profile, checkpoint_request: {
                    "context_tokens": 100_000,
                    "conversation_item_count": len(checkpoint_request["input"]),
                    "conversation_prefix_hash": (
                        router_module._conversation_checkpoint_prefix_hash(
                            checkpoint_request,
                            len(checkpoint_request["input"]),
                        )
                    ),
                }
            )

            async def save_slot(
                saved_profile: router_module.ModelProfile,
                filename: str,
            ) -> dict[str, object]:
                directory = state._slot_save_dir(saved_profile)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / filename).write_bytes(b"synthetic slot checkpoint")
                return {"status": "saved"}

            state.save_slot = mock.AsyncMock(side_effect=save_slot)
            state.restore_slot = mock.AsyncMock(return_value={"status": "restored"})
            conversation_input = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ]
            original = {
                "instructions": "one",
                "tools": [],
                "input": conversation_input,
            }
            candidate = router_module.ConversationCheckpointCandidate(
                profile_slug=profile.slug,
                profile_alias=profile.alias,
                prompt_cache_key="synthetic-session",
                response_id="response-one",
                scaffold_fingerprint=state._conversation_scaffold_fingerprint(
                    profile,
                    original,
                ),
                context_tokens=100_000,
                checkpoint_request=original,
            )
            asyncio.run(state._save_conversation_checkpoint(candidate, force=False))

            result = asyncio.run(
                state.prepare_conversation_checkpoint(
                    profile,
                    {
                        "instructions": "two",
                        "tools": [],
                        "input": conversation_input,
                    },
                    "synthetic-session",
                )
            )

            self.assertEqual(result["mode"], "conversation-checkpoint-miss")
            state.restore_slot.assert_not_awaited()

    def test_starter_cache_fingerprint_tracks_scaffold_and_backend(self) -> None:
        profile = fixture_profile()
        first = router_module._starter_cache_fingerprint(
            profile,
            "backend-v1",
            router_module._starter_scaffold_chat_body(
                {"instructions": "one", "tools": []}
            ),
        )
        changed_prompt = router_module._starter_cache_fingerprint(
            profile,
            "backend-v1",
            router_module._starter_scaffold_chat_body(
                {"instructions": "two", "tools": []}
            ),
        )
        changed_backend = router_module._starter_cache_fingerprint(
            profile,
            "backend-v2",
            router_module._starter_scaffold_chat_body(
                {"instructions": "one", "tools": []}
            ),
        )

        self.assertNotEqual(first, changed_prompt)
        self.assertNotEqual(first, changed_backend)

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
            def post(
                self,
                _url: str,
                *,
                json: dict[str, object],
                headers: dict[str, str] | None = None,
            ):
                self.request = json
                self.headers = headers
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
        self.assertEqual(state.http_client.headers, {})

    def test_streamed_reasoning_uses_resume_safe_content_type(self) -> None:
        profile = fixture_profile(65_536)
        reasoning = {
            "id": "reasoning-one",
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [
                {"type": "reasoning_text", "text": "synthetic reasoning"}
            ],
            "encrypted_content": "",
        }

        class Content:
            async def iter_chunked(self, _size: int):
                events = [
                    {"type": "response.output_item.done", "item": reasoning},
                    {
                        "type": "response.completed",
                        "response": {"id": "resp_reasoning", "usage": {}},
                    },
                ]
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n".encode()

        class Response:
            status = 200
            content = Content()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class Client:
            def post(self, *_args, **_kwargs):
                return Response()

        state = object.__new__(router_module.RouterState)
        state.http_client = Client()
        streamed: list[dict[str, object]] = []

        async def sink(event: dict[str, object]) -> bool:
            streamed.append(event)
            return True

        response = asyncio.run(
            state._request_responses_stream(
                profile,
                {"input": [], "tools": []},
                event_sink=sink,
            )
        )

        done = next(
            event for event in streamed if event["type"] == "response.output_item.done"
        )
        self.assertEqual(done["item"]["content"][0]["type"], "text")
        self.assertTrue(
            done["item"]["encrypted_content"].startswith(
                router_module.MARATHON_LOCAL_REASONING_CAPSULE_PREFIX
            )
        )
        self.assertEqual(response["output"][0]["content"][0]["type"], "text")

    def test_http_fallback_accepts_context_sized_request_bodies(self) -> None:
        state = object.__new__(router_module.RouterState)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            app = router_module.build_app(state)

        self.assertEqual(app._client_max_size, router_module.ROUTER_MAX_REQUEST_BYTES)
        self.assertGreater(app._client_max_size, 1_110_965)

    def test_http_fallback_recognizes_compaction_metadata(self) -> None:
        profile = fixture_profile()
        state = object.__new__(router_module.RouterState)
        state.debug = False
        state.telemetry = mock.Mock()
        state.trace_request = mock.Mock()
        state.ensure_model_async = mock.AsyncMock(return_value=profile)
        state.defer_compaction_checkpoint = mock.Mock(
            return_value={"status": "skipped"}
        )
        state.http_client = None

        class Request:
            app = {"state": state}
            path = "/v1/responses"
            method = "POST"
            headers = {
                "x-codex-turn-metadata": json.dumps(
                    {
                        "request_kind": "compaction",
                        "window_id": "synthetic-window",
                    }
                )
            }

            async def read(self) -> bytes:
                return json.dumps(
                    {
                        "model": profile.slug,
                        "prompt_cache_key": "synthetic-session",
                        "input": [],
                        "tools": [],
                    }
                ).encode()

        response = asyncio.run(router_module.handle_http_proxy(Request()))

        self.assertEqual(response.status, 500)
        state.defer_compaction_checkpoint.assert_called_once_with(
            profile,
            "synthetic-session",
            context_tokens=0,
            transport="http",
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
