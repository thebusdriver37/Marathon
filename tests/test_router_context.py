from __future__ import annotations

import asyncio
import sys
import unittest
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
        self.assertEqual(model["auto_compact_token_limit"], 235_929)
        self.assertEqual(model["effective_context_window_percent"], 100)

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


if __name__ == "__main__":
    unittest.main()
