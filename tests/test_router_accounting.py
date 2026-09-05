import asyncio
import json
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from test_router_context import fixture_profile, router_module


class ResponseAccountingTests(unittest.TestCase):
    def test_partial_timing_coverage_is_explicit(self):
        accounting = router_module.ResponseAccounting()
        measured = {
            "usage": {"input_tokens": 20, "output_tokens": 11},
            "timings": {
                "predicted_n": 11,
                "predicted_ms": 200,
                "prompt_n": 20,
                "prompt_ms": 100,
            },
        }
        accounting.add(measured)
        accounting.add({"usage": {"input_tokens": 30, "output_tokens": 10}})
        result = {}
        accounting.apply(result)
        metrics = result["usage_metadata"]["marathon"]
        self.assertEqual(
            (metrics["backend_calls"], metrics["timed_backend_calls"]), (2, 1)
        )
        self.assertEqual(result["usage"]["total_tokens"], 71)
        self.assertEqual(metrics["context_usage"]["total_tokens"], 40)

    def test_invalid_timing_is_not_a_zero_duration_measurement(self):
        for duration in [None, -1, float("nan"), float("inf"), True, "100", 0]:
            with self.subTest(duration=duration):
                accounting = router_module.ResponseAccounting()
                accounting.add(
                    {
                        "timings": {
                            "predicted_n": 10,
                            "predicted_ms": duration,
                            "prompt_ms": 100,
                        }
                    }
                )
                self.assertEqual(accounting.timed_backend_calls, 0)


class RouterAccountingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_tool_loop_counts_all_calls_and_preserves_context(self):
        calls = []
        usage = [
            {
                "input_tokens": 10_000,
                "output_tokens": 50,
                "total_tokens": 10_050,
                "input_tokens_details": {"cached_tokens": 9_000},
            },
            {
                "input_tokens": 12_000,
                "output_tokens": 75,
                "total_tokens": 12_075,
                "input_tokens_details": {"cached_tokens": 10_000},
            },
        ]

        async def backend(request):
            calls.append(await request.json())
            index = len(calls) - 1
            item = (
                {
                    "id": "fc_search",
                    "type": "function_call",
                    "name": "web_search",
                    "call_id": "search_1",
                    "arguments": '{"query":"test"}',
                }
                if index == 0
                else {
                    "id": "msg_answer",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Finished."}],
                }
            )
            events = [
                {"type": "response.output_item.done", "item": item},
                {
                    "type": "response.completed",
                    "response": {
                        "id": f"backend_{index}",
                        "output": [item],
                        "usage": usage[index],
                    },
                    # llama.cpp puts timings beside response, not inside it.
                    "timings": {
                        "prompt_n": 1000 + index * 1000,
                        "prompt_ms": 500.0,
                        "predicted_n": 50 + index * 25,
                        "predicted_ms": 1000.0,
                        "draft_n": 100,
                        "draft_n_accepted": 25,
                    },
                },
            ]
            return web.Response(
                text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
                content_type="text/event-stream",
            )

        upstream = web.Application()
        upstream.router.add_post("/v1/responses", backend)
        async with TestServer(upstream) as backend_server, ClientSession() as http:
            profile = replace(
                fixture_profile(),
                target=str(backend_server.make_url("")),
                supports_slots=False,
            )
            state = object.__new__(router_module.RouterState)
            state.http_client = http
            state.ensure_model_async = mock.AsyncMock(return_value=profile)
            state.lineage_lock = asyncio.Lock()
            state.backend_lock = asyncio.Lock()
            state.lineage = {}
            state.last_response_by_model = {}
            state.live_slot_by_model = {}
            state.live_prompt_cache_key_by_model = {}
            state.experimental_delta_only = False
            state.slot_id = 0
            state.active_ws_tasks = {}
            state.web_search_settings = SimpleNamespace(max_iterations=3)
            state.web_search = mock.Mock()
            state._execute_managed_call = mock.AsyncMock(
                return_value={
                    "type": "function_call_output",
                    "call_id": "search_1",
                    "output": "result",
                }
            )
            state.schedule_conversation_checkpoint = mock.Mock(
                return_value={"status": "skipped"}
            )
            state.telemetry = mock.Mock()
            state.trace_request = mock.Mock()
            state.lock = threading.Lock()
            state._trace_seq = 0
            state._response_id_seq = 0
            state.debug = False
            app = web.Application()
            app["state"] = state
            app.router.add_get("/v1/responses", router_module.handle_ws_responses)
            async with TestServer(app) as server:
                async with http.ws_connect(server.make_url("/v1/responses")) as ws:
                    await ws.send_json(
                        {
                            "type": "response.create",
                            "model": profile.slug,
                            "input": [{"role": "user", "content": "research test"}],
                            "tools": [{"type": "web_search"}],
                        }
                    )
                    while True:
                        event = await asyncio.wait_for(ws.receive_json(), timeout=5)
                        self.assertNotEqual(event["type"], "response.failed", event)
                        if event["type"] == "response.completed":
                            response = event["response"]
                            break

        self.assertEqual(len(calls), 2)
        self.assertEqual(response["usage"]["output_tokens"], 125)
        self.assertEqual(response["usage"]["input_tokens"], 22_000)
        self.assertEqual(response["usage"]["total_tokens"], 22_125)
        self.assertEqual(
            response["usage"]["input_tokens_details"]["cached_tokens"], 19_000
        )
        metrics = response["usage_metadata"]["marathon"]
        self.assertEqual(metrics["context_usage"]["total_tokens"], 12_075)
        self.assertEqual(metrics["decode_tokens"], 123)
        self.assertEqual(metrics["decode_microseconds"], 2_000_000)
        self.assertEqual(metrics["prefill_tokens"], 3000)
        self.assertEqual(metrics["prefill_microseconds"], 1_000_000)
        self.assertEqual(metrics["backend_calls"], 2)
        self.assertEqual(metrics["timed_backend_calls"], 2)
        self.assertEqual(
            router_module.RouterState._response_context_tokens(response), 12_075
        )
