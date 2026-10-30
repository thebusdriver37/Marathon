"""Transport-level regressions using real sockets and a scripted upstream.

No model, GPU, external network, or user workspace is needed.
Each defect has two independent scenarios, each run with fresh servers twice.
"""

import asyncio
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from test_router_context import fixture_profile, router_module as router


def call(name, key, arguments):
    return {"type": "function_call", "id": f"fc_{key}", "call_id": key,
            "name": name, "arguments": json.dumps(arguments)}


def answer():
    return {"type": "message", "role": "assistant", "id": "answer",
            "content": [{"type": "output_text", "text": "Verified answer."}]}


def patch_call(key):
    return call("apply_patch", key, {"operations": [
        {"action": "add", "path": "example.txt", "content": "hello"}]})


def events_from_bytes(wire):
    events = []
    while wire:
        frame, wire = router._pop_sse_frame(wire)
        if frame is None:
            raise AssertionError("unterminated SSE frame")
        event = router._parse_sse_frame(frame)
        if event is not None and event.data != "[DONE]":
            events.append(json.loads(event.data))
    return events


@asynccontextmanager
async def serving(backend, *, alias=None):
    upstream = web.Application()
    upstream.router.add_post("/v1/responses", backend)
    upstream.router.add_post("/v1/chat/completions", backend)
    async def models(_request):
        return web.json_response({"data": [{"id": alias or fixture_profile().alias}]})
    upstream.router.add_get("/v1/models", models)
    with tempfile.TemporaryDirectory() as temporary:
        async with TestServer(upstream) as upstream_server, ClientSession() as http:
            profile = replace(fixture_profile(), supports_slots=False,
                              alias=alias or fixture_profile().alias, external=bool(alias),
                              target=str(upstream_server.make_url("")))
            with mock.patch.object(router, "_available_profiles", return_value={profile.slug: profile}), \
                    mock.patch.dict(os.environ, {"MARATHON_LAZY_POOL_BACKEND": "",
                                                "MARATHON_SLOT_SNAPSHOTS_ENABLED": "0"}):
                root = Path(temporary)
                state = router.RouterState(profile.slug, root / "state", root / "logs")
                state.http_client = http
                state.telemetry = mock.Mock()
                state.web_search_settings = SimpleNamespace(max_iterations=2)
                state.web_search = mock.Mock()
                state._execute_managed_call = mock.AsyncMock(side_effect=lambda item, index: {
                    "type": "function_call_output", "call_id": item["call_id"],
                    "output": "Sufficient verified evidence."})
                app = web.Application()
                app["state"] = state
                app.router.add_get("/v1/responses", router.handle_ws_responses)
                app.router.add_post("/v1/responses", router.handle_http_proxy)
                app.router.add_post("/v1/chat/completions", router.handle_http_proxy)
                async with TestServer(app) as server:
                    yield state, http, server


def scripted_backend(select_items, requests):
    async def backend(request):
        data = await request.json()
        requests.append(data)
        items = select_items(data, len(requests))
        result = {"id": f"backend_{len(requests)}", "output": items,
                  "usage": {"output_tokens": 10}}
        if not data.get("stream"):
            return web.json_response(result)
        events = []
        for item in items:
            events.extend({"type": f"response.output_item.{kind}", "item": item}
                          for kind in ("added", "done"))
        events.append({"type": "response.completed", "response": result})
        return web.Response(text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
                            content_type="text/event-stream")
    return backend


async def websocket_turn(http, server, tools):
    async with http.ws_connect(server.make_url("/v1/responses")) as ws:
        await ws.send_json({"type": "response.create", "input": [
            {"type": "message", "role": "user", "content": "Verify this task."}],
            "tools": tools})
        events = []
        while True:
            event = await asyncio.wait_for(ws.receive_json(), 5)
            events.append(event)
            if event["type"] in {"response.completed", "response.failed"}:
                return events


class RouterTransportRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_external_alias_and_catalog_id_share_response_lineage(self):
        alias = "spark/Qwen3.8-Flash-Next-LR7"
        for transport in ("http", "websocket"):
            with self.subTest(transport=transport):
                requests = []
                backend = scripted_backend(lambda data, count: [answer()], requests)
                async with serving(backend, alias=alias) as (state, http, server):
                    previous = None
                    for model in (alias, alias, fixture_profile().slug):
                        payload = {"model": model, "input": [{"role": "user", "content": "Reply OK."}]}
                        if previous:
                            payload["previous_response_id"] = previous
                        if transport == "websocket":
                            async with http.ws_connect(server.make_url('/v1/responses')) as ws:
                                await ws.send_json(dict(payload, type="response.create"))
                                while True:
                                    event = await asyncio.wait_for(ws.receive_json(), 5)
                                    if event["type"] in {"response.completed", "response.failed"}:
                                        break
                                self.assertEqual(event["type"], "response.completed", event)
                                body = event["response"]
                        else:
                            async with http.post(server.make_url('/v1/responses'), json=payload) as response:
                                body = await response.json()
                                self.assertEqual(response.status, 200, body)
                        previous = body["id"]
                        self.assertEqual(state.lineage[previous].profile_slug, fixture_profile().slug)
                    self.assertEqual([r['model'] for r in requests], [alias] * 3)
                    other = replace(state.available_profiles[fixture_profile().slug], slug="another-model")
                    state.available_profiles[other.slug] = other
                    async with http.post(server.make_url('/v1/responses'), json={
                        "model": other.slug, "previous_response_id": previous, "input": []
                    }) as response:
                        body = await response.json()
                        self.assertIn("belongs to model", json.dumps(body))
                    self.assertEqual(len(requests), 3)

    async def test_image_limit_recovers_history_for_http_and_websocket(self):
        def image_item(index):
            return {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": f"Screenshot {index}"},
                {"type": "input_image", "image_url": f"https://example.invalid/{index}.png"}]}

        for transport in ("http", "websocket"):
            with self.subTest(transport=transport):
                counts, successful = [], []
                normal = scripted_backend(lambda data, count: [answer()], successful)

                async def backend(request):
                    data = await request.json()
                    count = sum(part.get("type") == "input_image"
                                for item in data["input"]
                                for part in item.get("content", []) if isinstance(part, dict))
                    counts.append(count)
                    if count > 4:
                        return web.json_response({"error": {"error": {
                            "message": "a request supports at most 4 images",
                            "type": "invalid_request_error"}}}, status=400)
                    return await normal(request)

                async with serving(backend) as (state, http, server):
                    previous = None
                    for count in (4, 5, 6):
                        payload = {"input": [image_item(i) for i in range(count)], "tools": []}
                        if transport == "websocket":
                            if previous:
                                payload["input"] = [image_item(count - 1)]
                                payload["previous_response_id"] = previous
                            async with http.ws_connect(server.make_url('/v1/responses')) as ws:
                                await ws.send_json(dict(payload, type="response.create"))
                                while True:
                                    event = await asyncio.wait_for(ws.receive_json(), 5)
                                    if event["type"] in {"response.completed", "response.failed"}:
                                        break
                                self.assertEqual(event["type"], "response.completed", event)
                                previous = event["response"]["id"]
                        else:
                            async with http.post(server.make_url('/v1/responses'), json=payload) as response:
                                body = await response.json()
                                self.assertEqual(response.status, 200, body)
                    self.assertEqual(counts, [4, 5, 4, 4])
                    retained = [part["image_url"] for item in successful[-1]["input"]
                                for part in item.get("content", [])
                                if isinstance(part, dict) and part.get("type") == "input_image"]
                    self.assertEqual(retained, [f"https://example.invalid/{i}.png" for i in range(2, 6)])
                    self.assertIn("Earlier image omitted", json.dumps(successful[-1]))

    @unittest.skipUnless(os.environ.get("MARATHON_ROUTER_TEST_BIN"),
                         "Set MARATHON_ROUTER_TEST_BIN for installed CLI integration")
    async def test_installed_cli_http_research_and_edit(self):
        for filename in ("example.txt", "second.txt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                requests = []
                def select(data, n):
                    if n == 1:
                        return [call("web_fetch", "web", {"url": "https://example.org"})]
                    if n == 2:
                        return [call("apply_patch", "patch", {"operations": [
                            {"action": "add", "path": filename, "content": "hello"}]})]
                    return [answer()]
                async with serving(scripted_backend(select, requests)) as (state, http, server):
                    catalog = Path(temporary) / "models.json"
                    catalog.write_text(json.dumps(state.model_catalog()))
                    provider = ('model_providers.marathon-local={name="Test",wire_api="responses",'
                                f'base_url="{server.make_url("/v1")}",supports_websockets=false}}')
                    process = await asyncio.create_subprocess_exec(
                        os.environ["MARATHON_ROUTER_TEST_BIN"],
                        "-c", 'model_provider="marathon-local"', "-c", provider,
                        "-c", f'model_catalog_json="{catalog}"',
                        "-c", 'web_search="live"', "-m", fixture_profile().slug,
                        "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "--json",
                        f"Research the example documentation, then create {filename} containing hello.",
                        cwd=temporary,
                        env=dict(os.environ, CODEX_HOME=temporary, CODEX_SQLITE_HOME=temporary,
                                 MARATHON_LOCAL_ONLY="1"),
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
                    finally:
                        if process.returncode is None:
                            process.kill()
                            await process.wait()
                self.assertEqual(process.returncode, 0, (stdout + stderr).decode())
                self.assertTrue((Path(temporary) / filename).exists(), (stdout + stderr).decode())
                self.assertEqual((Path(temporary) / filename).read_text(), "hello\n")
                self.assertEqual(len(requests), 3)
                self.assertIn("web_fetch", [tool.get("name") for tool in requests[0]["tools"]])
                self.assertTrue(any(item.get("type") == "function_call_output"
                                    and item.get("call_id") == "patch"
                                    for item in requests[2]["input"]))
                self.assertTrue(any(item.get("type") == "function_call_output"
                                    and item.get("call_id") == "web"
                                    for item in requests[2]["input"]),
                                "HTTP full-history replay lost the retrieved evidence")

    async def test_web_history_restore_is_thread_scoped_and_unambiguous(self):
        profile = fixture_profile()
        web_call = call("web_fetch", "web", {"url": "https://example.org"})
        output = {"type": "function_call_output", "call_id": "web", "output": "evidence"}
        marker = {"type": "web_search_call", "action": {"type": "open_page", "url": "https://example.org"}}
        state = object.__new__(router.RouterState)
        state.lineage = {"first": SimpleNamespace(
            profile_slug=profile.slug, prompt_cache_key="thread", input_items=[],
            output_items=[web_call, output])}
        for key, expected in (("other-thread", [marker]), ("thread", [web_call, output])):
            request = {"prompt_cache_key": key, "input": [marker]}
            state._restore_managed_web_input(profile, request)
            self.assertEqual(request["input"], expected)
        state.lineage["second"] = SimpleNamespace(
            profile_slug=profile.slug, prompt_cache_key="thread", input_items=[],
            output_items=[dict(web_call, call_id="second"), dict(output, call_id="second", output="changed")])
        request = {"prompt_cache_key": "thread", "input": [marker]}
        state._restore_managed_web_input(profile, request)
        self.assertEqual(request["input"], [marker], "ambiguous ID-less markers must not recover arbitrary evidence")
        request["input"] = [dict(marker, id="web")]
        state._restore_managed_web_input(profile, request)
        self.assertEqual(request["input"], [web_call, output])

    async def test_http_compaction_uses_shared_validation_and_preserves_headers(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                requests = []
                async with serving(scripted_backend(lambda data, n: [answer()], requests)) as (_, http, server):
                    async with http.post(server.make_url("/v1/responses"), headers={
                        "x-codex-turn-metadata": json.dumps({"request_kind": "compaction"})},
                        json={"input": "Summarize the task.", "tools": [], "stream": streaming}) as response:
                        self.assertEqual(response.status, 200)
                        if streaming:
                            events = events_from_bytes(await response.read())
                            self.assertEqual(events[-1]["type"], "response.completed", events)
                        else:
                            self.assertEqual((await response.json())["output"][0]["content"], answer()["content"])
                self.assertEqual(router._codex_request_kind(requests[0]), "compaction")
                self.assertFalse(requests[0]["stream"])
                self.assertEqual(requests[0]["input"][0]["content"], "Summarize the task.")

    async def test_recovery_does_not_publish_calls_from_a_rejected_attempt(self):
        for first_tool in ("exec_command", "apply_patch"):
            for repetition in range(2):
                with self.subTest(first_tool=first_tool, repetition=repetition):
                    requests = []
                    valid = (call(first_tool, "first", {"cmd": "echo harmless"})
                             if first_tool == "exec_command" else patch_call("first"))
                    invalid = dict(patch_call("broken"), arguments="{invalid")
                    retry = dict(valid, id="fc_retry", call_id="retry")
                    backend = scripted_backend(lambda data, n: [valid, invalid] if n == 1 else [retry], requests)
                    async with serving(backend) as (state, http, server):
                        events = await websocket_turn(http, server, [
                            {"type": "function", "name": "exec_command"},
                            {"type": "custom", "name": "apply_patch"}])
                        self.assertEqual(events[-1]["type"], "response.completed", events)
                        published = [event["item"]["call_id"] for event in events
                                     if event["type"] == "response.output_item.done"
                                     and event["item"].get("call_id")]
                        snapshot = state.lineage[events[-1]["response"]["id"]]
                        saved = [item["call_id"] for item in snapshot.output_items if item.get("call_id")]
                        self.assertEqual(published, ["retry"])
                        self.assertEqual(saved, published)
                        self.assertEqual(len(requests), 2)

    async def test_recovery_releases_required_tool_choice_after_success(self):
        for failure in ("empty", "malformed_patch"):
            for repetition in range(2):
                with self.subTest(failure=failure, repetition=repetition):
                    requests = []
                    def select(data, n):
                        if n == 1:
                            return [] if failure == "empty" else [dict(patch_call("bad"), arguments="{invalid")]
                        if n == 2 or data.get("tool_choice") == "required":
                            return [call("web_fetch", f"web_{n}", {"url": f"https://example.org/{n}"})]
                        return [answer()]
                    async with serving(scripted_backend(select, requests)) as (state, http, server):
                        events = await websocket_turn(http, server, [
                            {"type": "web_search"}, {"type": "custom", "name": "apply_patch"}])
                        self.assertEqual(events[-1]["type"], "response.completed", events)
                        self.assertEqual(requests[1]["tool_choice"], "required")
                        self.assertEqual(requests[2].get("tool_choice", "auto"), "auto")
                        self.assertEqual(len(requests), 3)
                        self.assertEqual(state._execute_managed_call.await_count, 1)

    async def test_recovery_sampler_rejection_falls_back_without_publishing_invalid_calls(self):
        for transport in ("http", "stream", "websocket"):
            with self.subTest(transport=transport):
                requests = []
                tools = [{"type": "function", "name": "exec_command", "parameters": {
                    "type": "object", "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"]}}]
                valid = call("exec_command", "valid", {"cmd": "echo harmless"})
                invalid = call("exec_command", "invalid", {"cmd": 42})
                scripted = scripted_backend(lambda data, n: [invalid] if n == 1 else [valid], requests)
                async def backend(request):
                    data = await request.json()
                    if data.get("tool_choice") == "required":
                        requests.append(data)
                        if len(requests) == 2:
                            # Reproduce llama.cpp closing the reused connection
                            # before it sends the grammar rejection headers.
                            request.transport.close()
                            return web.Response()
                        return web.json_response({"error": {
                            "message": "Failed to initialize samplers: std::exception",
                            "type": "invalid_request_error"}}, status=400)
                    return await scripted(request)
                async with serving(backend) as (state, http, server):
                    if transport == "websocket":
                        events = await websocket_turn(http, server, tools)
                        self.assertEqual(events[-1]["type"], "response.completed", events)
                        items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
                    else:
                        async with http.post(server.make_url("/v1/responses"), json={
                            "input": "Run a harmless command.", "tools": tools,
                            "stream": transport == "stream"}) as response:
                            self.assertEqual(response.status, 200)
                            if transport == "stream":
                                events = events_from_bytes(await response.read())
                                self.assertEqual(events[-1]["type"], "response.completed", events)
                                items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
                            else:
                                items = (await response.json())["output"]
                    self.assertEqual([i["call_id"] for i in items if i.get("call_id")], ["valid"])
                    self.assertEqual([r.get("tool_choice", "auto") for r in requests],
                                     ["auto", "required", "required", "auto"])
                    self.assertEqual(requests[1], requests[2])
                    self.assertEqual(requests[1]["input"], requests[3]["input"])
                    self.assertEqual(requests[1]["tools"], requests[3]["tools"])

    async def test_sampler_fallback_is_bounded_and_preserves_client_constraints(self):
        for choice in ("auto", "required", {"type": "function", "name": "exec_command"}):
            for message in ("Failed to initialize samplers: std::exception", "unrelated backend error", "disconnect"):
                with self.subTest(choice=choice, message=message):
                    requests = []
                    async def backend(request):
                        data = await request.json()
                        requests.append(data)
                        if len(requests) == 1:
                            return web.json_response({"output": [call("exec_command", "bad", {"cmd": 42})]})
                        if message == "disconnect":
                            request.transport.close()
                            return web.Response()
                        return web.json_response({"error": {"message": message}}, status=400)
                    async with serving(backend) as (state, http, server):
                        profile = state.available_profiles[fixture_profile().slug]
                        with self.assertRaises((RuntimeError, router.ServerDisconnectedError)):
                            await state._run_responses_loop(profile=profile, forward_request={
                                "input": [], "tool_choice": choice, "tools": [{"type": "function",
                                    "name": "exec_command", "parameters": {"type": "object",
                                        "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}],
                            }, web_search_enabled=False)
                    expected = 3 if message == "disconnect" or (
                        choice == "auto" and message.startswith("Failed to initialize")) else 2
                    self.assertEqual(len(requests), expected)

    async def test_recovery_does_not_retry_a_truncated_response_body(self):
        requests = []
        first = scripted_backend(lambda data, n: [call("exec_command", "bad", {"cmd": 42})], requests)
        async def backend(request):
            if not requests:
                return await first(request)
            data = await request.json()
            requests.append(data)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"type":"response.created"}\n\n')
            request.transport.close()
            return response
        async with serving(backend) as (state, http, server):
            async with http.post(server.make_url("/v1/responses"), json={
                "input": "Run a harmless command.", "stream": True,
                "tools": [{"type": "function", "name": "exec_command", "parameters": {
                    "type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}],
            }) as response:
                events = events_from_bytes(await response.read())
                self.assertEqual(events[-1]["type"], "response.failed", events)
        self.assertEqual(len(requests), 2)

    async def test_http_preserves_web_execution_and_patch_translation(self):
        for streaming in (False, True):
            for repetition in range(2):
                with self.subTest(streaming=streaming, repetition=repetition):
                    requests = []
                    def select(data, n):
                        return ([call("web_fetch", "web", {"url": "https://example.org"})]
                                if n == 1 else [patch_call("patch")])
                    async with serving(scripted_backend(select, requests)) as (state, http, server):
                        async with http.post(server.make_url("/v1/responses"), json={
                            "input": [{"type": "message", "role": "user", "content": "Research then edit."}],
                            "tools": [{"type": "web_search"}, {"type": "custom", "name": "apply_patch"}],
                            "stream": streaming}) as response:
                            self.assertEqual(response.status, 200)
                            if streaming:
                                events = events_from_bytes(await response.read())
                                self.assertEqual(events[-1]["type"], "response.completed", events)
                                items = [event["item"] for event in events if event["type"] == "response.output_item.done"]
                            else:
                                items = (await response.json())["output"]
                        self.assertEqual(len(requests), 2)
                        self.assertEqual(state._execute_managed_call.await_count, 1)
                        self.assertIn("web_fetch", [tool.get("name") for tool in requests[0]["tools"]])
                        patches = [item for item in items if item.get("name") == "apply_patch"]
                        self.assertEqual(len(patches), 1)
                        self.assertEqual(patches[0]["type"], "custom_tool_call")
                        self.assertIn("*** Add File: example.txt", patches[0]["input"])

    async def test_http_keepalive_never_splits_upstream_sse_frames(self):
        for split_kind in ("inside_json", "between_data_lines"):
            for repetition in range(2):
                with self.subTest(split_kind=split_kind, repetition=repetition):
                    if split_kind == "inside_json":
                        first = b'data: {"type":"delta","text":"hel'
                        last = b'lo"}\n\n'
                    else:
                        first = b'data: {"type":"delta",\n'
                        last = b'data: "text":"hello"}\n\n'
                    async def backend(request):
                        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                        await response.prepare(request)
                        await response.write(first)
                        await asyncio.sleep(.05)
                        await response.write(last)
                        return response
                    async with serving(backend) as (_, http, server):
                        with mock.patch.object(router, "HTTP_STREAM_KEEPALIVE_INTERVAL_SECONDS", .01):
                            async with http.post(server.make_url("/v1/chat/completions"), json={"stream": True}) as response:
                                wire = await response.read()
                    self.assertIn(b": marathon-keepalive", wire)
                    self.assertEqual(events_from_bytes(wire), [{"type": "delta", "text": "hello"}])
