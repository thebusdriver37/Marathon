"""Synthetic tool-contract checks. No commands or inference are executed."""
import asyncio
import copy
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from test_router_context import router_module as router, fixture_profile


TOOLS = [{"type": "function", "name": "exec_command", "parameters": {
    "type": "object", "properties": {"cmd": {"type": "string"}},
    "required": ["cmd"], "additionalProperties": False}}]


def call(name="exec_command", arguments=None, identity="call"):
    return {"type": "function_call", "id": identity, "call_id": identity,
            "name": name, "arguments": json.dumps(arguments if arguments is not None else {"cmd": "echo synthetic"})}


def message(text):
    return {'type': 'message', 'role': 'assistant', 'id': 'synthetic-message',
            'content': [{'type': 'output_text', 'text': text}]}


class ToolContractTests(unittest.TestCase):
    def test_schema_and_names(self):
        for item in (call("write"), call("run"), call(arguments={"cmd": "x", "cmd-verify": "y"}),
                     call(arguments={}), call(arguments={"cmd": 1}), call(arguments=[]),
                     dict(call(), arguments='{"cmd":'), dict(call(), arguments='{"cmd":NaN}')):
            with self.subTest(item=item):
                self.assertIsNotNone(router._response_tool_protocol_error({"output": [item]}, 10000, TOOLS))
        self.assertIsNone(router._response_tool_protocol_error({"output": [call()]}, 10000, TOOLS))
        self.assertIsNotNone(router._response_tool_protocol_error({"output": [call()]}, 10000, []))

    def test_refs_nested_types_and_namespaced_names(self):
        tools = [{"type": "function", "name": "mcp__history__lookup", "parameters": {
            "type": "object", "$defs": {"ids": {"type": "array", "items": {"type": "integer"}}},
            "properties": {"ids": {"$ref": "#/$defs/ids"}}, "required": ["ids"]}}]
        for ids, valid in (([1, 2], True), ([1, "bad"], False)):
            error = router._response_tool_protocol_error({"output": [call(tools[0]['name'], {"ids": ids})]}, 10000, tools)
            self.assertEqual(error is None, valid)
        tools[0]['parameters'] = {'$ref': 'https://invalid.example/schema'}
        self.assertIn('offline', router._response_tool_protocol_error(
            {'output': [call(tools[0]['name'])]}, 10000, tools))

    def exercise(self, bad, streaming, repeated=False):
        state = object.__new__(router.RouterState)
        state.web_search_settings = SimpleNamespace(max_iterations=1)
        state.telemetry = mock.Mock()
        good = call(identity='recovered')
        # An earlier valid call in the rejected batch must not escape either.
        batches = [[call(identity='must-not-execute'), bad], [bad] if repeated else [good]]
        attempts = []
        received = []

        class Response:
            status = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            @property
            def content(self): return self
            async def iter_chunked(self, size):
                for item in self.items:
                    if item['type'] == 'message':
                        yield ('data: ' + json.dumps({'type': 'response.output_item.added',
                               'item': {**item, 'content': []}}) + '\n\n').encode()
                        # Deliberately split every marker, even '<', into chunks.
                        for char in item['content'][0]['text']:
                            yield ('data: ' + json.dumps({'type': 'response.output_text.delta',
                                   'item_id': item['id'], 'delta': char}) + '\n\n').encode()
                        yield ('data: ' + json.dumps({'type': 'response.output_item.done', 'item': item}) + '\n\n').encode()
                        continue
                    for kind in ('added', 'done'):
                        yield ('data: ' + json.dumps({'type': 'response.output_item.' + kind, 'item': item}) + '\n\n').encode()
                yield ('data: ' + json.dumps({'type': 'response.completed', 'response': {'output': self.items, 'usage': {}}}) + '\n\n').encode()

        class Client:
            def post(self, url, **kwargs):
                attempts.append(copy.deepcopy(kwargs['json']))
                response = Response()
                response.items = batches.pop(0)
                return response

        async def request_json(profile, method, path, payload):
            attempts.append(copy.deepcopy(payload))
            return {'output': batches.pop(0), 'usage': {}}

        async def sink(event):
            received.append(event)
            return True

        state.http_client = Client()
        state._request_json = request_json
        async def run():
            return await state._run_responses_loop(profile=fixture_profile(),
                forward_request={'input': [], 'tools': TOOLS}, web_search_enabled=False,
                event_sink=sink if streaming else None)
        with mock.patch.dict('os.environ', {'MARATHON_TOOL_PROTOCOL_RECOVERIES': '1'}):
            if repeated:
                with self.assertRaises(router.ToolProtocolError): asyncio.run(run())
            else:
                _, items, _ = asyncio.run(run())
                self.assertEqual([i['call_id'] for i in items], ['recovered'])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[1]['tools'], TOOLS)
        self.assertNotIn('must-not-execute', json.dumps(received))
        self.assertNotIn('bad-call', json.dumps(received))
        streamed_text = ''.join(e.get('delta', '') for e in received if e.get('type') == 'response.output_text.delta')
        self.assertNotIn('<tool_call', streamed_text)
        self.assertNotIn('<function=', streamed_text)
        if repeated:
            self.assertFalse(any(e.get('item', {}).get('type') in ('function_call', 'custom_tool_call') for e in received))

    def test_recovery_and_bounded_failure_both_transports(self):
        for streaming in (False, True):
            for bad in (call('write', identity='bad-call'),
                        call(arguments={'cmd': 'x', 'cmd-verify': 'y'}, identity='bad-call')):
                for repeated in (False, True):
                    with self.subTest(streaming=streaming, bad=bad, repeated=repeated):
                        self.exercise(bad, streaming, repeated)

    def test_plain_text_tool_markup_recovers_without_leaking_split_markers(self):
        for text in ('Preparing the file.\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\necho synthetic',
                     '<function=exec_command>\n<parameter=cmd>\necho another'):
            for streaming in (False, True):
                for repeated in (False, True):
                    with self.subTest(text=text, streaming=streaming, repeated=repeated):
                        self.exercise(message(text), streaming, repeated)

    def test_quoted_protocol_examples_are_not_calls(self):
        for text in ('Example:\n```xml\n<tool_call>\n<function=exec_command>\n```',
                     '> <tool_call>\n> <function=exec_command>',
                     'Use the literal `<tool_call>` marker.', 'For a comparison, x < y.'):
            with self.subTest(text=text):
                self.assertIsNone(router._response_tool_protocol_error({'output': [message(text)]}, 10000, TOOLS))

    def test_ordinary_text_streams_and_quoted_examples_are_preserved(self):
        text = 'An example follows.\n```xml\n<tool_call>\n<function=exec_command>\n```\nx < y.'
        state = object.__new__(router.RouterState)
        context = mock.MagicMock()
        context.__aenter__.return_value = SimpleNamespace(status=200, content=None)
        state.http_client = mock.Mock()
        state.http_client.post.return_value = context
        received = []
        async def events(_content):
            yield {'type': 'response.output_item.added', 'item': {**message(''), 'content': []}}
            for index, char in enumerate(text):
                yield {'type': 'response.output_text.delta', 'item_id': 'synthetic-message', 'delta': char}
                if index == 10:
                    self.assertTrue(any(e['type'] == 'response.output_text.delta' for e in received))
            yield {'type': 'response.output_item.done', 'item': message(text)}
            yield {'type': 'response.completed', 'response': {'output': [message(text)], 'usage': {}}}
        state._iter_sse_json = events
        async def sink(event):
            received.append(event)
            return True
        asyncio.run(state._request_responses_stream(fixture_profile(), {'tools': TOOLS, 'input': []}, event_sink=sink))
        self.assertEqual(''.join(e.get('delta', '') for e in received if e['type'] == 'response.output_text.delta'), text)
