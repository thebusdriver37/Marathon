"""Synthetic regressions for streaming work bounds and repetition semantics."""
import asyncio
import json
from pathlib import Path
import random
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from test_router_context import fixture_profile, router_module as router

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/benchmarks'))
from bench_rust_audit import production_definitions

BASELINE = production_definitions(
    Path(__file__).resolve().parents[1] / 'scripts/benchmarks/fixtures/router_hotpaths_before.py')


def response_for(chunks):
    async def iterate(size):
        for chunk in chunks:
            yield chunk
    return SimpleNamespace(content=SimpleNamespace(iter_chunked=iterate))


async def collect(chunks):
    state = object.__new__(router.RouterState)
    return [event async for event in state._iter_sse_json(response_for(chunks))]


class StreamingBufferTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_split_preserves_utf8_delimiters_multiline_and_final_frame(self):
        wire = (b': heartbeat\r\n\r\nevent: message\ndata: {"x":\ndata: 1}\n\n'
                + 'data: {"text":"é漢😀"}\r\n\r\n'.encode()
                + b'data: []\n\ndata: malformed\n\ndata: [DONE]\n\n'
                + b'data: {"replacement":"\xff"}\n\ndata: {"final":true}')
        expected = [{'x': 1}, {'text': 'é漢😀'}, {'replacement': '\ufffd'}, {'final': True}]
        for split in range(len(wire) + 1):
            self.assertEqual(await collect([wire[:split], wire[split:]]), expected)
        self.assertEqual(await collect([wire[i:i + 1] for i in range(len(wire))]), expected)

    async def test_yields_immediately_without_reading_next_chunk(self):
        received_next = False
        async def chunks(size):
            nonlocal received_next
            yield b'data: {"n":1}\n\ndata: {"n":2}\n\n'
            received_next = True
            await asyncio.Event().wait()
        state = object.__new__(router.RouterState)
        stream = state._iter_sse_json(SimpleNamespace(content=SimpleNamespace(iter_chunked=chunks)))
        try:
            self.assertEqual(await stream.__anext__(), {'n': 1})
            self.assertFalse(received_next)
            self.assertEqual(await stream.__anext__(), {'n': 2})
            self.assertFalse(received_next)
            task = asyncio.create_task(stream.__anext__())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            await stream.aclose()

    async def test_large_frame_scanning_work_is_linear(self):
        scanned = 0
        class CountedBuffer(bytearray):
            def find(self, sub, start=0):
                nonlocal scanned
                scanned += max(0, len(self) - start)
                return super().find(sub, start)
        text = 'x' * (2 * 1024 * 1024)
        wire = b'data: ' + json.dumps({'text': text}).encode() + b'\n\n'
        chunks = [wire[i:i + 1024] for i in range(0, len(wire), 1024)]
        with mock.patch.object(router, 'bytearray', CountedBuffer, create=True):
            self.assertEqual(await collect(chunks), [{'text': text}])
        self.assertLess(scanned, 3 * len(wire))

    async def test_frame_limit_counts_each_frame_and_allows_split_delimiters(self):
        frame = b'data: {"text":"' + b'x' * 100 + b'"}'
        with mock.patch.object(router, 'ROUTER_MAX_SSE_FRAME_BYTES', len(frame)):
            for delimiter in (b'\n\n', b'\r\n\r\n'):
                wire = (frame + delimiter) * 3
                self.assertEqual(await collect([wire[i:i + 1] for i in range(len(wire))]),
                                 [{'text': 'x' * 100}] * 3)
            for suffix in (b'', b'\n\n', b'\r\n\r\n'):
                with self.assertRaisesRegex(RuntimeError, 'SSE frame exceeded'):
                    await collect([frame + b' ' + suffix])

    async def test_unterminated_frame_fails_before_reading_more_upstream(self):
        async def chunks(size):
            yield b'x' * 132
            self.fail('oversized unfinished event must stop upstream consumption')
        state = object.__new__(router.RouterState)
        with mock.patch.object(router, 'ROUTER_MAX_SSE_FRAME_BYTES', 128):
            with self.assertRaisesRegex(RuntimeError, 'SSE frame exceeded'):
                async for _ in state._iter_sse_json(
                        SimpleNamespace(content=SimpleNamespace(iter_chunked=chunks))):
                    self.fail('not a complete event')

    async def test_raw_proxy_preserves_frame_bytes_and_unterminated_tail(self):
        wire = b': ping\r\n\r\ndata: {"n":\ndata: 1}\n\ndata: unfinished'
        expected = [b': ping\r\n\r\n', b'data: {"n":\ndata: 1}\n\n', b'data: unfinished']
        for split in range(len(wire) + 1):
            chunks = iter([part for part in (wire[:split], wire[split:]) if part])
            async def read(size):
                return next(chunks, b'')
            output = [part async for part in router._iter_proxy_content(
                SimpleNamespace(read=read), keepalive_interval_seconds=0, frame_sse=True)]
            self.assertEqual(output, expected)

    async def test_raw_proxy_cancels_pending_read_while_buffering_partial_frame(self):
        cancelled = asyncio.Event()
        reads = 0
        async def read(size):
            nonlocal reads
            reads += 1
            if reads == 1:
                return b'data: {"partial":'
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        stream = router._iter_proxy_content(SimpleNamespace(read=read),
            keepalive_interval_seconds=.001, frame_sse=True)
        try:
            while reads < 2:
                self.assertIsNone(await asyncio.wait_for(stream.__anext__(), 1))
        finally:
            await stream.aclose()
        self.assertTrue(cancelled.is_set())

    async def test_non_sse_proxy_does_not_apply_frame_limit(self):
        chunks = iter([b'not an SSE event'])
        async def read(size):
            return next(chunks, b'')
        with mock.patch.object(router, 'ROUTER_MAX_SSE_FRAME_BYTES', 1):
            output = [part async for part in router._iter_proxy_content(
                SimpleNamespace(read=read), keepalive_interval_seconds=0)]
        self.assertEqual(output, [b'not an SSE event'])


class RepetitionAndTracingTests(unittest.TestCase):
    def test_repetition_matches_old_guard_at_every_period_and_threshold(self):
        rng = random.Random(471)
        old = BASELINE['_has_runaway_repetition']
        for width in range(1, 130):
            unit = ''.join(rng.choices('abcXYZé漢😀', k=width))
            span = width * max(32, (2048 + width - 1) // width)
            noise = ''.join(rng.choices('abcXYZé漢😀', k=8192))
            for size in (2047, span - 1, span, span + 1, 8192, 9000):
                repeated = (unit * (size // width + 1))[:size]
                for value in (repeated, noise + repeated, repeated + noise, noise + repeated + '!'):
                    self.assertEqual(router._has_runaway_repetition(value), old(value),
                                     (width, size, len(value)))
        # Long matching 32-character suffixes need not form a valid repeated span.
        for value in ('a' * 6000 + 'b' + 'a' * 2047,
                      'a' * 100 + 'b' + 'a' * 2047, '😀' * 2048, ''):
            self.assertEqual(router._has_runaway_repetition(value), old(value))

    def test_trace_skips_work_only_when_both_consumers_are_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            for debug, enabled in ((False, False), (False, True), (True, False)):
                state = object.__new__(router.RouterState)
                state.debug = debug
                state.telemetry = router.EventWriter(
                    Path(directory) / 'events.jsonl' if enabled else None,
                    'synthetic' if enabled else None, 'test')
                state.trace_log_path = Path(directory) / 'debug.jsonl'
                state.lock = threading.Lock()
                state._trace_seq = 0
                state._last_trace_by_model = {}
                request = {'input': [{'role': 'user', 'content': 'synthetic'}], 'tools': []}
                with mock.patch.object(router, '_input_fingerprint_summary',
                                       wraps=router._input_fingerprint_summary) as fingerprint:
                    state.trace_request(requested_model='fixture', profile=fixture_profile(),
                                        raw_request=request, normalized_request=request,
                                        path='/v1/responses', method='POST')
                    self.assertEqual(fingerprint.call_count, int(debug or enabled))
                    self.assertEqual(state._trace_seq, int(debug or enabled))
                    if debug:
                        self.assertIn('input_hash', state.trace_log_path.read_text())


class IncrementalToolMarkupTests(unittest.TestCase):
    def test_every_prefix_matches_original_guard(self):
        from bench_stream_markup import full_prefix_guard
        cases = [
            '', 'plain text', 'x < y', '<tool_call>', '  <function=exec_command>',
            'Use `<tool_call>` literally.', '<tool_`discard`call>',
            '> <tool_call>\r\nordinary text',
            '```xml\n<tool_call>\n```\n<function=exec_command>',
            '````xml\n```\n<tool_call>\n````\n<tool_call>',
            '~~~xml\n```\n<tool_call>\n~~~suffix\n<tool_call>\n~~~\n',
            '```\nexample\n```still fenced\n<tool_call>\n```\n',
            'Use ``a ` <tool_call> b`` literally.',
        ]
        rng = random.Random(20261025)
        fragments = ['<', '<tool_call>', '<function=', '`', '```', '~~~~', 'x < y',
                     '> quote', 'é漢😀', ' ', '\n', '\r', '\r\n', '\v', '\f',
                     '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029']
        cases += [''.join(rng.choices(fragments, k=40)) for _ in range(160)]
        for text in cases:
            for width in (1, 2, 7, 31, max(1, len(text))):
                scanner = router._ToolMarkupScanner()
                prefix = ''
                for start in range(0, len(text), width):
                    delta = text[start:start + width]
                    prefix += delta
                    self.assertEqual(scanner.feed(delta), full_prefix_guard(prefix),
                                     (repr(prefix), width))
                    self.assertEqual(router._has_unquoted_tool_call(prefix), full_prefix_guard(prefix))
                self.assertEqual(scanner.feed(''), full_prefix_guard(prefix))

    def test_completed_lines_are_not_rescanned(self):
        text = 'x < y\n' + '```python\n' + ('    assert left < right\n' * 1000) + '```\n'
        scanned = 0
        original = router._tool_markup_line
        def count(line, fence):
            nonlocal scanned
            scanned += len(line)
            return original(line, fence)
        scanner = router._ToolMarkupScanner()
        with mock.patch.object(router, '_tool_markup_line', side_effect=count):
            for start in range(0, len(text), 8):
                self.assertFalse(scanner.feed(text[start:start + 8]))
        self.assertLess(scanned, 4 * len(text))
        self.assertEqual(scanner.partial, '')

    def test_empty_deltas_do_not_accumulate(self):
        scanner = router._ToolMarkupScanner()
        for _ in range(1000):
            self.assertFalse(scanner.feed(''))
        self.assertEqual(scanner.prefix, [])
