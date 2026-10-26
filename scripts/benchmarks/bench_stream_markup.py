#!/usr/bin/env python3
"""Synthetic tool-markup streaming benchmark; never opens runtime/user data.

Compare the full-prefix guard from f655a0c with the current incremental scanner.
Timings isolate parsing work, not model generation or total session latency.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import platform
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from bench_rust_audit import measure_group

ROOT = Path(__file__).resolve().parents[2]


def full_prefix_guard(text: str) -> bool:
    """Frozen f655a0c semantics, also used as an independent test oracle."""
    if '<' not in text:
        return False
    fence = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('>'):
            continue
        marker = re.match(r'(`{3,}|~{3,})', stripped)
        if marker:
            delimiter = marker.group(1)
            if fence is None:
                fence = delimiter
            elif (delimiter[0] == fence[0] and len(delimiter) >= len(fence)
                  and not stripped[len(delimiter):].strip()):
                fence = None
            continue
        if fence is not None:
            continue
        unquoted = re.sub(r'(`+).*?\1', '', line)
        if '<tool_call>' in unquoted or re.match(r'\s*<function=', unquoted):
            return True
    return False


def load_scanner():
    source = ROOT / 'scripts/routers/codex_local_router.py'
    tree = ast.parse(source.read_text())
    names = {'_tool_markup_line', '_ToolMarkupScanner'}
    nodes = [n for n in tree.body if getattr(n, 'name', '') in names]
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_ToolMarkupScanner']


def stream_replay(text, baseline_source):
    """Replay identical SSE bytes through the real router method in both arms."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts/routers'))
    import codex_local_router as router

    tree = ast.parse(baseline_source)
    state_class = next(n for n in tree.body if getattr(n, 'name', '') == 'RouterState')
    method = next(n for n in state_class.body if getattr(n, 'name', '') == '_request_responses_stream')
    namespace = dict(vars(router), _has_unquoted_tool_call=full_prefix_guard)
    validation = next(n for n in tree.body if getattr(n, 'name', '') == '_response_tool_protocol_error')
    exec(compile(ast.Module(body=[validation, method], type_ignores=[]), '<baseline-stream>', 'exec'), namespace)
    methods = {'before': namespace['_request_responses_stream'],
               'after': router.RouterState._request_responses_stream}
    item = {'type': 'message', 'role': 'assistant', 'id': 'synthetic',
            'content': [{'type': 'output_text', 'text': text}]}
    events = [{'type': 'response.output_item.added', 'item': {**item, 'content': []}}]
    events.extend({'type': 'response.output_text.delta', 'item_id': 'synthetic',
                   'delta': text[i:i + 8]} for i in range(0, len(text), 8))
    events.extend([{'type': 'response.output_item.done', 'item': item},
                   {'type': 'response.completed', 'response': {'output': [item], 'usage': {}}}])
    wire = b''.join(('data: ' + json.dumps(e) + '\n\n').encode() for e in events)

    class Response:
        status = 200
        @property
        def content(self): return self
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def iter_chunked(self, size):
            for start in range(0, len(wire), size):
                yield wire[start:start + size]

    profile = SimpleNamespace(target='http://synthetic.invalid', external=False,
                              upstream_headers=lambda: {})
    async def run(method):
        state = object.__new__(router.RouterState)
        state.http_client = SimpleNamespace(post=lambda *args, **kwargs: Response())
        received = []
        async def sink(event):
            received.append(event)
            return True
        result = await method(state, profile, {'tools': [{'type': 'function', 'name': 'synthetic',
                              'parameters': {'type': 'object'}}]}, event_sink=sink)
        return result, received
    arms = {name: lambda method=method: asyncio.run(run(method)) for name, method in methods.items()}
    if arms['before']() != arms['after']():
        raise RuntimeError('Full router replay output differs')
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-router', type=Path, help='Optional f655a0c router source for full SSE replay')
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('--repeats must be at least three')
    scanner_type = load_scanner()
    report = {'baseline_revision': 'f655a0c', 'repeats': args.repeats,
              'python': sys.version, 'platform': platform.platform(),
              'scope': 'Synthetic parsing and optional router replay; no model or private data',
              'router_sha256': hashlib.sha256((ROOT / 'scripts/routers/codex_local_router.py').read_bytes()).hexdigest(),
              'cases': {}}
    cases = {
        'ordinary_prose': 'A synthetic explanation of dictionaries and sorting.\n' * 480,
        'comparison_prose': 'For comparison, x < y.\n' + 'A synthetic explanation of dictionaries and sorting.\n' * 480,
        'fenced_code': 'Example:\n```python\n' + '    if left < right: result.append(left)  # synthetic example\n' * 400 + '```\n',
        'quoted_protocol': '> <tool_call>\n' + 'A synthetic explanation of dictionaries and sorting.\n' * 480,
        'single_long_line': 'x < y. ' + 'synthetic explanation ' * 1150,
    }
    for name, text in cases.items():
        chunks = [text[i:i + 8] for i in range(0, len(text), 8)]
        def before():
            prefix = ''
            hits = []
            for chunk in chunks:
                prefix += chunk
                hits.append(full_prefix_guard(prefix))
            return hits
        def after():
            scanner = scanner_type()
            return [scanner.feed(chunk) for chunk in chunks]
        if before() != after():
            raise RuntimeError(f'Output mismatch: {name}')
        samples = measure_group({'before': before, 'after': after}, args.repeats)
        report['cases'][name] = {'characters': len(text), 'chunks': len(chunks), **samples}
        b, a = (samples[arm]['median_wall_ms'] for arm in ('before', 'after'))
        print(f'{name}: {b:.3f} -> {a:.3f} ms ({b / a:.2f}x)', flush=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    if args.baseline_router:
        source = args.baseline_router.read_text()
        report['baseline_router_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        for name in ('ordinary_prose', 'comparison_prose', 'fenced_code', 'single_long_line'):
            arms = stream_replay(cases[name], source)
            samples = measure_group(arms, args.repeats)
            report['cases']['router_' + name] = samples
            b, a = (samples[arm]['median_wall_ms'] for arm in ('before', 'after'))
            print(f'router_{name}: {b:.3f} -> {a:.3f} ms ({b / a:.2f}x)', flush=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
