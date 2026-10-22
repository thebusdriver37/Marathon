#!/usr/bin/env python3
"""Reproducible synthetic CPU audit. Never opens Marathon runtime/user data.

Build: CARGO_TARGET_DIR=/tmp/marathon-rust-audit-target cargo build --release \
    --locked --manifest-path scripts/benchmarks/rust_audit/Cargo.toml
Run: python3 scripts/benchmarks/bench_rust_audit.py \
    --library /tmp/marathon-rust-audit-target/release/libmarathon_rust_audit.so \
    --output /tmp/marathon-rust-audit.json

Loads selected production definitions via AST so router initialization cannot read
configuration, credentials, telemetry, or conversations. Journals are generated
in TemporaryDirectory. Native code is experimental and never used by the app.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import cProfile
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import random
import statistics
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ROUTER = ROOT / 'scripts/routers/codex_local_router.py'


def production_definitions(source=ROUTER):
    tree = ast.parse(source.read_text())
    names = {'SseEvent', '_SseFrameBuffer', '_pop_sse_frame', '_parse_sse_frame',
             '_has_runaway_repetition', '_stable_json', '_input_fingerprint_summary'}
    nodes = [n for n in tree.body if getattr(n, 'name', None) in names]
    state = next(n for n in tree.body if getattr(n, 'name', '') == 'RouterState')
    stream = next(n for n in state.body if getattr(n, 'name', '') == '_iter_sse_json')
    nodes.append(stream)
    nodes.extend(n for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'ROUTER_MAX_SSE_FRAME_BYTES'
                     for t in n.targets))
    ns = dict(Any=Any, dataclass=dataclass, json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns


class Native:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.audit_json.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t,
                                       ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        self.lib.audit_json.restype = ctypes.c_size_t
        self.lib.audit_repetition.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.audit_repetition.restype = ctypes.c_int32
        self.output = ctypes.create_string_buffer(16 * 1024 * 1024)

    def raw(self, op, data, chunk=8192):
        count = self.lib.audit_json(op, data, len(data), self.output, len(self.output), chunk)
        if count == ctypes.c_size_t(-1).value:
            raise RuntimeError('native prototype failed')
        return ctypes.string_at(self.output, count)

    def parsed(self, op, data, chunk=8192):
        return json.loads(self.raw(op, data, chunk))

    def repetition(self, value):
        data = value.encode('utf-8')  # Include Python -> Rust conversion in timing.
        result = self.lib.audit_repetition(data, len(data))
        if result < 0:
            raise RuntimeError('native repetition failed')
        return bool(result)


async def python_sse(ns, data, chunk):
    async def chunks(_):
        for i in range(0, len(data), chunk):
            yield data[i:i + chunk]
    response = types.SimpleNamespace(content=types.SimpleNamespace(iter_chunked=chunks))
    return [item async for item in ns['_iter_sse_json'](None, response)]


def cursor_sse(ns, data, chunk):
    """Same Python parser/JSON decoder; avoid suffix copies and Python min/lists."""
    buffer = bytearray()
    result = []
    scan = 0
    for i in range(0, len(data), chunk):
        buffer.extend(data[i:i + chunk])
        start = 0
        while True:
            lf = buffer.find(b'\n\n', scan)
            crlf = buffer.find(b'\r\n\r\n', scan)
            if lf < 0 and crlf < 0:
                scan = max(start, len(buffer) - 3)
                break
            end, width = (lf, 2) if lf >= 0 and (crlf < 0 or lf < crlf) else (crlf, 4)
            event = ns['_parse_sse_frame'](buffer[start:end])
            start = end + width
            scan = start
            if event is not None:
                try:
                    value = json.loads(event.data)
                    if isinstance(value, dict): result.append(value)
                except json.JSONDecodeError:
                    pass
        if start:
            del buffer[:start]
            scan -= start
    if buffer.strip():
        event = ns['_parse_sse_frame'](buffer)
        if event is not None:
            try:
                value = json.loads(event.data)
                if isinstance(value, dict): result.append(value)
            except json.JSONDecodeError:
                pass
    return result


def measure_group(functions, repeats):
    """Interleave arms, rotate order, retain every wall/CPU sample; no tail claims."""
    samples = {name: {'wall_ms': [], 'cpu_ms': []} for name in functions}
    for fn in functions.values(): fn()
    names = list(functions)
    for repeat in range(repeats):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in order:
            wall, cpu = time.perf_counter(), time.process_time()
            functions[name]()
            samples[name]['cpu_ms'].append((time.process_time() - cpu) * 1000)
            samples[name]['wall_ms'].append((time.perf_counter() - wall) * 1000)
    for values in samples.values():
        values['median_wall_ms'] = statistics.median(values['wall_ms'])
        values['median_cpu_ms'] = statistics.median(values['cpu_ms'])
        values['min_wall_ms'] = min(values['wall_ms'])
        values['max_wall_ms'] = max(values['wall_ms'])
    return samples


def profile(fn):
    profiler = cProfile.Profile()
    profiler.runcall(fn)
    stats = pstats.Stats(profiler)
    return [{'function': f'{Path(file).name}:{line}:{name}', 'calls': values[1],
             'self_ms': values[2] * 1000, 'cumulative_ms': values[3] * 1000}
            for (file, line, name), values in
            sorted(stats.stats.items(), key=lambda pair: pair[1][3], reverse=True)[:12]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=11)
    args = parser.parse_args()
    if args.repeats < 3: parser.error('--repeats must be at least 3')
    native, ns = Native(args.library), production_definitions()
    rng = random.Random(20261022)
    checks = 0
    for width in (1, 2, 7, 32, 64, 128, 129):
        for alphabet in ('abcXYZ09', 'é漢😀ß\n'):
            unit = ''.join(rng.choices(alphabet, k=width))
            for size in (0, 2047, 2048, 4096, 8192, 24576):
                for value in ((unit * (size // width + 1))[:size],
                              ''.join(rng.choices(alphabet, k=size))):
                    assert native.repetition(value) == ns['_has_runaway_repetition'](value)
                    checks += 1
    # Mixed prefix/suffix cases detect checking the wrong end of a Unicode tail.
    for width in (1, 7, 64, 128):
        unit = ''.join(rng.choices('é漢😀ßabc', k=width))
        noise = ''.join(rng.choices('é漢😀ßabc', k=8192))
        for size in (2048, 4096, 8192):
            periodic = (unit * (size // width + 1))[:size]
            for value in (noise + periodic, periodic + noise, noise + periodic + '!'):
                assert native.repetition(value) == ns['_has_runaway_repetition'](value)
                checks += 1
    for data in (b'', b': ping\n\ndata: [DONE]\n\n',
                 b'data: {"a":\ndata: 1}\r\n\r\ndata: {"z":2}',
                 b'data: nope\n\ndata: []\n\ndata: {"a":"\xff"}\n\n'):
        for chunk in (1, 2, 3, 4, 7, 16, 64, 8192):
            expected = asyncio.run(python_sse(ns, data, chunk))
            assert native.parsed(2, data, chunk) == expected == cursor_sse(ns, data, chunk)
            checks += 1
    # Vary frame boundaries independently of message boundaries.
    for _ in range(40):
        stream = b''.join(b'data: ' + json.dumps({'n': i, 'text': 'é漢😀\\n'}).encode() +
                          rng.choice((b'\n\n', b'\r\n\r\n')) for i in range(10))
        chunk = rng.randint(1, 300)
        expected = asyncio.run(python_sse(ns, stream, chunk))
        assert native.parsed(2, stream, chunk) == expected == cursor_sse(ns, stream, chunk)
        checks += 1
    cpu_model = next((line.split(':', 1)[1].strip() for line in
                     subprocess.check_output(['lscpu'], text=True, env={**os.environ, 'LC_ALL': 'C'}).splitlines()
                     if line.startswith('Model name:')), platform.processor())
    results = {'metadata': {'python': sys.version,
        'cpu_model': cpu_model, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'native_library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(), 'platform': platform.platform(),
        'rustc': subprocess.check_output(['rustc', '--version'], text=True).strip(),
        'git_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'logical_cpus': os.cpu_count(), 'repeats': args.repeats,
        'privacy': 'Generated fixtures only; no runtime configuration or private data read.',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROUTER, ROOT / 'marathon_app/history_archive.py', Path(__file__),
                      ROOT / 'scripts/benchmarks/rust_audit/src/lib.rs',
                      ROOT / 'scripts/benchmarks/rust_audit/Cargo.toml',
                      ROOT / 'scripts/benchmarks/rust_audit/Cargo.lock')}}, 'cases': {}}
    for count in (128, 2048, 8192):
        items = [{'type': 'message', 'role': 'user' if i % 2 else 'assistant',
                  'content': [{'type': 'input_text', 'text': f'Fixture {i}: ' + 'synthetic résumé 漢字 code\n' * 40}]} for i in range(count)]
        wire = json.dumps(items, ensure_ascii=False, separators=(',', ':')).encode()
        expected = ns['_input_fingerprint_summary'](items, 256)
        expected['fingerprints'] = list(expected['fingerprints'])
        assert native.parsed(1, wire) == expected
        checks += 1
        arms = {'python': lambda: ns['_input_fingerprint_summary'](items, 256),
                'rust_from_wire': lambda: native.parsed(1, wire),
                'rust_from_python_objects': lambda: native.parsed(1, json.dumps(items, ensure_ascii=False, separators=(',', ':')).encode()),
                'python_deepcopy_only': lambda: copy.deepcopy(items)}
        results['cases'][f'fingerprint_{count}'] = {'items': count, 'input_bytes': len(wire),
            'timings': measure_group(arms, args.repeats), 'profile_python': profile(arms['python'])}
    text = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789 {}[]\n', k=24576))
    for step in (8, 64):
        prefixes = [text[:i] for i in range(step, len(text) + 1, step)]
        funcs = {'python': lambda: sum(ns['_has_runaway_repetition'](s) for s in prefixes),
                 'rust_bridge': lambda: sum(native.repetition(s) for s in prefixes)}
        assert funcs['python']() == funcs['rust_bridge']()
        checks += 1
        results['cases'][f'repetition_stream_{step}'] = {'argument_chars': len(text), 'checks': len(prefixes),
            'timings': measure_group(funcs, args.repeats)}
    events = [{'type': 'response.output_text.delta', 'delta': f'part {i} 漢字', 'output_index': 0} for i in range(8192)]
    frames = [b'data: ' + json.dumps(event, ensure_ascii=False).encode() + b'\n\n' for event in events]
    wire = b''.join(frames)
    for chunk in (128, 8192):
        funcs = {'python': lambda: asyncio.run(python_sse(ns, wire, chunk)),
                 'python_cursor': lambda: cursor_sse(ns, wire, chunk),
                 'rust_bridge': lambda: native.parsed(2, wire, chunk),
                 'rust_to_bytes': lambda: native.raw(2, wire, chunk),
                 'rust_complete_frame_bridge': lambda: [v for frame in frames for v in native.parsed(2, frame)]}
        assert funcs['python']() == funcs['python_cursor']() == funcs['rust_bridge']() == events
        checks += 1
        results['cases'][f'sse_{chunk}'] = {'events': len(events), 'input_bytes': len(wire), 'chunk_bytes': chunk,
            'timings': measure_group(funcs, args.repeats)}
    for size in (1024 * 1024, 8 * 1024 * 1024):
        wire = b'data: ' + json.dumps({'type': 'response.completed', 'text': 'a' * size}).encode() + b'\n\n'
        funcs = {'python': lambda: asyncio.run(python_sse(ns, wire, 8192)),
                 'python_cursor': lambda: cursor_sse(ns, wire, 8192),
                 'rust_bridge': lambda: native.parsed(2, wire, 8192)}
        assert funcs['python']() == funcs['python_cursor']() == funcs['rust_bridge']()
        checks += 1
        results['cases'][f'sse_large_frame_{size}'] = {'input_bytes': len(wire),
            'timings': measure_group(funcs, args.repeats)}
    # This module has no import-time state access. Only explicitly generated journals are opened.
    sys.path.insert(0, str(ROOT))
    from marathon_app.history_archive import HistoryArchive
    for count in (2000, 20000):
        with tempfile.TemporaryDirectory(prefix='marathon-rust-audit-') as directory:
            root = Path(directory).resolve()
            journal = root / 'synthetic.jsonl'
            with journal.open('w') as stream:
                for i in range(count):
                    payload = {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text',
                        'text': ('needle ' if i % 100 == 0 else '') + f'fixture {i} ' + 'synthetic content ' * 120}]}
                    stream.write(json.dumps({'type': 'response_item', 'payload': payload}) + '\n')
            archive = HistoryArchive(root, journal)
            expected = archive.search('needle')
            assert expected['total_matches'] == count // 100
            assert len(expected['matches']) == 6
            funcs = {'python_first_index_and_search': lambda: HistoryArchive(root, journal).search('needle'),
                     'python_warm_search': lambda: archive.search('needle'),
                     'python_indexed_read': lambda: archive.read(expected['matches'][0]['id'])}
            for fn in funcs.values(): fn()
            checks += 1
            results['cases'][f'history_{count}'] = {'records': count, 'input_bytes': journal.stat().st_size,
                'timings': measure_group(funcs, args.repeats),
                'profile_warm_search': profile(funcs['python_warm_search'])}
    results['parity_checks'] = checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results['metadata']['completed_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    for name, case in results['cases'].items():
        print(name, json.dumps({arm: round(t['median_wall_ms'], 3) for arm, t in case['timings'].items()}), flush=True)
    print(f'{checks} fixture parity checks passed; results: {args.output}')


if __name__ == '__main__':
    main()
