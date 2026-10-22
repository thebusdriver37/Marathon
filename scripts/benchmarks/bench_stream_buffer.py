#!/usr/bin/env python3
"""Compare the frozen pre-fix streaming buffer with current Marathon on fake data."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random

from bench_rust_audit import measure_group, production_definitions, python_sse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, help='Save timing samples as JSON')
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('--repeats must be at least 3')

    ns = production_definitions()
    baseline_path = Path(__file__).parent / 'fixtures/router_hotpaths_before.py'
    baseline = production_definitions(baseline_path)
    results = {'repeats': args.repeats, 'cases': {}, 'source_sha256': {
        str(path.relative_to(Path(__file__).resolve().parents[2])):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), baseline_path.resolve(),
                     Path(__file__).resolve().parents[2] / 'scripts/routers/codex_local_router.py')}}
    small_events = [
        {'type': 'response.output_text.delta', 'delta': f'part {i} 漢字'}
        for i in range(8192)
    ]
    cases = [('8,192 small events', small_events)]
    for mib in (1, 8):
        cases.append((f'One {mib} MiB event', [
            {'type': 'response.completed', 'text': 'a' * (mib * 1024 * 1024)}
        ]))

    print('BEFORE = frozen pre-fix parser; AFTER = current Marathon parser.')
    print('Generated data only. No model or private conversations are used.')
    print('Large events are stress tests; these timings exclude model generation.\n')
    print(f'{"Workload":<24} {"Before":>12} {"After":>12} {"Speedup":>10}', flush=True)
    for label, events in cases:
        wire = b''.join(
            b'data: ' + json.dumps(event, ensure_ascii=False).encode() + b'\n\n'
            for event in events
        )
        arms = {
            'before': lambda: asyncio.run(python_sse(baseline, wire, 8192)),
            'after': lambda: asyncio.run(python_sse(ns, wire, 8192)),
        }
        if arms['before']() != events or arms['after']() != events:
            raise RuntimeError(f'Output mismatch for {label}')
        samples = measure_group(arms, args.repeats)
        results['cases'][label] = samples
        before = samples['before']['median_wall_ms']
        after = samples['after']['median_wall_ms']
        print(f'{label:<24} {before:>9.2f} ms {after:>9.2f} ms {before / after:>9.1f}x',
              flush=True)
    rng = random.Random(20261022)
    arguments = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789 {}[]\n', k=24576))
    prefixes = [arguments[:i] for i in range(8, len(arguments) + 1, 8)]
    old_guard, guard = baseline['_has_runaway_repetition'], ns['_has_runaway_repetition']
    if [old_guard(s) for s in prefixes] != [guard(s) for s in prefixes]:
        raise RuntimeError('Repetition guard output mismatch')
    samples = measure_group({'before': lambda: sum(old_guard(s) for s in prefixes),
                             'after': lambda: sum(guard(s) for s in prefixes)}, args.repeats)
    results['cases']['24 KiB argument guard'] = samples
    before, after = (samples[arm]['median_wall_ms'] for arm in ('before', 'after'))
    print(f'{"24 KiB argument guard":<24} {before:>9.2f} ms {after:>9.2f} ms {before / after:>9.1f}x')
    print(f'\nOutputs matched. Times are medians of {args.repeats} interleaved runs.')
    print('The AFTER implementation is the router code in this checkout.')
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(f'Timing samples saved to {args.output}')


if __name__ == '__main__':
    main()
