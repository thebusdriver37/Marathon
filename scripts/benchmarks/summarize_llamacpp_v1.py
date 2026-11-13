#!/usr/bin/env python3
"""Summarize synthetic v1 qualification; keep copying separate from generation."""
import argparse
import json
from pathlib import Path


def summarize(path):
    result = json.loads(path.read_text())
    groups = {}
    copy_checks = []
    for row in result['rows']:
        t = row['timings']
        key = f"{row['lines']}:{row['task']}"
        group = groups.setdefault(key, dict(runs=0, tokens=0, seconds=0, prompt_tokens=0, prompt_seconds=0, wall_seconds=0))
        group['runs'] += 1
        group['tokens'] += t['predicted_n']
        group['seconds'] += t['predicted_ms'] / 1000
        group['prompt_tokens'] += t['prompt_n']
        group['prompt_seconds'] += t['prompt_ms'] / 1000
        group['wall_seconds'] += row['wall_s']
        if row['task'] == 'copy':
            expected = [f'Ledger entry {j}: warehouse {j % 97}, pallet {j * 7919 % 100003}, inspected and sealed.'
                        for j in range(row['lines'] // 3, row['lines'] // 3 + 13)]
            copy_checks.append(dict(lines=row['lines'], repetition=row['repetition'],
                                    passed=all(line in row['content'] for line in expected)))
    for group in groups.values():
        group['decode_tps'] = group['tokens'] / group['seconds'] if group['seconds'] else None
        group['prefill_tps'] = group['prompt_tokens'] / group['prompt_seconds'] if group['prompt_seconds'] else None
        group['response_tps'] = group['tokens'] / group['wall_seconds'] if group['wall_seconds'] else None
    return dict(label=result['label'], completed=result.get('completed', False), error=result.get('error'),
                checks=[{'check': c['check'], 'passed': c['passed']} for c in result['checks']],
                copy_checks=copy_checks, measurements=groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', type=Path, nargs='+')
    a = ap.parse_args()
    print(json.dumps([summarize(p) for p in a.results], indent=2))


if __name__ == '__main__':
    main()
