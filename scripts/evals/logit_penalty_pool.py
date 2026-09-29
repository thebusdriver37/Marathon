#!/usr/bin/env python3
"""Pool one or more real-CLI blocks into one paired analysis.

Reads only retained receipts: `results.json` per block plus `regrade.json` when
present. Corrected (second-pass) grading is applied consistently per task where a
regrade exists; the frozen pass is reported next to it so both are auditable.
Metrics are per-pair paired ratios over matched same-GPU pairs, with a
pair bootstrap and explicit dominant-outlier sensitivity.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

BASELINE = 'baseline'


def load(rows_path):
    return json.loads(rows_path.read_text())


def corrected_pass(block, row):
    """Pass verdict after the documented instrument corrections, when available."""
    regrade = block / 'regrade.json'
    if not regrade.exists():
        return row['pass']
    entries = json.loads(regrade.read_text())['second_pass']
    entry = entries.get(f"{row['task']}-r{row['repeat']}-{row['arm']}")
    if entry is None:
        return row['pass']
    for key in ('second_pass', 'corrected_all'):
        if key in entry:
            return bool(entry[key])
    if 'artifact_second_pass' in entry:
        return bool(entry['artifact_second_pass']['ok'])
    return row['pass']


def ratio(items, metric):
    base = sum(item[0] for item in items)
    cand = sum(item[1] for item in items)
    return round(100 * (cand / base - 1), 3) if base else None


def bootstrap(items, seed=20260928, draws_count=5000):
    rng = random.Random(seed)
    draws = sorted(ratio(rng.choices(items, k=len(items)), None) for _ in range(draws_count))
    return [draws[draws_count // 40], draws[draws_count - draws_count // 40 - 1]]


def metrics(pairs):
    out = {}
    for name, index in (('output_tokens', 0), ('wall_s', 1)):
        items = [(b[index], c[index]) for b, c in pairs.values()]
        out[name] = {
            'change_pct': ratio(items, name),
            'pair_bootstrap_95': bootstrap(items) if items else None,
            'median_paired_change_pct': round(statistics.median(
                [100 * (c / b - 1) for b, c in items if b]), 3) if items else None,
            'pairs_improved': sum(1 for b, c in items if b and c < b),
            'pairs': len(items)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for block in args.blocks:
        block = block.resolve()
        for row in load(block / 'results.json'):
            rows.append({**row, 'block': block.name, 'corrected_pass': corrected_pass(block, row)})
    by = {(r['block'], r['task'], r['repeat'], r['arm']): r for r in rows}
    report = {'blocks': [b.name for b in args.blocks], 'pairs': 0, 'arms': {}, 'paired': {},
              'infra': {}}
    for arm in (BASELINE, *sorted({r['arm'] for r in rows} - {BASELINE})):
        subset = [r for r in rows if r['arm'] == arm]
        report['arms'][arm] = {
            'runs': len(subset),
            'pass': sum(1 for r in subset if r['pass']),
            'pass_corrected': sum(1 for r in subset if r['corrected_pass']),
            'incomplete': sum(1 for r in subset if r['incomplete']),
            'output_tokens': sum(r['output_tokens'] for r in subset),
            'input_tokens': sum(r['input_tokens'] for r in subset),
            'wall_s': round(sum(r['wall_s'] for r in subset), 3),
            'command_calls': sum(r['command_calls'] for r in subset)}
        report['infra'].setdefault(arm, {
            'lease_wait_failed': sum(1 for r in subset if not r.get('lease_wait_ok')),
            'turn_exit_nonzero': [f"{r['block']}/{r['task']}-r{r['repeat']}-{r['arm']}:t{t['turn']}"
                                  for r in subset for t in r['turns'] if t['exit'] != 0],
            'turn_timeouts': sum(t['timed_out'] for r in subset for t in r['turns'])})
    keys = sorted({(r['block'], r['task'], r['repeat']) for r in rows})
    for arm in sorted({r['arm'] for r in rows} - {BASELINE}):
        per_pair = {}
        for key in keys:
            base = by.get((*key, BASELINE))
            cand = by.get((*key, arm))
            if not base or not cand:
                continue
            per_pair[f'{key[0]}/{key[1]}-r{key[2]}'] = {
                'gpu': cand['gpu'],
                'baseline': [base['output_tokens'], base['wall_s'], base['pass'],
                             base['corrected_pass'], base['command_calls'], base['incomplete']],
                'candidate': [cand['output_tokens'], cand['wall_s'], cand['pass'],
                              cand['corrected_pass'], cand['command_calls'], cand['incomplete']]}
        pairs = {k: (v['baseline'][:2], v['candidate'][:2]) for k, v in per_pair.items()}
        # Infrastructure-excluded subset: a pair is dropped when either run had a
        # nonzero turn exit or a turn timeout, both instrument events.
        clean = {k: v for k, v in pairs.items()
                 if not (per_pair[k]['baseline'][5] or per_pair[k]['candidate'][5])}
        entry = {'per_pair': per_pair, 'metrics': metrics(pairs),
                 'metrics_infrastructure_excluded': metrics(clean)}
        def quality_of(items, index):
            return {'wins': sum(1 for v in items.values()
                                if not v['baseline'][index] and v['candidate'][index]),
                    'losses': sum(1 for v in items.values()
                                  if v['baseline'][index] and not v['candidate'][index])}
        clean_full = {k: v for k, v in per_pair.items()
                      if not (v['baseline'][5] or v['candidate'][5])}
        entry['quality'] = {'wins': sum(1 for v in per_pair.values()
                                        if not v['baseline'][2] and v['candidate'][2]),
                            'losses': sum(1 for v in per_pair.values()
                                          if v['baseline'][2] and not v['candidate'][2]),
                            'wins_corrected': sum(1 for v in per_pair.values()
                                                  if not v['baseline'][3] and v['candidate'][3]),
                            'losses_corrected': sum(1 for v in per_pair.values()
                                                    if v['baseline'][3] and not v['candidate'][3])}
        entry['quality_infrastructure_excluded'] = {
            'frozen': quality_of(clean_full, 2), 'corrected': quality_of(clean_full, 3),
            'pairs': len(clean_full)}
        ordered_by_time = sorted(pairs, key=lambda k: -pairs[k][0][1])
        ordered_by_tokens = sorted(pairs, key=lambda k: -pairs[k][0][0])
        entry['sensitivity'] = {}
        for label, drop in (('drop-top1-time', ordered_by_time[:1]),
                            ('drop-top2-time', ordered_by_time[:2]),
                            ('drop-top3-time', ordered_by_time[:3]),
                            ('drop-top1-tokens', ordered_by_tokens[:1]),
                            ('drop-top2-tokens', ordered_by_tokens[:2])):
            kept = {k: v for k, v in pairs.items() if k not in drop}
            entry['sensitivity'][label] = {'dropped': drop, **metrics(kept)}
        report['paired'][arm] = entry
        report['pairs'] = max(report['pairs'], len(pairs))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({arm: {
        'metrics': {m: v['change_pct'] for m, v in blk['metrics'].items()},
        'infra_excluded': {m: v['change_pct'] for m, v in
                          blk['metrics_infrastructure_excluded'].items()},
        'pairs': {m: v['pairs'] for m, v in blk['metrics'].items()},
        'clean_pairs': {m: v['pairs'] for m, v in blk['metrics_infrastructure_excluded'].items()},
        'quality': blk['quality'],
        'sensitivity': {name: val['output_tokens']['change_pct']
                        for name, val in blk['sensitivity'].items()}}
        for arm, blk in report['paired'].items()}, indent=2))
    print(json.dumps({'arms': report['arms'], 'infra': report['infra']}, indent=2))


if __name__ == '__main__':
    main()
