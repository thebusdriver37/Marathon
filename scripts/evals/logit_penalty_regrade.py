#!/usr/bin/env python3
"""Second-pass grading for Stage B, from retained artifacts only.

Two documented instrument corrections, applied identically to every arm:
- prose: score the requested artifact `release.md` instead of the chat message.
- research: corrected expected values, because `notes/delta.json` is present in
  the initial workspace, so the authoritative value is 240 from turn 1 on.
Also reports the paired totals with and without the dominant outlier pair.
No inference happens here; it only re-reads receipts.
"""
import argparse
import json
from pathlib import Path
import re

from scripts.evals.logit_penalty_marathon_eval import json_of, task_definitions

PROSE_SECOND = {'min_words': 70, 'max_words': 110, 'bullets': 3,
                'tokens': ['2.4.1', 'cache', 'timeout', '2.4.0']}
RESEARCH_SECOND = [{'ttl_seconds': 240}, {'ttl_seconds': 240},
                   {'ttl_seconds': 240, 'disagreeing_sources': 3}]


def prose_checks(text, want):
    words = len(text.split())
    bullets = [line for line in text.splitlines() if line.strip().startswith('- ')]
    return {'word_count': words, 'words_in_range': want['min_words'] <= words <= want['max_words'],
            'bullet_count': len(bullets),
            'required_tokens': {t: t.lower() in text.lower() for t in want['tokens']},
            'no_em_dash': '\u2014' not in text}


def ok_prose(checks, want):
    return (checks['words_in_range'] and checks['bullet_count'] == want['bullets']
            and all(checks['required_tokens'].values()) and checks['no_em_dash'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    tasks = {task['id']: task for task in task_definitions()}
    report = {}
    for trial in sorted(p for p in out.iterdir() if p.is_dir() and '-r' in p.name):
        key, _, worker_part = trial.name.partition('-w')
        task_id, repeat, arm = key.split('-', 2)[0], key.split('-')[1], '-'.join(key.split('-')[2:])
        result = json.loads((trial / 'result.json').read_text())
        entry = {'task': task_id, 'arm': arm, 'frozen_pass': result['pass']}
        if task_id == 'prose':
            artifact = trial / 'workspace' / 'release.md'
            text = artifact.read_text() if artifact.exists() else ''
            checks = prose_checks(text, PROSE_SECOND)
            entry['artifact_second_pass'] = {'checks': checks, 'ok': ok_prose(checks, PROSE_SECOND)}
        elif task_id == 'research':
            answers = [(trial / f'answer-{i}.md').read_text()
                       for i in range(len(tasks['research']['turns']))]
            per_turn = {}
            for index, want in enumerate(RESEARCH_SECOND):
                got = json_of(answers[index]) if index < len(answers) else None
                per_turn[f'turn{index + 1}'] = {
                    'parsed': got,
                    'ok': bool(got) and all(label in got and got[label] == value
                                            for label, value in want.items())}
            entry['corrected_second_pass'] = per_turn
            entry['corrected_all'] = all(row['ok'] for row in per_turn.values())
        else:
            entry['second_pass'] = result['pass']
        report[f'{task_id}-{repeat}-{arm}'] = entry
    paired = {}
    results = json.loads((out / 'results.json').read_text())
    by = {(r['task'], r['repeat'], r['arm']): r for r in results}
    for arm in {r['arm'] for r in results} - {'baseline'}:
        pairs = [(task, repeat) for (task, repeat, name) in by if name == arm]
        for label, subset in (('all', pairs),
                              ('excluding-ledger-r1', [p for p in pairs if p != ('ledger', 1)])):
            totals = {}
            for metric in ('output_tokens', 'wall_s'):
                base = sum(by[(t, r, 'baseline')][metric] for t, r in subset if (t, r, 'baseline') in by)
                cand = sum(by[(t, r, arm)][metric] for t, r in subset)
                totals[metric] = round(100 * (cand / base - 1), 3) if base else None
            paired.setdefault(arm, {})[label] = totals
    save_path = out / 'regrade.json'
    save_path.write_text(json.dumps({'second_pass': report, 'paired_sensitivity': paired}, indent=2) + '\n')
    print(json.dumps({'paired_sensitivity': paired}, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'checks'}
                      for k, v in report.items()}, indent=2)[:2500])


if __name__ == '__main__':
    main()
