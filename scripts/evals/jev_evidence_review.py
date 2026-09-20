#!/usr/bin/env python3
"""Review curated aggregate experiment evidence with Jev; never ingest conversations.

Input is an explicitly authored JSON evidence packet, not a directory of traces.
Prepare requests locally by default; --send uses TYPESAFE_API_KEY from --env-file.
Responses are resumable and bound to exact request hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def key_from(path):
    key = os.environ.get('TYPESAFE_API_KEY')
    if key:
        return key
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().partition('=')
        if sep and name == 'TYPESAFE_API_KEY':
            key = value.strip().strip('\"\'')
            if key:
                return key
    raise SystemExit('TYPESAFE_API_KEY unavailable; prepared requests remain local.')


def post(request, key):
    for attempt in range(3):
        req = urllib.request.Request('https://api.typesafe.ai/v1/systemone',
            data=json.dumps(request).encode(), headers={
                'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=55) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f'Jev HTTP {error.code}; response body withheld') from None
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise RuntimeError('Jev network request failed') from None
        time.sleep(2 ** attempt)


def questions(candidates, challenge=False):
    result = {}
    for candidate in candidates:
        cid = candidate['id']
        reference = f"For candidate `{cid}` in `candidates`, using only supplied evidence: "
        if challenge:
            dimensions = {
                'survives': 'Does this direction remain worth prioritizing after considering ALL contradictory evidence, current deployment, and the hard constraints? Unknown is not affirmative evidence.',
                'already_done': 'Is the proposed action already deployed or substantially exhausted by the supplied experiments, so repeating it lacks a new mechanism?',
                'new_gpu_needed': 'Is a NEW GPU experiment necessary before making a useful next decision, rather than analyzing existing artifacts or inspecting code?',
            }
            for name, instruction in dimensions.items():
                result[cid + '_' + name] = {'type': 'noul', 'instructions': reference + instruction}
        else:
            for name, instruction in {
                'evidence': 'How strong is the supplied evidence that this offers an ADDITIONAL benefit over the CURRENT merged-target deployment? Already deployed gains are not additional gains.',
                'decode': 'How strong is the evidence for an ADDITIONAL sustained decode tokens/second gain, excluding shorter outputs and prefill improvements?',
                'task': 'How strong is the evidence for ADDITIONAL successful-task wall-time reduction without a material quality regression?',
            }.items():
                result[cid + '_' + name] = {'type': 'score', 'instructions': reference + instruction,
                    'criteria': ['Contradicted, already exhausted, already deployed, or no evidence.',
                                 'Mechanistically plausible but unmeasured on relevant workload.',
                                 'Supported by limited relevant measurements with material caveats.',
                                 'Repeated relevant measurements establish an additional benefit.']}
            result[cid + '_action'] = {'type': 'choice', 'instructions': reference +
                'Which next action best saves the user from repeating low-value work? Do not reward novelty alone.',
                'criteria': {'retain': 'Keep an already deployed successful choice; no new work.',
                             'analyze': 'Resolve a specific gap from existing artifacts or source first.',
                             'research': 'Research a materially different mechanism before any training or GPU test.',
                             'validate': 'Evidence justifies one narrowly targeted new validation.',
                             'stop': 'Reject, defer, or stop this line given constraints and counterevidence.'}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--send', action='store_true')
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    assert packet['privacy'] == 'curated aggregate measurements only; no conversation text'
    assert set(packet) == {'privacy', 'constraints', 'baseline', 'evidence', 'candidates'}
    args.output.mkdir(parents=True, exist_ok=True)
    key = key_from(args.env_file) if args.send else None
    responses = []
    for phase in ['screen', 'challenge']:
        state = json.loads(json.dumps(packet))
        if phase == 'challenge':
            state['candidates'].reverse()
            state['evidence'].reverse()
        # Deliberately withhold the screen's answers to avoid anchoring the challenge.
        request = {'model': 'jev-1.13.0', 'state': state,
                   'questions': questions(state['candidates'], phase == 'challenge')}
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        result_path = args.output / f'{phase}-response.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['request_sha256'] != digest:
                raise SystemExit(f'Request changed; use a new output directory: {phase}')
        save(args.output / f'{phase}-request.json', request)
        if not result_path.exists() and args.send:
            result = {'request_sha256': digest, 'response': post(request, key)}
            save(result_path, result)
        elif not result_path.exists():
            continue
        response = result['response']
        missing = set(request['questions']) - set(response['answers'])
        if missing:
            raise RuntimeError(f'Missing Jev answers: {sorted(missing)}')
        responses.append(response)
        print(json.dumps({'phase': phase, 'model': response.get('model'),
                          'answers': len(response['answers']), 'usage': response.get('usage')}), flush=True)
    if len(responses) == 2:
        rows = []
        for candidate in packet['candidates']:
            cid = candidate['id']
            row = {'id': cid, 'description': candidate['description']}
            for dim in ['evidence', 'decode', 'task']:
                row[dim] = responses[0]['answers'][cid + '_' + dim]['score']
            row['action'] = responses[0]['answers'][cid + '_action']['choice']
            for dim in ['survives', 'already_done', 'new_gpu_needed']:
                row[dim] = responses[1]['answers'][cid + '_' + dim]['noul']
            rows.append(row)
        save(args.output / 'review.json', {'warning': 'Advisory judgments, not measured gains or independent votes.',
                                         'candidates': rows})


if __name__ == '__main__':
    main()
