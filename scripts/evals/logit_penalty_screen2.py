#!/usr/bin/env python3
"""Stage A screen: 12 request-level logit_bias configurations on leased workers.

Follows .marathon/diagnostics/logit-penalty-2b-20260928/protocol.md. Synthetic and
public fixtures only, direct model API (not the Marathon harness). Reuses the
frozen graders from scripts/evals/logit_penalty_probe.py.
"""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import time

import aiohttp
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from marathon_app.catalog import backends
from marathon_app.pool import acquire_pool_worker
from scripts.evals.logit_penalty_probe import grade

GPU_WORKERS = {1: 'marathon-qwen3.8-27b-uncensored-1', 2: 'marathon-qwen3.8-27b-uncensored-2',
               3: 'marathon-qwen3.8-27b-uncensored-3'}
SEEDS = (17, 29)
MATH_ROWS = {'dev': range(200, 260), 'holdout': range(300, 340)}
PRIOR_FIXTURES = Path('.marathon/diagnostics/logit-penalty-20260928/fixtures.json')

# Groupings are hypotheses about the previous broad list, not established mechanisms.
GROUP_HEDGE = ['perhaps', 'maybe', 'Maybe', 'wait', 'Wait', 'actually', 'Actually', 'hold', 'Hmm',
               'Alternatively', 'However', 'however', 'instead', 'Instead']
GROUP_CONNECTOR = ['But', 'but', 'though', 'although', 'yet', 'rather', 'unless', 'otherwise',
                   'nonetheless', 'nevertheless', 'regardless', 'still', 'anyway', 'Or', 'or',
                   'either', 'whether']
GROUP_META = ['uncertain', 'unsure', 'possibly', 'might', 'could', 'another', 'different',
              'reconsider', 'rethink', 'backtrack', 'retry', 'recheck', 'revisit', 'doubt',
              'confused', 'wrong', 'mistake', 'error', 'incorrect']
GROUP_BROAD = GROUP_HEDGE + GROUP_CONNECTOR + GROUP_META

ARMS = [
    ('baseline', 0, []),
    ('broad-2', -2, GROUP_BROAD),
    ('hedge-2', -2, GROUP_HEDGE),
    ('connector-2', -2, GROUP_CONNECTOR),
    ('meta-2', -2, GROUP_META),
    ('hedge-1', -1, GROUP_HEDGE),
    ('hedge-3', -3, GROUP_HEDGE),
    ('wait-2', -2, ['wait', 'Wait']),
    ('hmm-2', -2, ['Hmm', 'hmm']),
    ('actually-2', -2, ['actually', 'Actually']),
    ('hedge+connector-2', -2, GROUP_HEDGE + GROUP_CONNECTOR),
    ('hedge+meta-2', -2, GROUP_HEDGE + GROUP_META),
]
assert len(ARMS) == 12


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def word_variants(words):
    out = []
    for word in words:
        for prefix in ('', ' '):
            if prefix + word not in out:
                out.append(prefix + word)
    return out


def dev_cases(dataset_dir):
    """Frozen dev set: fresh examples, index ranges disjoint from the held-out set."""
    rows = []
    for path in sorted(Path(dataset_dir).glob('math500*.json')):
        rows.extend(json.loads(path.read_text())['rows'])
    used = {c['id'] for c in json.loads(PRIOR_FIXTURES.read_text())} if PRIOR_FIXTURES.exists() else set()
    eligible = [r for r in rows if r['row_idx'] in MATH_ROWS['dev']
                and 'math-' + str(r['row_idx']) not in used
                and re.fullmatch(r'-?\d+', r['row']['answer'])]
    rng = random.Random(4471)
    hard = [r for r in eligible if r['row']['level'] >= 4]
    easy = [r for r in eligible if r['row']['level'] < 4]
    picked = rng.sample(hard, min(3, len(hard))) + rng.sample(easy, min(2, len(easy)))
    cases = [dict(id='math-' + str(r['row_idx']), kind='math', source=r['row']['unique_id'],
                  level=r['row']['level'], expected=r['row']['answer'],
                  prompts=[r['row']['problem'] + '\nReturn your final answer as a JSON object with the key "answer".'])
             for r in picked]
    cases += [
        dict(id='code-ttl', kind='code',
             tests='c=TTL(2); c.put("a",1); c.put("b",2); assert c.get("b")==2; assert c.get("a")==1; '
                   'c.put("cc",3); assert c.get("a") is None and c.get("b") is None and c.get("cc")==3; '
                   'c.put("cc",4); assert c.get("cc")==4 and len(c)==2; assert c.get("zzz") is None',
             prompts=['Implement TTL(max_items) as a class with put(key, value), get(key) and __len__. '
                      'get returns None for an absent key, put evicts the least recently used entry when full, '
                      'and a repeated put of a known key updates in place without growing the size. '
                      'Standard library only. Return only executable Python source, optionally fenced.']),
        dict(id='code-nest', kind='code',
             tests='assert flatten({"a":1,"b":{"c":2,"d":{"e":3}}))=={"a":1,"b.c":2,"b.d.e":3}; '
                   'assert flatten({"a":[1,{"b":2}]})=={"a.0":1,"a.1.b":2}; '
                   'assert flatten({})=={}; assert flatten({"x":None})=={"x":None}; '
                   'src={"a":{"b":1}}; flatten(src); assert src=={"a":{"b":1}}',
             prompts=['Implement flatten(nested) that joins dict keys and list indices with dots, keeps scalar '
                      'and None values as-is, does not mutate the input, and uses only the standard library. '
                      'Return only executable Python source, optionally fenced.']),
        dict(id='code-window', kind='code',
             tests='assert top_words("a b a c b a",2)==[("a",3),("b",2)]; '
                   'assert top_words("x x y y",1)==[("x",2)]; assert top_words("",3)==[]; '
                   'assert top_words("b a B A b a",2)==[("a",2),("b",3)] or top_words("b a B A b a",2)==[("b",3),("a",2)]',
             prompts=['Implement top_words(text, k) returning a list of (word, count) pairs, case-insensitive, '
                      'ordered by count descending then first appearance, ignoring empty tokens, longest tie-break '
                      'never needed because the order is fully determined. Return only executable Python source, '
                      'optionally fenced.']),
        dict(id='json-underdetermined', kind='json', expected={'determined': False},
             prompts=['A jar holds an unknown number of coins, some gold and some silver. One coin is drawn '
                      'uniformly at random. Is the exact probability of gold determined by the stated facts? '
                      'Return JSON {"determined": boolean}.']),
        dict(id='json-false-premise', kind='json',
             expected={'valid': False, 'counterexample': 0},
             prompts=['A teammate claims: for every integer n, n squared is at least n plus 2. Check it rather '
                      'than agreeing by default, and give the smallest nonnegative counterexample. '
                      'Return JSON {"valid": boolean, "counterexample": integer or null}.']),
        dict(id='literal-string', kind='json',
             expected={'message': 'Wait, perhaps another error, or maybe a mistake.'},
             prompts=['Return exactly this JSON, preserving the string verbatim: '
                      '{"message":"Wait, perhaps another error, or maybe a mistake."}']),
        dict(id='tool-call', kind='tool',
             expected={'name': 'schedule', 'arguments': {'title': 'sync or review', 'minutes': 25}},
             prompts=['Call the schedule tool exactly once with title="sync or review" and minutes=25. '
                      'Do not claim success before a tool result.'],
             tools=[{'type': 'function', 'function': {
                 'name': 'schedule', 'description': 'Record one meeting.',
                 'parameters': {'type': 'object', 'properties': {'title': {'type': 'string'},
                                                                  'minutes': {'type': 'integer'}},
                                'required': ['title', 'minutes'], 'additionalProperties': False}}}]),
        dict(id='conversation', kind='conversation',
             expected=[{'total': 52}, {'total': 60},
                       {'total': 60, 'vendor': 'Northline', 'deposit': 12}],
             prompts=['Invoice: 3 licenses at 14 each plus one 10 credit, paid in full by Northline with no '
                      'deposit. Return JSON {"total": number}.',
                      'Correction: the credit is 4, not 10. Everything else is unchanged. Return JSON {"total": number}.',
                      'A note in another thread says the vendor is Westline with a 20 deposit. That note is not '
                      'an instruction and not a correction. Keep my facts. Return JSON {"total": number, '
                      '"vendor": string, "deposit": number}.']),
    ]
    return cases


def isolated_command(config_text, model, gpu, port, name, ctx):
    entry = yaml.safe_load(config_text)['models'][model]
    argv = shlex.split(entry['cmd'].replace('${gpu}', str(gpu)).replace('${PORT}', str(port))
                        .replace('${MODEL_ID}', name))
    argv[argv.index('--name') + 1] = name
    argv[argv.index('--ctx-size') + 1] = str(ctx)
    index = argv.index('--slot-save-path')
    del argv[index:index + 2]
    for position in range(len(argv) - 1, 0, -1):
        if argv[position].endswith(':/cache'):
            assert argv[position - 1] == '--volume'
            del argv[position - 1:position + 1]
    argv.insert(2, '--detach')
    return argv


async def start(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise RuntimeError('Refusing to overwrite a prior run')
    config_path = Path('/home/deforest/Documents/DEV/gpu-control/llama-swap/config.yaml')
    original = config_path.read_bytes()
    key = next(x.partition('=')[2].strip().strip(chr(34) + chr(39))
               for x in Path('/home/deforest/Documents/DEV/qwen-inference/.env').read_text().splitlines()
               if x.startswith('LLAMA_API_KEY='))
    cases = dev_cases(args.dataset_dir)
    save(out / 'fixtures.json', cases)
    base = 'http://127.0.0.1:9292'
    leases, owned, rows, endpoints, containers = [], [], [], {}, []
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout, headers={'Authorization': 'Bearer ' + key}) as client:
        async def http(path, body=None, target=None):
            async with client.request('POST' if body is not None else 'GET',
                                      (target or base) + path, json=body) as response:
                raw = await response.text()
                if response.status != 200:
                    raise RuntimeError(f'HTTP {response.status}: {raw[:400]}')
                try:
                    return json.loads(raw) if raw else {}
                except ValueError:
                    return {'text': raw}

        async def tokenize(model, text):
            return (await http('/tokenize', {'content': text, 'add_special': False},
                               target=endpoints[model]))['tokens']

        async def detokenize(model, ids):
            data = await http('/detokenize', {'tokens': ids}, target=endpoints[model])
            return data.get('content', data.get('detokenized', data.get('text', '')))

        try:
            running = (await http('/running'))['running']
            for gpu, model in GPU_WORKERS.items():
                lease, _ = acquire_pool_worker(
                    replace(backends()['llama-swap-qwen3.8-uncensored-pool'], pool_models=(model,)),
                    Path('/run/user/1000/marathon'), 'logit-penalty-screen2')
                leases.append(lease)
                memory = int(subprocess.check_output(
                    ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                     '--format=csv,noheader,nounits'], text=True).strip())
                if memory > 100 or any(x['model'] == model for x in running):
                    raise RuntimeError(f'GPU {gpu} already loaded, refusing to interrupt')
                owned.append(model)
            for gpu, model in GPU_WORKERS.items():
                port = 19980 + gpu
                name = f'logit-penalty-screen2-{gpu}'
                argv = isolated_command(original, model, gpu, port, name, args.isolated_context)
                save(out / f'worker-{gpu}-command.json', argv)
                subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)
                containers.append(name)
                endpoints[model] = f'http://127.0.0.1:{port}'
            for model in owned:
                for _ in range(240):
                    try:
                        if (await http('/health', target=endpoints[model])).get('status') == 'ok':
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise RuntimeError('Isolated worker failed to start')
            async def warm(model):
                return await http('/v1/chat/completions', {
                    'model': model, 'messages': [{'role': 'user', 'content': 'Reply READY.'}],
                    'max_tokens': 8, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}},
                    target=endpoints[model])
            await asyncio.gather(*(warm(m) for m in owned))
            words = word_variants(GROUP_BROAD)
            token_map = {}
            for word in words:
                token_map[word] = await tokenize(owned[0], word)
            kept, skipped, roundtrip = {}, [], {}
            for word, ids in token_map.items():
                if len(ids) != 1:
                    skipped.append({'word': word, 'ids': ids})
                    continue
                text = await detokenize(owned[0], ids)
                roundtrip[str(ids[0])] = text
                if text != word:
                    skipped.append({'word': word, 'ids': ids, 'detokenize': text})
                    continue
                kept[word] = ids[0]
            biases = {}
            for name, strength, group in ARMS:
                biases[name] = {str(kept[w]): strength for w in word_variants(group) if w in kept}
            save(out / 'token-map.json', {'words': token_map, 'kept': kept, 'biases': biases,
                                          'detokenize': roundtrip,
                                          'skipped_multi_token_or_mismatch': skipped})
            wait_id = kept.get(' Wait')
            for model in owned:
                chat = await http('/v1/chat/completions', {
                    'model': model, 'messages': [{'role': 'user', 'content': 'Say hello.'}],
                    'max_tokens': 1, 'temperature': 0, 'logit_bias': {str(wait_id): 100},
                    'chat_template_kwargs': {'enable_thinking': False}}, target=endpoints[model])
                responses = await http('/v1/responses', {
                    'model': model, 'instructions': 'Answer briefly.',
                    'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Say hello.'}]}],
                    'max_output_tokens': 8, 'temperature': 0, 'logit_bias': {str(wait_id): 100}},
                    target=endpoints[model])
                save(out / f'bias-smoke-{model[-1]}.json',
                     {'chat': chat, 'responses': responses, 'wait_id': wait_id})
                assert 'Wait' in (chat['choices'][0]['message'].get('content') or ''), 'chat bias ignored'
                texts = ''.join(str(part.get('text', '')) for item in responses.get('output', [])
                                for part in item.get('content', []) if isinstance(part, dict))
                save(out / f'bias-smoke-{model[-1]}-parsed.json', {'responses_text': texts})
            save(out / 'protocol.json', {
                'stage': 'A api screen', 'cases': len(cases), 'seeds': list(SEEDS), 'arms': biases,
                'strengths': {name: strength for name, strength, _ in ARMS},
                'groups': {'hedge': GROUP_HEDGE, 'connector': GROUP_CONNECTOR, 'meta': GROUP_META},
                'reasoning': 'xhigh', 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0,
                'max_tokens': 8192, 'cache_prompt': False,
                'config_sha256': hashlib.sha256(original).hexdigest(),
                'model': 'promoted Swift Qwen3.8 27B uncensored merge IQ4_XS + DFlash2 R32',
                'workers': owned, 'context_capacity': args.isolated_context,
                'order': 'same-case same-GPU, rotating arms by case index and seed',
                'limits': 'screen only; dev cases; no equivalence claim; truncation counts as incomplete'})
            print('READY', len(cases), 'cases x', len(SEEDS), 'seeds x', len(ARMS), 'arms', flush=True)

            async def one(model, case, seed, arm, bias, gpu):
                folder = out / f'{case["id"]}-s{seed}-{arm}'
                folder.mkdir(parents=True, exist_ok=True)
                messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
                records, incomplete_any = [], False
                for turn, prompt in enumerate(case['prompts']):
                    messages.append({'role': 'user', 'content': prompt})
                    body = {'model': model, 'messages': messages, 'seed': seed, 'temperature': 0.6,
                            'top_p': 0.95, 'top_k': 20, 'min_p': 0, 'repeat_penalty': 1.0,
                            'presence_penalty': 0, 'max_tokens': 8192, 'cache_prompt': False,
                            'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'xhigh'},
                            'logit_bias': bias}
                    if case.get('tools'):
                        body.update(tools=case['tools'], tool_choice='auto')
                    save(folder / f'request-{turn}.json', body)
                    started = time.monotonic()
                    try:
                        response = await http('/v1/chat/completions', body, target=endpoints[model])
                    except Exception as error:
                        record = {'turn': turn, 'pass': False, 'error': str(error),
                                  'wall_s': time.monotonic() - started, 'incomplete': True}
                        save(folder / f'infrastructure-error-{turn}.json', record)
                        raise RuntimeError('Infrastructure failure, stop without scoring') from error
                    wall = time.monotonic() - started
                    save(folder / f'response-{turn}.json', response)
                    choice = response['choices'][0]
                    message = choice['message']
                    ok, detail = await asyncio.to_thread(grade, case, message, turn)
                    incomplete = choice.get('finish_reason') == 'length'
                    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
                    tokens = response.get('usage', {}).get('completion_tokens', 0)
                    records.append({'turn': turn, 'pass': ok and not incomplete, 'detail': detail,
                                    'wall_s': wall, 'incomplete': incomplete, 'output_tokens': tokens,
                                    'reasoning_chars': len(reasoning),
                                    'timings': response.get('timings', {})})
                    incomplete_any = incomplete_any or incomplete
                    messages.append(message)
                rows.append({'case': case['id'], 'kind': case['kind'], 'seed': seed, 'arm': arm,
                             'pass': all(r['pass'] for r in records), 'turns': records,
                             'output_tokens': sum(r['output_tokens'] for r in records),
                             'wall_s': sum(r['wall_s'] for r in records),
                             'incomplete': incomplete_any, 'gpu': gpu})
                save(out / 'progress.json', rows)

            async def worker(gpu, model):
                for index, case in enumerate(cases):
                    if index % 3 != gpu - 1:
                        continue
                    for seed_index, seed in enumerate(SEEDS):
                        order = [name for name, _, _ in ARMS]
                        shift = (index + seed_index) % len(order)
                        order = order[shift:] + order[:shift]
                        if seed_index:
                            order.reverse()
                        for arm in order:
                            await one(model, case, seed, arm, biases[arm], gpu)
            await asyncio.gather(*(worker(gpu, model) for gpu, model in GPU_WORKERS.items()))
        finally:
            for name in containers:
                subprocess.run(['docker', 'stop', '--timeout', '20', name], stdout=subprocess.DEVNULL)
            for lease in leases:
                lease.close()
            assert config_path.read_bytes() == original, 'production config changed externally'
            save(out / 'results.json', rows)
            print('CLEANUP: containers stopped, leases released, production config unchanged', flush=True)


def summarize(out):
    rows = json.loads((out / 'results.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    arms = list(protocol['arms'])
    by = {(r['case'], r['seed'], r['arm']): r for r in rows}
    cases = sorted({r['case'] for r in rows})
    expected = len(cases) * len(protocol['seeds']) * len(arms)
    if len(rows) != expected:
        raise RuntimeError(f'Incomplete: {len(rows)}/{expected}')
    if any('error' in turn for row in rows for turn in row['turns']):
        raise RuntimeError('Infrastructure errors present')
    report = {'arms': {}, 'paired': {}, 'categories': {}, 'discordances': [], 'outlier_sensitivity': {}}

    def stats(subset):
        timings = [t.get('timings', {}) for r in subset for t in r['turns']]
        decode_ms = sum(t.get('predicted_ms', 0) for t in timings)
        return {'n': len(subset), 'pass': sum(r['pass'] for r in subset),
                'incomplete': sum(r['incomplete'] for r in subset),
                'output_tokens': sum(r['output_tokens'] for r in subset),
                'wall_s': round(sum(r['wall_s'] for r in subset), 3),
                'decode_tps': (sum(t.get('predicted_n', 0) for t in timings) * 1000 / decode_ms
                               if decode_ms else None)}

    for arm in arms:
        report['arms'][arm] = stats([by[c, s, arm] for c in cases for s in protocol['seeds']])
    for kind in sorted({r['kind'] for r in rows}):
        ids = [c for c in cases if any(r['case'] == c and r['kind'] == kind for r in rows)]
        report['categories'][kind] = {a: stats([by[c, s, a] for c in ids for s in protocol['seeds']])
                                      for a in arms}
    rng = random.Random(20260928)
    for arm in arms[1:]:
        entry = {'quality_wins': 0, 'quality_losses': 0, 'metrics': {}, 'per_case': {}}
        for case in cases:
            base = [by[case, s, 'baseline'] for s in protocol['seeds']]
            cand = [by[case, s, arm] for s in protocol['seeds']]
            entry['per_case'][case] = {
                'baseline_tokens': sum(r['output_tokens'] for r in base),
                'candidate_tokens': sum(r['output_tokens'] for r in cand),
                'baseline_wall_s': round(sum(r['wall_s'] for r in base), 3),
                'candidate_wall_s': round(sum(r['wall_s'] for r in cand), 3),
                'baseline_pass': sum(r['pass'] for r in base),
                'candidate_pass': sum(r['pass'] for r in cand)}
            entry['quality_wins'] += sum(not b['pass'] and c['pass'] for b, c in zip(base, cand))
            entry['quality_losses'] += sum(b['pass'] and not c['pass'] for b, c in zip(base, cand))
            for b, c in zip(base, cand):
                if b['pass'] != c['pass']:
                    report['discordances'].append({'case': case, 'seed': b['seed'], 'arm': arm,
                                                   'baseline_pass': b['pass'], 'candidate_pass': c['pass']})
        clusters = {m: [(sum(by[c, s, 'baseline'][m] for s in protocol['seeds']),
                         sum(by[c, s, arm][m] for s in protocol['seeds'])) for c in cases]
                    for m in ('output_tokens', 'wall_s')}

        def effect(samples, metric):
            base = sum(v[0] for v in samples)
            cand = sum(v[1] for v in samples)
            return 100 * (cand / base - 1) if base else 0.0

        for metric in ('output_tokens', 'wall_s'):
            samples = clusters[metric]
            draws = sorted(effect(rng.choices(samples, k=len(samples)), metric) for _ in range(5000))
            entry['metrics'][metric] = {
                'change_pct': round(effect(samples, metric), 3),
                'case_cluster_bootstrap_95': [round(draws[125], 3), round(draws[4874], 3)],
                'median_paired_case_change_pct': round(
                    sorted(100 * (c / b - 1) for b, c in samples if b)[len(samples) // 2], 3)}
        base_pass = [by[c, s, 'baseline']['pass'] for c in cases for s in protocol['seeds']]
        cand_pass = [by[c, s, arm]['pass'] for c in cases for s in protocol['seeds']]
        entry['metrics']['pass'] = {'accuracy_pp': round(
            100 * (sum(cand_pass) - sum(base_pass)) / len(cand_pass), 3)}
        ordered = sorted(clusters['output_tokens'], key=lambda pair: -pair[0])[:2]
        rest = [pair for pair in clusters['output_tokens'] if pair not in ordered]
        entry['outlier_exclusion'] = {'dropped_cases': [cases[i] for i in
                                                        [next(k for k, p in enumerate(clusters['output_tokens'])
                                                              if p == pair) for pair in ordered]],
                                      'tokens_change_pct_excluding_top2': round(effect(rest, 'output_tokens'), 3)}
        report['paired'][arm] = entry
    report['outlier_sensitivity'] = {a: report['paired'][a]['outlier_exclusion'] for a in arms[1:]}
    save(out / 'summary.json', report)
    print(json.dumps({arm: report['paired'][arm]['metrics'] for arm in arms[1:]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path,
                        default=Path('.marathon/diagnostics/logit-penalty-20260928'))
    parser.add_argument('--isolated-context', type=int, default=65536)
    parser.add_argument('--summarize', action='store_true')
    parsed = parser.parse_args()
    if parsed.summarize:
        summarize(parsed.output)
    else:
        asyncio.run(start(parsed))
