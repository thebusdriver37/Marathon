#!/usr/bin/env python3
"""Bounded matched replays of captured Marathon requests on the loaded Spark.

No service changes. Keeps thinking enabled, with an optional medium/xhigh
comparison. Saves raw evidence and checks structured constraint tasks
independently of the model's self-report.
"""
import argparse
import asyncio
import copy
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import time
from aiohttp import ClientSession, ClientTimeout

SOURCE = Path('.marathon/diagnostics/spark-joe-20260922/nvfp4-main')
OUT = Path('.marathon/diagnostics/spark-constraints-20260922')
GUARD = ('Constraint discipline: Treat explicit user requirements and exclusions as binding. '
         'When a later message changes a plan, update only the stated fields and retain every unchanged restriction. '
         'Do not turn your own suggestions or guesses into user-provided facts. '
         'Before answering, check that your proposed action satisfies all applicable constraints, including output format and length. '
         'If a necessary fact is missing, say it is unknown or ask a focused question; do not invent it. '
         'Use available tools when requested or needed to verify a claim, and distinguish verified results from assumptions.')

@asynccontextmanager
async def target(local_merge):
    if not local_merge:
        yield 'http://127.0.0.1:18088', 'Qwen3.8-Flash-Next', {}
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from marathon_app.catalog import backends
    from marathon_app.pool import acquire_pool_worker
    backend = backends()['llama-swap-qwen3.8-uncensored-pool']
    # This is Marathon's configured shared broker credential, not application data.
    key = os.environ.get(backend.api_key_env, '')
    if not key:
        key = next(line.split('=', 1)[1].strip().strip('\"\'')
                   for line in Path(backend.api_key_file).read_text().splitlines()
                   if line.startswith(backend.api_key_env + '='))
    headers = {'Authorization': 'Bearer ' + key}
    lease, model = acquire_pool_worker(backend, Path('/run/user/1000/marathon'), 'instruction-probe')
    owned = False
    try:
        async with ClientSession(headers=headers) as client:
            async with client.get(backend.proxy + '/running') as response:
                response.raise_for_status()
                running = (await response.json())['running']
                owned = not any(row.get('model') == model for row in running)
        print(json.dumps({'backend':backend.proxy, 'worker':model, 'unload_after':owned}), flush=True)
        yield backend.proxy, model, headers
    finally:
        try:
            if owned:
                async with ClientSession(headers=headers, timeout=ClientTimeout(total=60)) as client:
                    async with client.post(backend.proxy + '/api/models/unload/' + model) as response:
                        response.raise_for_status()
        finally:
            lease.close()

def tasks():
    cases = []
    for name, turn in [('cafe',8),('departure',2)]:
        cases.append((name,json.loads((SOURCE/f'turn-{turn:02d}'/'request-0.json').read_text()),None))
    heldout = [
        ('pottery', 'Fictional independent task: Pottery workshop notice A says Friday Oct 9, 6-8pm, registration required, age 16+, no outside clay. Notice B says moved to Sunday Oct 11, 2-4pm; all other rules unchanged. I am 15 and have my own clay. Return only JSON with keys date (YYYY-MM-DD in 2026), start (HH:MM 24-hour), end (HH:MM), registration_required (boolean), eligible_by_age (boolean), outside_clay_allowed (boolean). Do not use tools.',
         {'date':'2026-10-11','start':'14:00','end':'16:00','registration_required':True,'eligible_by_age':False,'outside_clay_allowed':False}),
        ('pickup', 'Fictional independent task: Sam picks up my dog Wednesday at 18:20. Only Lee may borrow my red bike. My appointment is Thursday 14:15; travel 35 minutes; arrive 15 minutes early, no extra buffers. Correction: Priya replaces Sam at 19:05 Wednesday; appointment moves to Friday 13:40; other rules unchanged. Return only JSON: pickup_person, pickup_time (HH:MM), appointment_day, leave_time (HH:MM), bike_borrower. Do not use tools.',
         {'pickup_person':'Priya','pickup_time':'19:05','appointment_day':'Friday','leave_time':'12:50','bike_borrower':'Lee'}),
    ]
    for name,prompt,expected in heldout:
        body=copy.deepcopy(cases[0][1])
        body['input'][-1]['content']=[{'type':'input_text','text':prompt}]
        cases.append((name,body,expected))
    return cases

async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reasoning-comparison', action='store_true')
    parser.add_argument('--local-merge', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    cases = tasks()
    if args.reasoning_comparison:
        cases.append(('diary', json.loads((SOURCE/'turn-10'/'request-0.json').read_text()), None))
    results=[]
    started=time.monotonic()
    async with target(args.local_merge) as (endpoint, model, headers), ClientSession(headers=headers, timeout=ClientTimeout(total=300)) as client:
        for seed in [41,73]:
            for name,base,expected in cases:
                # Reverse arm order for the second seed to reduce fixed-order bias.
                arms=['stock','no_old_thinking','constraint_guard']
                if args.reasoning_comparison: arms=['medium','xhigh']
                if seed==73: arms.reverse()
                for arm in arms:
                    if time.monotonic()-started>720: raise RuntimeError('12-minute probe budget reached')
                    body=copy.deepcopy(base)
                    body.update(model=model,seed=seed,store=False,stream=True)
                    body['chat_template_kwargs'].update(enable_thinking=True,reasoning_effort='medium')
                    if args.reasoning_comparison:
                        body['chat_template_kwargs']['reasoning_effort'] = arm
                        body.setdefault('reasoning', {})['effort'] = arm
                        # Same output allowance for both efforts; do not impose a
                        # medium-specific custom thinking budget on extra-high.
                        body.pop('thinking_budget_tokens', None)
                        body['max_output_tokens'] = 8192
                    if arm=='no_old_thinking': body['chat_template_kwargs']['preserve_thinking']=False
                    if arm=='constraint_guard': body['instructions'] += '\n\n'+GUARD
                    folder=out/f'{name}-{seed}-{arm}'
                    folder.mkdir()
                    (folder/'request.json').write_text(json.dumps(body))
                    print(json.dumps({'start':name,'seed':seed,'arm':arm}),flush=True)
                    begin=time.monotonic()
                    chunks=[]
                    text=[]
                    final={}
                    async with client.post(endpoint + '/v1/responses',json=body) as response:
                        response.raise_for_status()
                        async for line in response.content:
                            chunks.append(line)
                            if not line.startswith(b'data: '): continue
                            try: event=json.loads(line[6:])
                            except ValueError: continue
                            if event.get('type')=='response.output_text.delta': text.append(event.get('delta',''))
                            if event.get('type') in ('response.completed','response.incomplete','response.failed'): final=event.get('response',{})
                    answer=''.join(text)
                    (folder/'stream.sse').write_bytes(b''.join(chunks))
                    (folder/'answer.txt').write_text(answer)
                    (folder/'completed.json').write_text(json.dumps(final))
                    passed=None
                    if expected:
                        try: passed=json.loads(answer.strip())==expected
                        except ValueError: passed=False
                    row={'case':name,'seed':seed,'arm':arm,'wall_s':time.monotonic()-begin,'status':final.get('status'),'passed':passed,'usage':final.get('usage'),'words':len(answer.split()),'answer':answer}
                    results.append(row)
                    (out/'results.json').write_text(json.dumps(results,indent=2))
                    print(json.dumps(row),flush=True)

if __name__=='__main__': asyncio.run(main())
