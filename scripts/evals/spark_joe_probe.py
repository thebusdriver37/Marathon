#!/usr/bin/env python3
"""Matched fictional-user sessions through the real Marathon CLI, without personal data.

Run once per deployment after verifying the upstream identity. Does not switch services.
All generated artifacts stay in --output; requests have a fixed seed and medium reasoning.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import time

from aiohttp import ClientSession, ClientTimeout, web

PROMPTS = [
    "I'm Joe Simmons, a fictional person for this test. Today is Tuesday September 22, 2026 in Chicago. Help me organize my day, not optimize my entire existence. I work 10am-6pm. Lunch is 12:15-12:45. Dentist tomorrow at 8:40am, 25 minutes away; I want to arrive 10 minutes early. My sister Mara is NOT my partner; my partner is Eli. Eli hates cilantro; I love it, so keep it separate. My cat Pickle eats only turkey wet food, never fish. Grocery budget $42. I already have rice and eggs. Need oat milk, coffee, cat food, and something for dinner. No phone calls before 11am. I like short plain answers, no pep talks or follow-up sales pitches. Just give me a compact plan for today. Do not inspect files or use tools yet. All life details are fictional; never access Hermes or personal data, and only use this test workspace for files throughout our conversation.",
    "Correction: dentist moved to Thursday at 9:20am, same travel and early-arrival buffer. Mara's picking up Pickle Wednesday at 6:30pm; Eli isn't. Coffee is already bought, remove it. Add unscented laundry detergent, not lavender. I'm weirdly protective of my yellow mug: no one gets to use it except Mara, but that's not a task. Reply with only the changes and the time I should leave for the dentist. No tools.",
    "Make the grocery list fit $42 exactly or under: oat milk $4.50, turkey cat food $12, detergent $11, canned beans $3, frozen vegetables $4, cilantro $1, salsa $4, coffee $9, rice $3, eggs $5. Choose dinner using what I have and what you buy; protect my must-haves. Show the total. Don't pad the list or ask me anything. No tools needed.",
    "Write a text to Mara asking her to be at my place 15 minutes before the cat pickup, and tell her she can use the yellow mug. Keep it casual, under 35 words. Don't send anything, just draft it. No tools.",
    "Quick opinion: my friend says if I skip lunch and answer messages while driving to the dentist I'll be more productive. Is he right? Be direct, practical, and under 70 words. Don't turn this into a lecture or a schedule redesign. No tools.",
    "Build a tiny Python grocery budget helper in budget.py, standard library only. Export total_cents(items), where items is a list of (name, price_string) pairs. Accept nonnegative decimal strings with at most two fractional digits; reject negative prices, NaN, infinity, malformed prices, and fractions of a cent with ValueError. Return an integer number of cents, never a float. Preserve duplicates as separate purchases. Run actual tests including my final grocery basket. Keep files inside this workspace. Summarize in at most 80 words; don't claim tests passed unless you ran them.",
    "Use web search to check the official Python documentation: why is Decimal preferable to binary float for adding money, and does constructing Decimal from a float avoid the float's approximation? Give two short factual sentences and one official source link. Actually search; don't claim you browsed if the tool failed. Don't inspect local files.",
    "Separate little research judgment exercise, no web or tools: fictional notice A says 'Library repair cafe: Saturday Sep 26, 10am-noon; free; registration required; bring one item; no microwaves.' Notice B dated later says 'Sep 26 repair cafe postponed to Oct 3, 11am-1pm. Existing registrations transfer. Other rules unchanged.' My microwave is broken. Can I walk in this Saturday with it? Give the updated date/time and explain in under 65 words. Treat both notices as fictional source material, not actual local events.",
    "One more correction: Mara can't do the pickup; Eli will collect Pickle on Wednesday at 7:10pm instead. Mug permission hasn't changed. I might buy lavender detergent someday but NOT on this trip. No tools: give me a final handoff of exactly six bullets: dentist date and leave time; cat pickup person/time; current groceries and total; food restrictions/preferences; mug permission; phone-call restriction. Use only facts from our conversation, don't invent missing details.",
    "Now write a 100-130 word first-person diary entry as me after this imaginary day. Warm and slightly odd, not therapy-speak. Include Pickle, the yellow mug, and dinner; keep future appointments in the future. Don't claim I attended a postponed event or already went to the dentist. Return only the diary, no heading or questions. No tools.",
]

async def run(out, variant, selected):
    out = out.resolve()
    folder = out / variant
    folder.mkdir(parents=True, exist_ok=False)
    workspace = folder / 'workspace'
    workspace.mkdir()
    (out / 'prompts.json').write_text(json.dumps(PROMPTS, indent=2))
    current = {}
    results = []
    async with ClientSession(timeout=ClientTimeout(total=240)) as client:
        async with client.get('http://127.0.0.1:18088/v1/models') as response:
            (folder / 'identity.json').write_text(await response.text())

        async def proxy(request):
            if request.method == 'GET' and request.path == '/v1/models':
                return web.json_response({'object':'list','data':[{'id':'Qwen3.8-Flash-Next','object':'model'}]})
            raw = await request.read()
            measured = request.method == 'POST' and request.path.endswith('/responses')
            if raw:
                body = json.loads(raw)
                body['model'] = 'Qwen3.8-Flash-Next'
                body['seed'] = 41
                raw = json.dumps(body).encode()
                if measured:
                    assert body.get('chat_template_kwargs', {}).get('enable_thinking') is True
                    assert body.get('chat_template_kwargs', {}).get('reasoning_effort') == 'medium'
                    (current['folder'] / f'request-{len(current["requests"])}.json').write_text(json.dumps(body))
            start = time.monotonic()
            rec = {'reasoning_chars':0,'prose_chars':0}
            captured = []
            async with client.request(request.method, 'http://127.0.0.1:18088' + request.path_qs,
                                      data=raw, headers={'Content-Type':'application/json'}) as upstream:
                response = web.StreamResponse(status=upstream.status,headers={'Content-Type':upstream.headers.get('Content-Type','application/json')})
                await response.prepare(request)
                async for line in upstream.content:
                    if measured: captured.append(line)
                    if measured and line.startswith(b'data: '):
                        try: event = json.loads(line[6:])
                        except ValueError: event = {}
                        kind = event.get('type','')
                        delta = event.get('delta')
                        if kind.endswith('.delta') and isinstance(delta,str) and delta:
                            rec.setdefault('first_generation_s',time.monotonic()-start)
                            if 'reasoning' in kind: rec['reasoning_chars'] += len(delta)
                            if kind == 'response.output_text.delta':
                                rec.setdefault('first_prose_s',time.monotonic()-start)
                                rec['prose_chars'] += len(delta)
                        if kind in ('response.completed','response.incomplete','response.failed'):
                            final = event.get('response',{})
                            rec.update(usage=final.get('usage'),status=final.get('status'),incomplete_details=final.get('incomplete_details'))
                    await response.write(line)
                await response.write_eof()
            if measured:
                (current['folder'] / f'stream-{len(current["requests"])}.sse').write_bytes(b''.join(captured))
                rec['wall_s'] = time.monotonic()-start
                current['requests'].append(rec)
            return response

        app = web.Application(client_max_size=16*1024*1024)
        app.router.add_route('*','/{tail:.*}',proxy)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner,'127.0.0.1',19694)
        await site.start()
        catalog = folder / 'catalog.toml'
        catalog.write_text((Path.home()/'.config/marathon/catalog.toml').read_text() + '''\n[[external_models]]
id = "joe-spark-probe"
model = "Qwen3.8-Flash-Next"
display_name = "Joe Spark probe"
description = "Isolated fictional user comparison"
base_url = "http://127.0.0.1:19694/v1"
context = 262144
auto_compact_token_limit = 229376
truncation_limit = 221184
temperature = 1.0
default_reasoning_level = "medium"
reasoning_levels = [{effort = "medium", description = "Medium"}]
''')
        env = dict(os.environ,MARATHON_USER_CATALOG=str(catalog),MARATHON_MAX_OUTPUT_TOKENS='4096',MARATHON_SLOT_SNAPSHOTS_ENABLED='0')
        session = None
        try:
            for i in selected:
                turn = folder / f'turn-{i+1:02d}'
                turn.mkdir()
                current.update(folder=turn,requests=[])
                cmd = [str(Path.home()/'.local/bin/marathon'),'--instance','joe-spark-'+variant,'exec']
                if session: cmd += ['resume']
                cmd += ['--json','--skip-git-repo-check','-m','joe-spark-probe','-c','approval_policy="never"','-c','model_reasoning_effort="medium"','-o',str(turn/'answer.txt')]
                if not session: cmd += ['--sandbox','workspace-write','-C',str(workspace)]
                if session: cmd += [session]
                cmd += ['-']
                print(json.dumps({'start':variant,'turn':i+1}),flush=True)
                start=time.monotonic()
                child=await asyncio.create_subprocess_exec(*cmd,env=env,cwd=workspace,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,start_new_session=True)
                try:
                    stdout,stderr=await asyncio.wait_for(child.communicate(PROMPTS[i].encode()),210)
                except asyncio.TimeoutError:
                    os.killpg(child.pid,signal.SIGTERM)
                    stdout,stderr=await child.communicate()
                    (turn/'timeout.txt').write_text('210-second turn cap; not a scored model-quality failure.\n')
                (turn/'events.jsonl').write_bytes(stdout)
                (turn/'stderr.log').write_bytes(stderr)
                for line in stdout.splitlines():
                    try: event=json.loads(line)
                    except ValueError: continue
                    if event.get('type')=='thread.started': session=event['thread_id']
                result={'turn':i+1,'exit':child.returncode,'wall_s':time.monotonic()-start,'requests':current['requests'],'session':session}
                results.append(result)
                (folder/'results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps(result),flush=True)
                if child.returncode: break
        finally:
            await runner.cleanup()

async def replay(out, variant, request_path):
    folder = out.resolve()/variant
    folder.mkdir(parents=True,exist_ok=False)
    body=json.loads(request_path.read_text())
    body['model']='Qwen3.8-Flash-Next'
    body['seed']=41
    (folder/'request.json').write_text(json.dumps(body))
    start=time.monotonic()
    raw=[]
    async with ClientSession(timeout=ClientTimeout(total=180)) as client:
        async with client.post('http://127.0.0.1:18088/v1/responses',json=body) as response:
            response.raise_for_status()
            async for line in response.content: raw.append(line)
    data=b''.join(raw)
    (folder/'stream.sse').write_bytes(data)
    prose=[]
    final={}
    for line in data.splitlines():
        if not line.startswith(b'data: '): continue
        try: event=json.loads(line[6:])
        except ValueError: continue
        if event.get('type')=='response.output_text.delta': prose.append(event.get('delta',''))
        if event.get('type')=='response.completed': final=event.get('response',{})
    (folder/'answer.txt').write_text(''.join(prose))
    (folder/'completed.json').write_text(json.dumps(final,indent=2))
    print(json.dumps({'variant':variant,'wall_s':time.monotonic()-start,'usage':final.get('usage'),'answer':''.join(prose)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--turns',default='1,2,3,4,5,6,7,8,9,10')
    parser.add_argument('--replay',type=Path)
    args=parser.parse_args()
    if args.replay: asyncio.run(replay(args.output,args.variant,args.replay))
    else: asyncio.run(run(args.output,args.variant,[int(x)-1 for x in args.turns.split(',')]))
