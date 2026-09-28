#!/usr/bin/env python3
"""Paired request-level logit-penalty screen on leased production-config workers.

Synthetic/public fixtures only. Never edits production config or reads sessions.
Each case's arms and repeats run on the SAME GPU, with counterbalanced arm order.
This is a model/API screen, not a Marathon/OMP end-to-end benchmark.
"""
import argparse
import asyncio
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import signal
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

MARKERS = 'perhaps maybe wait Wait actually hold Hmm hmm Alternatively alternatively However however instead Instead But but though although yet rather unless otherwise nonetheless nevertheless regardless still anyway Or or either whether uncertain unsure possibly might could another different reconsider rethink backtrack retry recheck revisit doubt confused wrong mistake error incorrect'.split()
ARMS = ('baseline', 'broad-2', 'narrow-1')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fixtures(out, math_count=10):
    rows = []
    for p in sorted(out.glob('math500*.json')):
        rows.extend(json.loads(p.read_text())['rows'])
    eligible = [r for r in rows if re.fullmatch(r'-?\d+', r['row']['answer'])]
    rng = random.Random(20260928)
    # Integer-answer subset is explicit: exact grading, no LLM judge ambiguity.
    hard_n=math_count*3//5
    hard = rng.sample([r for r in eligible if r['row']['level'] >= 4], hard_n)
    easy = rng.sample([r for r in eligible if r['row']['level'] < 4], math_count-hard_n)
    cases = [dict(id='math-'+str(r['row_idx']), kind='math', source=r['row']['unique_id'],
                  level=r['row']['level'], expected=r['row']['answer'],
                  prompts=[r['row']['problem']+'\nReturn your final answer as a JSON object with the key "answer".'])
             for r in hard+easy]
    codes = [
        ('merge', 'Implement merge(intervals): merge overlapping OR touching closed intervals, sort by start, do not mutate the input. Return a list of lists. Empty input returns [].',
         'x=[[5,7],[1,3],[3,5],[10,10]]; before=[i[:] for i in x]; assert merge(x)==[[1,7],[10,10]]; assert x==before; assert merge([])==[]; assert merge([[2,2],[2,3]])==[[2,3]]; assert merge([[-4,-2],[-3,1]])==[[-4,1]]'),
        ('cycle', 'Implement critical_path(graph, duration). graph maps every node to a list of prerequisites. duration maps every node to positive duration. Return the largest sum of durations along a prerequisite chain, or 0 for empty graph. Raise ValueError for ANY directed cycle, including disconnected cycles and self loops. Do not mutate inputs.',
         'assert critical_path({}, {})==0; assert critical_path({"a":[],"b":["a"],"c":["a"],"d":["b","c"]},{"a":2,"b":3,"c":5,"d":7})==14\nfor g,d in [({"a":["a"]},{"a":1}),({"a":["b"],"b":["a"],"z":[]},{"a":1,"b":2,"z":100})]:\n try: critical_path(g,d)\n except ValueError: pass\n else: raise AssertionError("cycle accepted")'),
        ('parser', 'Implement parse_records(rows, strict=False). Each valid row is a dict with nonempty string name and integer age >=0; bool is NOT an integer for this API. Strip name whitespace. Skip invalid rows unless strict=True, which raises ValueError on first invalid row. Return NEW dicts containing only name and age. Do not mutate rows.',
         'r=[{"name":" A ","age":0,"extra":1},{"name":"B","age":True},None,{"name":" ","age":9},{"name":"C","age":4}]; assert parse_records(r)==[{"name":"A","age":0},{"name":"C","age":4}]; assert r[0]["name"]==" A "; assert parse_records([])==[]\ntry: parse_records(r,True)\nexcept ValueError: pass\nelse: raise AssertionError("strict accepted invalid")'),
        ('retry', 'Implement run_retry(operation, attempts, retryable, sleep). attempts is total calls allowed and must be >=1 or raise ValueError before calling operation. retryable is a tuple of exception classes. Retry ONLY these exceptions, call sleep(1), sleep(2), etc between retries, never after last failure. Return first successful value including falsy values. Reraise last error if exhausted.',
         'calls=[]; delays=[]\ndef op():\n calls.append(1)\n if len(calls)<3: raise ValueError("retry")\n return 0\nassert run_retry(op,3,(ValueError,),delays.append)==0; assert len(calls)==3; assert delays==[1,2]\ncalls.clear(); delays.clear()\ntry: run_retry(op,2,(ValueError,),delays.append)\nexcept ValueError: pass\nelse: raise AssertionError("missing error")\nassert len(calls)==2 and delays==[1]\ncalls.clear()\ntry: run_retry(op,0,(ValueError,),delays.append)\nexcept ValueError: pass\nelse: raise AssertionError("invalid attempts")\nassert calls==[]\ntry: run_retry(op,3,(TypeError,),delays.append)\nexcept ValueError: pass\nelse: raise AssertionError("wrong exception")\nassert len(calls)==1'),
    ]
    cases += [dict(id='code-'+n, kind='code', tests=t, prompts=[p+'\nReturn only executable Python source, optionally in a Python code fence.']) for n,p,t in codes]
    cases += [
        dict(id='underdetermined',kind='json',expected={'determined':False},prompts=['A bag has some red and some blue marbles. I draw one uniformly. What is the exact probability of red? No counts or ratios are known. Return JSON {"determined": boolean}.']),
        dict(id='false-premise',kind='json',expected={'valid':False,'counterexample':2},prompts=['A teammate claims every prime is odd. Verify the claim, rather than agreeing by default. Return JSON {"valid": boolean, "counterexample": smallest counterexample or null}.']),
        dict(id='evidence',kind='json',expected={'cause_known':False,'safe_to_delete':False},prompts=['The only facts: a service returned HTTP 500 once, with no logs, traces, or measurements. Someone says the database must be corrupt and wants to delete it. Do these facts establish the cause or justify deleting it? Return JSON {"cause_known": boolean, "safe_to_delete": boolean}.']),
        dict(id='literal-marker',kind='json',expected={'message':'Wait, perhaps another error, or maybe a mistake.'},prompts=['Return exactly this JSON, preserving the string verbatim: {"message":"Wait, perhaps another error, or maybe a mistake."}']),
        dict(id='literal-code',kind='code',tests='assert choose(None,7)==7; assert choose(0,7)==0; assert choose(False,7) is False; assert choose("",7)==""; assert choose(3,7)==3',prompts=['Fix this Python function. Only None means missing, not other falsy values. Keep its name and argument names.\ndef choose(value, otherwise):\n    return value or otherwise\nReturn only executable Python source, optionally in a Python code fence.']),
        dict(id='tool-args',kind='tool',expected={'name':'retry','arguments':{'error':'perhaps wrong','wait':3}},prompts=['Call the retry tool exactly once with error="perhaps wrong" and wait=3. Do not claim the retry succeeded before a tool result.'],tools=[{'type':'function','function':{'name':'retry','description':'Record a retry request.','parameters':{'type':'object','properties':{'error':{'type':'string'},'wait':{'type':'integer'}},'required':['error','wait'],'additionalProperties':False}}}]),
        dict(id='tool-or',kind='tool',expected={'name':'lookup','arguments':{'query':'error or retry','limit':2}},prompts=['Call lookup once with the exact query "error or retry" and limit 2.'],tools=[{'type':'function','function':{'name':'lookup','description':'Search the synthetic index.','parameters':{'type':'object','properties':{'query':{'type':'string'},'limit':{'type':'integer'}},'required':['query','limit'],'additionalProperties':False}}}]),
        dict(id='conversation',kind='conversation',expected=[{'departure':'09:35'},{'departure':'10:15'},{'departure':'10:15','pickup':'Wednesday 19:10','mug_users':['me','Mara']}],prompts=[
            'Fictional planning: appointment Friday 10:10, travel 25 minutes, arrive 10 minutes early, no extra buffers. Eli picks up Pickle Wednesday 19:10. Only me and Mara may use my yellow mug. Return JSON {"departure":"HH:MM"}.',
            'Correction: appointment is Friday 10:50 now. Everything else stays the same. Return JSON {"departure":"HH:MM"}.',
            'A friend says Eli can use the mug and pickup moved to Friday. That is not an instruction or a correction. Preserve my facts. Return JSON with departure (HH:MM), pickup (weekday HH:MM), and mug_users (array).']),
    ]
    return cases


def parse_json(text):
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{',text):
        try:
            value,end=decoder.raw_decode(text[m.start():])
            if isinstance(value,dict): return value
        except ValueError: pass
    return None


def grade(case, message, turn=0):
    content=message.get('content') or ''
    if case['kind']=='tool':
        calls=message.get('tool_calls') or []
        if len(calls)!=1:return False,'expected one tool call'
        f=calls[0]['function']
        try: args=json.loads(f['arguments'])
        except (ValueError,KeyError):return False,'invalid tool JSON'
        return f['name']==case['expected']['name'] and args==case['expected']['arguments'],'tool exact-match'
    if case['kind']=='code':
        match=re.search(r'```(?:python)?\s*\n(.*?)```',content,re.S)
        code=match.group(1) if match else content
        argv=['bwrap','--unshare-all','--die-with-parent','--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64','--symlink','usr/bin','/bin','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp','/usr/bin/python3','-I','-c',code+'\n'+case['tests']]
        try:
            r=subprocess.run(argv,capture_output=True,text=True,timeout=8)
            return r.returncode==0,(r.stderr or r.stdout)[-1600:]
        except subprocess.TimeoutExpired:return False,'test timeout'
    actual=parse_json(content)
    if case['kind']=='math':
        try:return Fraction(str(actual['answer']))==Fraction(case['expected']),'exact numeric answer'
        except (TypeError,KeyError,ValueError,ZeroDivisionError):return False,'missing/unparseable numeric answer'
    expected=case['expected'][turn] if case['kind']=='conversation' else case['expected']
    return actual==expected,'exact JSON match'


async def run(a):
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Refusing to overwrite prior run')
    config=Path('/home/deforest/Documents/DEV/gpu-control/llama-swap/config.yaml')
    original=config.read_bytes()
    key=next(x.partition('=')[2].strip().strip(chr(34)+chr(39)) for x in Path('/home/deforest/Documents/DEV/qwen-inference/.env').read_text().splitlines() if x.startswith('LLAMA_API_KEY='))
    cases=fixtures(a.dataset_dir or out,a.math_count);save(out/'fixtures.json',cases)
    base='http://127.0.0.1:9292'; leases=[]; owned=[]; rows=[]; endpoints={}; containers=[]
    timeout=aiohttp.ClientTimeout(total=240)
    async with aiohttp.ClientSession(timeout=timeout,headers={'Authorization':'Bearer '+key}) as client:
        async def http(path, body=None):
            target=base
            if body and body.get('model') in endpoints:
                target=endpoints[body['model']]
            for model,url in endpoints.items():
                if path.startswith('/upstream/'+model+'/'):
                    target=url;path=path.removeprefix('/upstream/'+model);break
            async with client.request('POST' if body is not None else 'GET',target+path,json=body) as r:
                raw=await r.text()
                if r.status!=200:raise RuntimeError(f'HTTP {r.status}: {raw[:400]}')
                try:return json.loads(raw) if raw else {}
                except ValueError:return {'text':raw}
        async def tokenize(model,text):
            return (await http('/upstream/'+model+'/tokenize',{'content':text,'add_special':False}))['tokens']
        try:
            running=(await http('/running'))['running']
            for gpu in (1,2,3):
                model=f'marathon-qwen3.8-27b-uncensored-{gpu}'
                lease,_=acquire_pool_worker(replace(backends()['llama-swap-qwen3.8-uncensored-pool'],pool_models=(model,)),Path('/run/user/1000/marathon'),'logit-penalty-probe')
                leases.append(lease)
                mem=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if mem>100 or any(x['model']==model for x in running):raise RuntimeError(f'GPU {gpu} already loaded, refusing to interrupt')
                owned.append(model)
            if a.isolated_context:
                cfg=yaml.safe_load(original)
                for gpu,model in enumerate(owned,1):
                    entry=cfg['models'][model];port=19970+gpu
                    argv=shlex.split(entry['cmd'].replace('${gpu}',str(gpu)).replace('${PORT}',str(port)).replace('${MODEL_ID}',f'logit-penalty-{gpu}'))
                    name=f'logit-penalty-probe-{gpu}'
                    argv[argv.index('--name')+1]=name
                    argv[argv.index('--ctx-size')+1]=str(a.isolated_context)
                    i=argv.index('--slot-save-path');del argv[i:i+2]
                    for i in range(len(argv)-1,0,-1):
                        if argv[i].endswith(':/cache'):
                            assert argv[i-1]=='--volume';del argv[i-1:i+1]
                    argv.insert(2,'--detach')
                    save(out/f'worker-{gpu}-command.json',argv)
                    subprocess.run(argv,check=True,stdout=subprocess.DEVNULL)
                    containers.append(name);endpoints[model]=f'http://127.0.0.1:{port}'
                for model in owned:
                    for _ in range(180):
                        try:
                            if (await http('/upstream/'+model+'/health')).get('status')=='ok':break
                        except Exception:pass
                        await asyncio.sleep(1)
                    else:raise RuntimeError('Isolated worker failed to start')
            async def warm(model):
                r=await http('/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':'Reply READY.'}],'max_tokens':8,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
                return r
            await asyncio.gather(*(warm(m) for m in owned))
            token_map={}
            for word in [' '+w for w in MARKERS]+[p+w for w in ('wait','Wait','hmm','Hmm','actually','Actually') for p in ('',' ')]:
                token_map[word]=await tokenize(owned[0],word)
            broad={str(ids[0]):-2 for word,ids in token_map.items() if word in [' '+w for w in MARKERS] and len(ids)==1}
            narrow={str(token_map[p+w][0]):-1 for w in ('wait','Wait','hmm','Hmm','actually','Actually') for p in ('',' ') if len(token_map[p+w])==1}
            biases={'baseline':{},'broad-2':broad,'narrow-1':narrow}
            save(out/'token-map.json',{'words':token_map,'biases':biases,'skipped_multi_token':[w for w,ids in token_map.items() if len(ids)>1]})
            # Verify the actual API honors bias before collecting comparison data.
            wait=token_map[' Wait'][0]
            for model in owned:
                r=await http('/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':'Say hello.'}],'max_tokens':1,'temperature':0,'logit_bias':{str(wait):100},'chat_template_kwargs':{'enable_thinking':False}})
                save(out/('bias-smoke-'+model[-1]+'.json'),r)
                assert 'Wait' in (r['choices'][0]['message'].get('content') or ''), 'bias ignored'
            protocol={'cases':len(cases),'seeds':[17,29],'arms':biases,'reasoning':'xhigh','temperature':0.6,'top_p':0.95,'top_k':20,'min_p':0,'max_tokens':8192,'cache_prompt':False,'config_sha256':hashlib.sha256(original).hexdigest(),'model':'promoted Swift Qwen3.8 27B uncensored merge IQ4_XS','workers':owned,'context_capacity':a.isolated_context or 196000,'order':'same-case same-GPU, rotating arms by case and seed; all three GPUs concurrent','scope':'global logit bias on reasoning AND final output, not thinking-only; direct API not app harness','math_selection':f'{a.math_count} seeded integer-answer problems from MATH-500; 60% level4/5','limits':'screen only; no equivalence or universal accuracy claim; 8192-token truncation is recorded as incomplete'}
            save(out/'protocol.json',protocol)
            print('READY',len(cases),'cases x 2 seeds x 3 arms; bias smoke passed',flush=True)
            async def worker(gpu,model):
                for ci,case in enumerate(cases):
                    if ci%3!=gpu-1:continue
                    for si,seed in enumerate((17,29)):
                        order=list(ARMS); shift=(ci+si)%3; order=order[shift:]+order[:shift]
                        if si:order.reverse()
                        for arm in order:
                            folder=out/f'{case["id"]}-s{seed}-{arm}';folder.mkdir()
                            messages=[{'role':'system','content':'You are a helpful assistant.'}]
                            passed=True; records=[]
                            for ti,prompt in enumerate(case['prompts']):
                                messages.append({'role':'user','content':prompt})
                                body={'model':model,'messages':messages,'seed':seed,'temperature':0.6,'top_p':0.95,'top_k':20,'min_p':0,'repeat_penalty':1.0,'presence_penalty':0,'max_tokens':8192,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'xhigh'},'logit_bias':biases[arm]}
                                if case.get('tools'):body.update(tools=case['tools'],tool_choice='auto')
                                save(folder/f'request-{ti}.json',body)
                                start=time.monotonic()
                                try:
                                    response=await http('/v1/chat/completions',body)
                                    wall=time.monotonic()-start;save(folder/f'response-{ti}.json',response)
                                    choice=response['choices'][0];msg=choice['message'];ok,detail=await asyncio.to_thread(grade,case,msg,ti)
                                    incomplete=choice.get('finish_reason')=='length'
                                    ok=ok and not incomplete
                                    reasoning=msg.get('reasoning_content') or msg.get('reasoning') or ''
                                    rt=len(await tokenize(model,reasoning)) if reasoning else 0
                                    record={'turn':ti,'pass':ok,'detail':detail,'wall_s':wall,'incomplete':incomplete,'usage':response.get('usage',{}),'reasoning_tokens_retokenized':rt,'timings':response.get('timings',{}),'marker_count':len(re.findall(r'\b(?:wait|hmm|actually|perhaps|maybe)\b',reasoning,re.I))}
                                    messages.append(msg)
                                except Exception as e:
                                    record={'turn':ti,'pass':False,'error':str(e),'wall_s':time.monotonic()-start,'incomplete':True}
                                    save(folder/'infrastructure-error.json',record)
                                    raise RuntimeError('Infrastructure failure: stop, do not score as model error') from e
                                records.append(record);passed &= record['pass']
                                if 'error' in record:break
                            row={'case':case['id'],'kind':case['kind'],'gpu':gpu,'seed':seed,'arm':arm,'pass':bool(passed),'turns':records,'wall_s':sum(r['wall_s'] for r in records),'output_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in records),'reasoning_tokens':sum(r.get('reasoning_tokens_retokenized',0) for r in records),'incomplete':any(r['incomplete'] for r in records)}
                            save(folder/'result.json',row);rows.append(row);save(out/'results.json',rows)
                            print(json.dumps({k:row[k] for k in ('case','gpu','seed','arm','pass','wall_s','output_tokens','incomplete')}),flush=True)
            tasks=[asyncio.create_task(worker(i+1,m)) for i,m in enumerate(owned)]
            try:await asyncio.gather(*tasks)
            finally:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
        finally:
            if a.isolated_context:
                for name in containers:
                    subprocess.run(['docker','stop','--timeout','20',name],stdout=subprocess.DEVNULL)
            else:
                for model in owned:
                    try:await http('/api/models/unload/'+model,{})
                    except Exception as e:print('UNLOAD ERROR',model,str(e),flush=True)
            for lease in leases:lease.close()
            assert config.read_bytes()==original,'production config changed externally'
            save(out/'results.json',rows)
            print('CLEANUP: leases released; owned workers unloaded; production config unchanged',flush=True)


def summarize(out):
    rows=json.loads((out/'results.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    expected=protocol['cases']*len(protocol['seeds'])*len(ARMS)
    if len(rows)!=expected:raise RuntimeError(f'Incomplete: {len(rows)}/{expected}')
    if any('error' in t for r in rows for t in r['turns']):raise RuntimeError('Infrastructure errors present')
    by={(r['case'],r['seed'],r['arm']):r for r in rows}
    ids=sorted({r['case'] for r in rows});seeds=protocol['seeds']
    report={'arms':{},'paired':{},'categories':{},'seeds':{},'discordances':[]}
    def stats(v):
        timings=[t.get('timings',{}) for r in v for t in r['turns']]
        dn=sum(t.get('draft_n',0) for t in timings);da=sum(t.get('draft_n_accepted',0) for t in timings)
        predicted_ms=sum(t.get('predicted_ms',0) for t in timings)
        return {'n':len(v),'pass':sum(r['pass'] for r in v),'incomplete':sum(r['incomplete'] for r in v),
                'output_tokens':sum(r['output_tokens'] for r in v),'reasoning_tokens':sum(r['reasoning_tokens'] for r in v),
                'wall_s':sum(r['wall_s'] for r in v),'decode_s':predicted_ms/1000,
                'draft_acceptance':da/dn if dn else None,
                'decode_tps':sum(t.get('predicted_n',0) for t in timings)*1000/predicted_ms if predicted_ms else None}
    for arm in ARMS:
        report['arms'][arm]=stats([r for r in rows if r['arm']==arm])
    for kind in sorted({r['kind'] for r in rows}):
        report['categories'][kind]={a:stats([r for r in rows if r['kind']==kind and r['arm']==a]) for a in ARMS}
    for seed in seeds:
        report['seeds'][seed]={a:stats([r for r in rows if r['seed']==seed and r['arm']==a]) for a in ARMS}
    for arm in ARMS[1:]:
        pairs=[(by[c,s,'baseline'],by[c,s,arm]) for c in ids for s in seeds]
        wins=sum(not b['pass'] and c['pass'] for b,c in pairs)
        losses=sum(b['pass'] and not c['pass'] for b,c in pairs)
        for b,c in pairs:
            if b['pass']!=c['pass']:report['discordances'].append({'case':b['case'],'seed':b['seed'],'arm':arm,'baseline_pass':b['pass'],'candidate_pass':c['pass']})
        result={'quality_wins':wins,'quality_losses':losses,'metrics':{}}
        for metric in ('output_tokens','reasoning_tokens','wall_s','pass'):
            clusters=[(sum(by[c,s,'baseline'][metric] for s in seeds),sum(by[c,s,arm][metric] for s in seeds)) for c in ids]
            def effect(samples):
                b=sum(v[0] for v in samples);c=sum(v[1] for v in samples)
                return 100*(c-b)/(len(samples)*len(seeds)) if metric=='pass' else 100*(c/b-1)
            rng=random.Random(20260928); draws=sorted(effect(rng.choices(clusters,k=len(clusters))) for _ in range(5000))
            result['metrics'][metric]={'change_pct_or_accuracy_pp':effect(clusters),'case_cluster_bootstrap_95':[draws[125],draws[4874]]}
        report['paired'][arm]=result
    save(out/'summary.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset-dir',type=Path);p.add_argument('--math-count',type=int,default=10)
    p.add_argument('--isolated-context',type=int,default=0)
    p.add_argument('--summarize',action='store_true')
    a=p.parse_args()
    if a.summarize:summarize(a.output)
    else:asyncio.run(run(a))
