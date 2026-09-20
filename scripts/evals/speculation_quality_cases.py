"""Synthetic, independently checked tasks for speculative runtime comparisons."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess


def cases():
    rows = {f'item_{i:02d}': {'quantity': 10+i, 'enabled': True} for i in range(24)}
    changed = json.loads(json.dumps(rows))
    for i in (0, 7, 15, 23):
        changed[f'item_{i:02d}']['quantity'] = 900-i
    changed['item_12']['enabled'] = False
    yield {'id': 'reject_stale_copy', 'kind': 'json', 'expected': changed,
           'prompt': 'Return only the full updated JSON object. Set quantities of item_00, item_07, item_15, item_23 to 900,893,885,877 respectively and disable item_12. Preserve everything else.\n'+json.dumps(rows)}
    yield {'id': 'latest_revision', 'kind': 'json', 'expected': {'owner':'Mira','port':8431,'enabled':False},
           'prompt': 'Return only JSON with owner, port, enabled from the final revision, not the old example. Old example: {"owner":"Taro","port":8080,"enabled":true}. Revision 1 changes owner to Mira. Revision 2 changes port to 8431 and enabled to false. The old example is obsolete.'}
    yield {'id': 'invoice', 'kind': 'json', 'expected': {'subtotal_cents':10027,'discount_cents':527,'shipping_cents':399,'total_cents':9899},
           'prompt': 'An order contains 17 items at 349 cents and 23 at 178 cents. Subtract a fixed 527-cent discount and add 399 cents shipping. Return only JSON integer fields subtotal_cents, discount_cents, shipping_cents, total_cents.'}
    yield {'id': 'merge_intervals', 'kind': 'code',
           'prompt': 'Implement Python merge_intervals(items) for closed real intervals whose endpoints are integers, returning sorted lists [start,end]. Merge overlap or a shared endpoint only: [1,2] and [2,3] merge; [1,2] and [3,4] must remain separate. Do not mutate input. Empty input returns []. Return one Python code block with the function, no explanation.'}
    yield {'id': 'tool_arguments', 'kind': 'tool', 'expected': {'project':'alpha','dry_run':True,'limit':7},
           'prompt': 'Call inspect_project for project alpha, dry run enabled, limit 7. Do not run any other action.',
           'tools':[{'type':'function','function':{'name':'inspect_project','description':'Inspect a project without modifying it.','parameters':{'type':'object','properties':{'project':{'type':'string'},'dry_run':{'type':'boolean'},'limit':{'type':'integer'}},'required':['project','dry_run','limit']}}}]}
    records = '\n'.join(f'Archive record {i:04d}: region=blue; revision=2; code=obsolete-{i:04d}.' for i in range(700))
    records += '\nFINAL OVERRIDE for record 0317: region=amber; revision=9; code=KITE-6284.\n'
    yield {'id':'document_override','kind':'json','expected':{'region':'amber','revision':9,'code':'KITE-6284'},
           'prompt': records+'Return only JSON region, revision, code for record 0317, using its FINAL OVERRIDE.'}


def long_cases():
    needles={137:('LARK-5921',17),3701:('MOSS-8406',23),7109:('REEF-1638',31)}
    lines=[]
    for i in range(7400):
        if i in needles:
            code,quantity=needles[i]
            lines.append(f'Archive record {i:04d}: region=amber; revision={quantity}; code={code}.')
        else:
            lines.append(f'Archive record {i:04d}: region=blue; revision=2; code=obsolete-{i:04d}.')
    expected={f'{i:04d}':{'code':code,'revision':revision} for i,(code,revision) in needles.items()}
    yield {'id':'long_three_needles','kind':'json','expected':expected,
           'prompt':'\n'.join(lines)+'\nReturn only JSON mapping record IDs 0137, 3701, 7109 to their code and revision. Preserve code spelling exactly.'}


def check(case, message):
    content=(message.get('content') or '').split('</think>')[-1].strip()
    try:
        if case['kind']=='tool':
            calls=message.get('tool_calls',[])
            return len(calls)==1 and calls[0]['function']['name']=='inspect_project' and json.loads(calls[0]['function']['arguments'])==case['expected']
        if case['kind']=='json':
            content=re.sub(r'^```(?:json)?\s*|\s*```$','',content)
            return json.loads(content)==case['expected']
        blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',content,re.S)
        code=blocks[-1] if blocks else content
        ast.parse(code)
        # No home directory, network, host /tmp, environment secrets, or GPU.
        runner='''import resource,sys,random,copy
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))
scope={}
exec(sys.stdin.read(),scope)
fn=scope['merge_intervals']
r=random.Random(92831)
tests=[[],[[1,2],[2,3]],[[4,5],[1,10]],[[3,3],[3,3]]]
for _ in range(300):
    tests.append([sorted([r.randint(-20,20),r.randint(-20,20)]) for _ in range(r.randrange(16))])
for items in tests:
    original=copy.deepcopy(items)
    expected=[]
    for a,b in sorted(items):
        if expected and a<=expected[-1][1]: expected[-1][1]=max(expected[-1][1],b)
        else: expected.append([a,b])
    assert fn(items)==expected
    assert items==original
print('PASS304')
'''
        result=subprocess.run(['bwrap','--unshare-all','--die-with-parent','--clearenv',
            '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
            '--proc','/proc','--dev','/dev','--tmpfs','/tmp','/usr/bin/python3','-I','-S','-c',runner],
            input=code,text=True,capture_output=True,timeout=6)
        return result.returncode==0 and result.stdout.strip()=='PASS304'
    except (ValueError,KeyError,TypeError,SyntaxError,subprocess.TimeoutExpired):
        return False


def run(request, prefix, model, save, width, long=False, only=None):
    rows=[]
    for temperature,seed in ([(0,6100)] if long else [(0,6100),(0.7,6101)]):
        for case in (long_cases() if long else cases()):
            if only and case['id']!=only:
                continue
            body={'model':model,'messages':[{'role':'user','content':case['prompt']}],
                  'max_tokens':1024 if long else 3072,'temperature':temperature,'seed':seed,'top_p':0.95,
                  'stream':False,'cache_prompt':False,'speculative.n_max':width,
                  'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'medium'}}
            if 'tools' in case:
                body['tools']=case['tools'];body['tool_choice']='required'
            stem=f"quality-{case['id']}-{seed}"
            save(stem+'-request.json',body)
            data=request(prefix+'/v1/chat/completions',body)
            save(stem+'-response.json',data)
            choice=data['choices'][0]
            row={'case':case['id'],'temperature':temperature,'seed':seed,
                 'pass':check(case,choice['message']),'finish_reason':choice['finish_reason'],
                 'timings':data.get('timings'),'usage':data.get('usage')}
            rows.append(row);save('quality-summary.json',rows)
            print(json.dumps(row),flush=True)
    return rows


def run_app(output, base, save):
    folder=output.resolve()/'app-smoke'
    workspace=folder/'workspace';workspace.mkdir(parents=True)
    subprocess.run(['git','init','-q',str(workspace)],check=True)
    catalog=folder/'catalog.toml'
    # Keep the lazy pool bootstrap configuration; route this task to our endpoint.
    machine_catalog=(Path.home()/'.config/marathon/catalog.toml').read_text()
    catalog.write_text(machine_catalog+'\n'+'''[[external_models]]
id = "lookup-quality-probe"
model = "synthetic-phase-cost"
display_name = "Isolated lookup quality probe"
description = "Synthetic test only"
base_url = "'''+base+'''/v1"
context = 196000
auto_compact_token_limit = 171500
truncation_limit = 163308
temperature = 0.7
default_reasoning_level = "medium"
reasoning_levels = [{effort = "medium", description = "Medium"}]
''')
    case=next(c for c in cases() if c['id']=='merge_intervals')
    prompt=('Work only in this fresh synthetic workspace. Do not read files outside it. '
            'Create solution.py and test_solution.py. '+case['prompt']+
            ' Put the implementation in solution.py rather than only in your final message. '
            'Write at least eight unit tests, run them with Python unittest, fix any failures, and report the result.')
    env=dict(os.environ,MARATHON_USER_CATALOG=str(catalog),MARATHON_MAX_OUTPUT_TOKENS='4096',
             MARATHON_SLOT_SNAPSHOTS_ENABLED='0',XDG_STATE_HOME=str(folder/'state'),XDG_CACHE_HOME=str(folder/'cache'))
    answer=folder/'answer.txt'
    instance='lookup-quality-'+output.name
    cmd=[str(Path.home()/'.local/bin/marathon'),'--instance',instance,
         'exec','--json','--sandbox','workspace-write','-m','lookup-quality-probe',
         '-c','approval_policy="never"','-c','model_reasoning_effort="medium"',
         '-C',str(workspace),'-o',str(answer),'-']
    save('app-smoke-request.json',{'prompt':prompt,'command':cmd})
    result=subprocess.run(cmd,input=prompt,text=True,capture_output=True,env=env,timeout=600)
    (folder/'events.jsonl').write_text(result.stdout);(folder/'stderr.log').write_text(result.stderr)
    solution=workspace/'solution.py'
    passed=solution.exists() and check(case,{'content':solution.read_text()})
    row={'exit_code':result.returncode,'solution_pass_304_cases':passed,
         'test_file_exists':(workspace/'test_solution.py').exists(),
         'tool_events_present':'command_execution' in result.stdout}
    save('app-smoke-summary.json',row);print(json.dumps(row),flush=True)
    return row
