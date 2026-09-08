#!/usr/bin/env python3
"""Paired speculative-inference screening with ten public, checkable workloads.

This runner sends requests to an already isolated worker; it does not manage GPUs.
Run with BASE_URL OUTPUT_DIR, or --self-check to verify the reference answers.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SEEDS = (424242, 1729, 8675309)
SYSTEM = "Follow the user's instructions carefully. Give only the requested answer."


def workloads():
    archive = []
    targets = {
        15: ("Zephyr", "Iris", "canary-7", "cobalt-731", "pending"),
        300: ("Juniper", "Omar", "staging-2", "amber-409", "approved"),
        585: ("Vesper", "Lin", "prod-8", "violet-862", "blocked"),
    }
    expected = {}
    for i in range(600):
        if i in targets:
            project, owner, host, key, status = targets[i]
            expected[project] = dict(owner=owner, host=host, key=key, status=status)
            archive.append(f"Record {i:04d}: project={project}; owner={owner}; host={host}; key={key}; status={status}.")
        else:
            archive.append(f"Record {i:04d}: project=Archive{i}; owner=Team{i % 11}; host=retired-{i % 19}; key=gray-{i * 37}; status=archived.")
    return [
        dict(id="coding", category="Python implementation", max_tokens=640, prompt="Write Python function merge_intervals(intervals). Input is a list of [start, end] pairs with start <= end. Return sorted merged intervals, merging touching intervals too. Handle empty input and negative bounds. Do not mutate the input. Return only Python code, with no imports or annotations.", reference="def merge_intervals(intervals):\n    result = []\n    for a, b in sorted(intervals):\n        if result and a <= result[-1][1]:\n            result[-1][1] = max(result[-1][1], b)\n        else:\n            result.append([a, b])\n    return result"),
        dict(id="debugging", category="Python bug repair", max_tokens=512, prompt="Fix this binary search. first_true(values) receives a list containing zero or more False values followed by zero or more True values. Return the index of the first True, or len(values) if none. It must handle an empty list. Keep O(log n) time. Return only the corrected Python function, with no imports or annotations.\nBuggy code:\ndef first_true(values):\n    lo, hi = 0, len(values)-1\n    while lo < hi:\n        mid = (lo+hi)//2\n        if values[mid]: hi = mid-1\n        else: lo = mid\n    return lo", reference="def first_true(values):\n    lo, hi = 0, len(values)\n    while lo < hi:\n        mid = (lo+hi)//2\n        if values[mid]: hi = mid\n        else: lo = mid+1\n    return lo"),
        dict(id="sql", category="SQL aggregation", max_tokens=512, prompt="SQLite tables: customers(id INTEGER PRIMARY KEY, name TEXT); orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount_cents INTEGER, status TEXT). Write one SELECT query returning customer id and total_completed_cents for EVERY customer, including zero for those without completed orders. Only status='completed' counts. Sort total descending, then customer id ascending. Return only SQL.", reference="SELECT c.id, COALESCE(SUM(CASE WHEN o.status='completed' THEN o.amount_cents ELSE 0 END),0) AS total_completed_cents FROM customers c LEFT JOIN orders o ON o.customer_id=c.id GROUP BY c.id ORDER BY total_completed_cents DESC,c.id ASC;"),
        dict(id="arithmetic", category="Exact arithmetic", max_tokens=384, prompt="A store starts with 120 red units and 80 blue units. It sells three fifths of the red units and one quarter of the blue units, then receives 30 red units and 15 blue units. Return only JSON with red_remaining, blue_remaining, total_remaining, and red_fraction. red_fraction must be the reduced fraction as a string.", reference={"red_remaining":78,"blue_remaining":75,"total_remaining":153,"red_fraction":"26/51"}),
        dict(id="extraction", category="Structured extraction", max_tokens=384, prompt="Extract the latest state for each account from this out-of-order log. Higher revision wins. Return only JSON {\"active_accounts\":[...]} listing active account names alphabetically, with no duplicates.\naccount=Zoe revision=2 active=true\naccount=Ada revision=3 active=false\naccount=Max revision=1 active=true\naccount=Ada revision=1 active=true\naccount=Zoe revision=1 active=false\naccount=Lin revision=4 active=true\naccount=Max revision=2 active=false\naccount=Ada revision=4 active=true\naccount=Lin revision=2 active=false", reference={"active_accounts":["Ada","Lin","Zoe"]}),
        dict(id="summarization", category="Meeting summarization", max_tokens=512, prompt="Summarize this meeting as JSON with keys summary, owner, deadline, blockers, cancelled. summary must be a string of 35-60 words. owner, deadline, and cancelled must be strings. blockers must be a list of strings. Do not invent facts.\nMeeting: The initial plan was Friday, but the team explicitly replaced that deadline with Tuesday. Maya owns the release. The release cannot proceed until the security review and migration rehearsal both finish. The marketing email was cancelled; the release itself was not cancelled. Noel offered to help with testing but is not the release owner. Return only JSON.", reference={"summary":"Maya owns the release, now scheduled for Tuesday instead of Friday. Security review and migration rehearsal must both finish before release. The marketing email was cancelled, but the release remains planned. Noel offered testing help and is not the release owner.","owner":"Maya","deadline":"Tuesday","blockers":["security review","migration rehearsal"],"cancelled":"marketing email"}),
        dict(id="planning", category="Dependency scheduling", max_tokens=512, prompt="Schedule five tasks on two identical workers numbered 0 and 1. Tasks cannot be interrupted. Durations: A=3, B=2, C=4, D=2, E=1. C depends on A. D depends on B. E depends on both C and D. Time starts at 0. Find a schedule finishing by time 8. Return only JSON mapping each task name to {\"worker\": integer, \"start\": integer}. A worker cannot run overlapping tasks.", reference={"A":{"worker":0,"start":0},"B":{"worker":1,"start":0},"C":{"worker":0,"start":3},"D":{"worker":1,"start":2},"E":{"worker":0,"start":7}}),
        dict(id="tool_use", category="Tool calling", max_tokens=384, prompt="Use lookup_inventory to check both AX-17 and BZ-42 in the north warehouse, including reserved stock. Do not guess the results or call another tool.", tools=[{"type":"function","function":{"name":"lookup_inventory","description":"Look up inventory for SKU codes in a warehouse.","parameters":{"type":"object","properties":{"skus":{"type":"array","items":{"type":"string"}},"warehouse":{"type":"string"},"include_reserved":{"type":"boolean"}},"required":["skus","warehouse","include_reserved"],"additionalProperties":False}}}], reference={"tool_calls":[{"type":"function","function":{"name":"lookup_inventory","arguments":json.dumps({"skus":["AX-17","BZ-42"],"warehouse":"north","include_reserved":True})}}]}),
        dict(id="creative", category="Constrained creative writing", max_tokens=512, prompt="Write a coherent miniature mystery in exactly four sentences and 90-130 words total. Start the sentences with Dawn, Inside, Meanwhile, and Finally, respectively. Include a rusty key, a comet, and a locked observatory. Use no dialogue or quotation marks. End with the exact words 'the door opened.' Return only the story.", reference="Dawn revealed fresh footprints circling the locked observatory, although the mountain road had been buried beneath untouched snow since the previous evening and nobody remembered seeing a visitor arrive. Inside the caretaker's abandoned cabin, Mara discovered a rusty key wrapped in a faded photograph of a comet that would not return for another century. Meanwhile the footprints slowly filled with warm water, and a ticking sound beneath the frozen ground matched the rhythm of the observatory's silent clock. Finally Mara placed the photograph against the frost on the entrance, saw yesterday's date appear beneath her reflection, and turned the key until the door opened."),
        dict(id="long_context", category="Long-context retrieval", max_tokens=512, prompt="Read the archive below. Retrieve ONLY Zephyr, Juniper, and Vesper. Return JSON mapping each project to exactly its owner, host, key, and status. Ignore archived projects.\n"+"\n".join(archive)+"\nNow return the requested JSON only.", reference=expected),
    ]


def strip_fence(text):
    text = text.strip()
    match = re.fullmatch(r"```[^\n]*\n(.*?)\n```", text, re.S)
    return match.group(1).strip() if match else text


def check_code(text, task):
    code = strip_fence(text)
    tree = ast.parse(code)
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        return False, "Expected function definitions only"
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal)
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, "Code exceeds pure-function test scope"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, "Dunder names are outside test scope"
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.args.defaults or node.args.kw_defaults and any(node.args.kw_defaults)):
            return False, "Decorators and defaults are outside test scope"
    # Run pure generated functions in a bounded child with no I/O builtins.
    driver = r"""
import copy,json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
data=json.load(sys.stdin)
allowed={k:__builtins__.__dict__[k] for k in ['abs','all','any','bool','dict','enumerate','float','int','len','list','max','min','range','reversed','set','sorted','sum','tuple','zip']}
ns={'__builtins__':allowed}
exec(compile(data['code'],'<candidate>','exec'),ns)
if data['task']=='coding':
    f=ns['merge_intervals']
    cases=[([],[]),([[1,3],[2,6],[8,10],[10,12]],[[1,6],[8,12]]),([[-5,-1],[-2,0],[4,4]],[[-5,0],[4,4]]),([[5,7],[1,2],[2,5]],[[1,7]]),([[1,1],[1,1]],[[1,1]])]
    for value,expected in cases:
        original=copy.deepcopy(value)
        assert f(value)==expected and value==original
else:
    f=ns['first_true']
    for n in range(33):
        for split in range(n+1):
            assert f([False]*split+[True]*(n-split))==split
    class Probe:
        def __init__(self, n, split): self.n=n; self.split=split; self.reads=0
        def __len__(self): return self.n
        def __getitem__(self, i):
            self.reads+=1
            assert self.reads<=16, 'Binary search exceeded logarithmic probe budget'
            if i<0 or i>=self.n: raise IndexError(i)
            return i>=self.split
    for split in [0,1,1023,2048,4095,4096]:
        assert f(Probe(4096,split))==split
print('PASS')
"""
    proc = subprocess.run([sys.executable, "-I", "-c", driver], input=json.dumps(dict(code=code,task=task)), text=True, capture_output=True, timeout=4)
    return proc.returncode == 0, "Executable function checks" if proc.returncode == 0 else "Function tests failed: "+proc.stderr[-200:]


def stable_message(message):
    """Exclude server-assigned call IDs from reproducibility comparisons."""
    result = dict(message)
    if "tool_calls" in result:
        result["tool_calls"] = [{k: v for k, v in call.items() if k != "id"}
                                for call in result["tool_calls"]]
    return result


def validate(task, message, finish_reason="stop"):
    try:
        if finish_reason == "length":
            return False, "Truncated answer"
        key = task["id"]
        text = message.get("content") or ""
        if key in {"coding","debugging"}:
            return check_code(text,key)
        if key == "sql":
            sql=strip_fence(text)
            db=sqlite3.connect(":memory:")
            db.executescript("CREATE TABLE customers(id INTEGER PRIMARY KEY,name TEXT); CREATE TABLE orders(id INTEGER PRIMARY KEY,customer_id INTEGER,amount_cents INTEGER,status TEXT); INSERT INTO customers VALUES(1,'Ada'),(2,'Bea'),(3,'Cal'),(4,'Dee'),(5,'Eli'); INSERT INTO orders VALUES(1,1,100,'completed'),(2,1,200,'completed'),(3,1,999,'cancelled'),(4,2,300,'completed'),(5,4,700,'pending'),(6,5,50,'completed');")
            db.set_authorizer(lambda action,*args: sqlite3.SQLITE_OK if action in {sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION} else sqlite3.SQLITE_DENY)
            result=db.execute(sql).fetchall();db.close()
            return result==[(1,300),(2,300),(5,50),(3,0),(4,0)], "SQL result comparison"
        if key == "tool_use":
            calls=message.get("tool_calls",[])
            ok=len(calls)==1 and calls[0]["function"]["name"]=="lookup_inventory"
            args=json.loads(calls[0]["function"]["arguments"]) if ok else {}
            ok=ok and set(args)=={"skus","warehouse","include_reserved"} and sorted(args["skus"])==["AX-17","BZ-42"] and args["warehouse"]=="north" and args["include_reserved"] is True
            return ok,"Tool name and argument checks"
        if key == "creative":
            sentences=re.split(r"(?<=[.!?])\s+",text.strip())
            words=len(text.split())
            ok=len(sentences)==4 and all(s.startswith(start+" ") for s,start in zip(sentences,["Dawn","Inside","Meanwhile","Finally"]))
            ok=ok and 90<=words<=130 and all(term in text.lower() for term in ["rusty key","comet","locked observatory"]) and text.endswith("the door opened.") and not any(c in text for c in '\"“”')
            return ok,f"Formal writing constraints; {words} words; coherence requires review"
        obj=json.loads(strip_fence(text))
        if key in {"arithmetic","extraction","long_context"}:
            return obj==task["reference"],"Exact structured answer"
        if key == "summarization":
            expected=task["reference"]
            ok=set(obj)==set(expected) and all(obj[k].casefold()==expected[k].casefold() for k in ["owner","deadline"])
            ok=ok and sorted(x.casefold() for x in obj["blockers"])==sorted(expected["blockers"]) and 35<=len(obj["summary"].split())<=60
            cancelled = obj["cancelled"].strip().casefold()
            cancellation_ok = cancelled in {"marketing email", "the marketing email"}
            # The prompt does not define whether this field names the cancelled
            # item or describes release status. Accept the latter interpretation
            # only when the prose explicitly preserves both cancellation facts.
            prose = obj["summary"].casefold()
            if cancelled in {"no", "none", "not cancelled", "not canceled"}:
                cancellation_ok = bool(re.search(r"marketing email.*cancel", prose) and
                                       re.search(r"release.*(?:active|not cancelled|not canceled)", prose))
            return bool(ok and cancellation_ok),"Summary facts and length; prose fidelity requires review"
        if key == "planning":
            duration=dict(A=3,B=2,C=4,D=2,E=1);deps=dict(C=["A"],D=["B"],E=["C","D"])
            if set(obj)!=set(duration):return False,"Missing tasks"
            for name,x in obj.items():
                if set(x)!={"worker","start"} or type(x['worker']) is not int or x['worker'] not in (0,1) or type(x['start']) is not int or x['start']<0:return False,"Invalid schedule fields"
                if x['start']+duration[name]>8:return False,"Deadline missed"
                if any(obj[d]['start']+duration[d]>x['start'] for d in deps.get(name,[])):return False,"Dependency violated"
            names=list(obj)
            for i,a in enumerate(names):
                for b in names[i+1:]:
                    if obj[a]['worker']==obj[b]['worker'] and max(obj[a]['start'],obj[b]['start'])<min(obj[a]['start']+duration[a],obj[b]['start']+duration[b]):return False,"Worker overlap"
            return True,"Schedule feasibility"
        return False,"Unknown workload"
    except Exception as exc:
        return False,type(exc).__name__+": "+str(exc)[:160]


def main():
    tasks=workloads()
    if sys.argv[1:]==["--self-check"]:
        for task in tasks:
            ref=task['reference'];message=ref if task['id']=='tool_use' else {'content':ref if isinstance(ref,str) else json.dumps(ref)}
            ok,detail=validate(task,message)
            print(task['id'],ok,detail)
            assert ok,task['id']
        return
    base,out=sys.argv[1:];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    public=[{k:v for k,v in t.items() if k!='reference'} for t in tasks]
    (out/'workloads.json').write_text(json.dumps(public,indent=2)+'\n')
    def request(body):
        req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        start=time.time_ns()
        with urllib.request.urlopen(req,timeout=300) as response:data=json.load(response)
        return data,start,time.time_ns()
    for index,task in enumerate(tasks):
        body=dict(model='qwen3.8-27b-uncensored',messages=[dict(role='system',content=SYSTEM),dict(role='user',content=task['prompt'])],temperature=1.0,max_tokens=task['max_tokens'],cache_prompt=True,stream=False)
        if 'tools' in task:body.update(tools=task['tools'],tool_choice='auto',parallel_tool_calls=False)
        warm,begin,end=request(dict(body,max_tokens=1,seed=SEEDS[0],cache_prompt=False))
        (out/(task['id']+'-warmup.json')).write_text(json.dumps(warm,indent=2)+'\n')
        # Rotate seed order by task while holding it fixed across all variants.
        seeds=SEEDS[index%3:]+SEEDS[:index%3]
        for seed in seeds:
            payload=dict(body,seed=seed);data,begin,end=request(payload)
            choice=data['choices'][0];message=choice['message'];ok,detail=validate(task,message,choice.get('finish_reason'))
            label=task['id']+'-'+str(seed)
            (out/(label+'.json')).write_text(json.dumps(data,indent=2)+'\n')
            row=dict(task=task['id'],category=task['category'],seed=seed,start_ns=begin,end_ns=end,wall_ms=(end-begin)/1e6,finish_reason=choice.get('finish_reason'),quality_pass=ok,quality_detail=detail,input_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),output_sha256=hashlib.sha256(json.dumps(stable_message(message),sort_keys=True).encode()).hexdigest(),**data.get('timings',{}))
            with (out/'evaluation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
