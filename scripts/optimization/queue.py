#!/usr/bin/env python3
"""Cooperative quick-win queue and exclusive GPU-3 benchmark gate."""
import argparse, contextlib, fcntl, hashlib, json, os, re, shutil, signal
import sqlite3, subprocess, sys, time, urllib.request, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.marathon/optimization-queue'
CONFIG = Path('/home/deforest/Documents/DEV/gpu-control/llama-swap/config.yaml')
BASE = ROOT / '.marathon/diagnostics/q8-staging-probe-20260920/source-compact'
IMAGE = 'sha256:8af4fa77e493f6765b7c66d4f6cbfb673e0add0919b470c11526e08d54365e04'
TASKS = [
 ('iq4-loads', 'Inspect dominant IQ4_XS J8 weight-load/dequantization costs. Seek ONE new mechanism supported by existing counters; exclude previously rejected grid, tile, L1-prefetch and activation-double-buffer changes.'),
 ('q8-residual', 'Inspect remaining stalls in the DEPLOYED compact Q8 attention pipeline. Seek ONE change preserving the launch partition and Q8 numerics; do not repeat the rejected two-buffer design.'),
 ('copy-gather', 'Inspect target-side copy/gather work in the existing runtime profile. Identify ONE avoidable transfer or launch with measurable cost; first prove it is not already fused.'),
 ('recurrent', 'Inspect recurrent/gated-delta decode work. Seek ONE evidence-backed launch or memory-traffic saving; do not repeat rejected prefill QK staging.'),
]
def emit(x): print(json.dumps(x, indent=2), flush=True)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def db():
 STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
 c = sqlite3.connect(STATE/'queue.sqlite', timeout=30)
 c.row_factory = sqlite3.Row
 c.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, brief TEXT, status TEXT DEFAULT "pending", owner TEXT, workspace TEXT, result TEXT)')
 c.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, task TEXT, status TEXT, details TEXT)')
 c.commit(); return c
@contextlib.contextmanager
def locked(path, timeout=0):
 path.parent.mkdir(parents=True, exist_ok=True)
 with path.open('a+') as f:
  end = time.monotonic()+timeout
  while True:
   try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); break
   except BlockingIOError:
    if time.monotonic() >= end: raise RuntimeError('Resource busy; leave it alone and retry later.')
    time.sleep(.5)
  try: yield f
  finally: fcntl.flock(f, fcntl.LOCK_UN)
def owned(c, task, owner):
 r = c.execute('SELECT * FROM tasks WHERE id=?', (task,)).fetchone()
 if not r or r['owner'] != owner or r['status'] != 'claimed': raise ValueError('Task is not claimed by this owner')
 return r
def initialize():
 c = db()
 with c:
  for task, brief in TASKS: c.execute('INSERT OR IGNORE INTO tasks(id,brief) VALUES (?,?)',(task,brief))
 # Do not silently reset the reference when workers are already running.
 p = STATE/'baseline.json'
 if not p.exists():
  data={'image':IMAGE,'config_sha256':digest(CONFIG),'source_header_sha256':digest(BASE/'fattn-mma-f16.cuh')}
  p.write_text(json.dumps(data,indent=2))
 emit({'state':str(STATE),'tasks':len(TASKS),'baseline':json.loads(p.read_text())})
def claim():
 baseline=json.loads((STATE/'baseline.json').read_text())
 if digest(BASE/'fattn-mma-f16.cuh')!=baseline['source_header_sha256']:raise RuntimeError('Qualified source changed; review before claiming')
 c=db(); c.execute('BEGIN IMMEDIATE')
 r=c.execute('SELECT * FROM tasks WHERE status="pending" ORDER BY rowid LIMIT 1').fetchone()
 if not r: c.rollback(); emit({'status':'no_pending_tasks'}); return
 owner=uuid.uuid4().hex; work=STATE/'workspaces'/r['id']
 try:
  work.mkdir(parents=True,exist_ok=False)
  shutil.copytree(BASE,work/'source')
  c.execute('UPDATE tasks SET status="claimed",owner=?,workspace=? WHERE id=?',(owner,str(work),r['id']))
  c.commit()
 except BaseException: c.rollback(); raise
 info={'task':r['id'],'owner':owner,'workspace':str(work),'brief':r['brief']}
 (work/'claim.json').write_text(json.dumps(info,indent=2));emit(info)
def finish(a):
 evidence=Path(a.evidence).resolve()
 if not evidence.is_file(): raise ValueError('Write the result/evidence file first')
 c=db(); c.execute('BEGIN IMMEDIATE');r=owned(c,a.task,a.owner)
 if not evidence.is_relative_to(Path(r['workspace'])): raise ValueError('Evidence must be in this task workspace')
 with c:c.execute('UPDATE tasks SET status=?,result=? WHERE id=?',(a.outcome,json.dumps({'summary':a.summary,'evidence':str(evidence)}),a.task))
 emit({'task':a.task,'status':a.outcome})
def broker_running():
 # Read one named key only; never emit it or child environment values.
 lines=Path('/home/deforest/Documents/DEV/qwen-inference/.env').read_text().splitlines()
 key=next(l.split('=',1)[1].strip().strip('\"\'') for l in lines if l.startswith('HERMES_API_KEY='))
 r=urllib.request.Request('http://127.0.0.1:9292/running',headers={'Authorization':'Bearer '+key})
 with urllib.request.urlopen(r,timeout=3) as f:return json.load(f)['running']
def conflicts():
 import yaml
 config=yaml.safe_load(CONFIG.read_text()); result=set()
 for name,m in config['models'].items():
  cmd=m.get('cmd','').replace('${gpu}',str(m.get('macros',{}).get('gpu','')))
  groups=re.findall(r'(?:device=|CUDA_VISIBLE_DEVICES=)([0-9,]+)',cmd)
  if any('3' in x.split(',') for x in groups):result.add(name)
 return result
def run(a):
 c=db();r=owned(c,a.task,a.owner);work=Path(r['workspace']);command=a.command
 if command and command[0]=='--':command=command[1:]
 if not command:raise ValueError('Specify a bounded benchmark command after --')
 baseline=json.loads((STATE/'baseline.json').read_text())
 runid=uuid.uuid4().hex[:12]; container='marathon-opt-'+runid
 details={'command':command,'container':container,'timeout_s':a.timeout,'log':str(work/(runid+'.log'))}
 with c:c.execute('INSERT INTO runs VALUES (?,?,?,?)',(runid,a.task,'queued',json.dumps(details)))
 def check():
  if digest(CONFIG)!=baseline['config_sha256']:raise RuntimeError('Production config changed; stop and review baseline')
  if any(x['model'] in conflicts() for x in broker_running()):raise RuntimeError('Registered worker needs GPU 3; benchmark refused/stopped')
 model='marathon-qwen3.8-27b-uncensored-3'
 pool=Path('/run/user/1000/marathon/backend-pools/llama-swap-qwen3.8-uncensored-pool')
 poollock=pool/(model+'-'+hashlib.sha256(model.encode()).hexdigest()[:12]+'.lock')
 child=None;started=None
 def interrupted(*_):raise RuntimeError('Benchmark interrupted')
 old={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
 try:
  with locked(STATE/'benchmark.lock',a.wait), locked(poollock):
   owned(c,a.task,a.owner);check()
   used=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
   if used>100:raise RuntimeError(f'GPU 3 occupied ({used} MiB); do not evict it')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUDA_DEVICE_ORDER='PCI_BUS_ID',OPT_GPU='3',OPT_CONTAINER=container,OPT_IMAGE=baseline['image'],OPT_WORKSPACE=str(work))
   with c:c.execute('UPDATE runs SET status="running" WHERE id=?',(runid,))
   started=time.monotonic()
   try:
    with Path(details['log']).open('w') as log:
     child=subprocess.Popen(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     while child.poll() is None:
      check()
      if time.monotonic()-started>a.timeout:raise TimeoutError('Benchmark time limit exceeded')
      time.sleep(1)
     if child.returncode:raise RuntimeError(f'Benchmark exited {child.returncode}; inspect log')
     check()
   finally:
    # Hold both locks until owned processes and the specifically named container exit.
    if child and child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if child:
     try:os.killpg(child.pid,signal.SIGTERM)
     except ProcessLookupError:pass
    # Contract: benchmark scripts MUST use this name for their only GPU container.
    subprocess.run(['docker','stop','--timeout','10',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
   status='completed'
 except Exception as e:
  status='failed';details['error']=str(e)
 finally:
  for s,h in old.items():signal.signal(s,h)
 details['elapsed_s']=None if started is None else time.monotonic()-started
 with c:c.execute('UPDATE runs SET status=?,details=? WHERE id=?',(status,json.dumps(details),runid))
 emit({'run':runid,'status':status,**details})
 if status!='completed':raise SystemExit(1)
def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
 s.add_parser('init');s.add_parser('claim');s.add_parser('status')
 f=s.add_parser('finish');f.add_argument('task');f.add_argument('--owner',required=True);f.add_argument('--outcome',choices=['rejected','promising','blocked'],required=True);f.add_argument('--summary',required=True);f.add_argument('--evidence',required=True)
 b=s.add_parser('bench');b.add_argument('task');b.add_argument('--owner',required=True);b.add_argument('--timeout',type=int,default=240);b.add_argument('--wait',type=int,default=600);# Command is split at -- before argparse, so options can follow task ID.
 argv=sys.argv[1:]; command=[]
 if argv and argv[0]=='bench' and '--' in argv:
  split=argv.index('--'); command=argv[split+1:]; argv=argv[:split]
 a=p.parse_args(argv); a.command=command
 if a.action=='init':initialize()
 elif a.action=='claim':claim()
 elif a.action=='finish':finish(a)
 elif a.action=='status':
  c=db();emit({'tasks':[dict(r) for r in c.execute('SELECT * FROM tasks ORDER BY rowid')],'runs':[dict(r) for r in c.execute('SELECT * FROM runs ORDER BY rowid')]})
 else:
  if not 1<=a.timeout<=600 or not 0<=a.wait<=1200:p.error('timeout must be 1..600; wait must be 0..1200 seconds')
  run(a)
if __name__=='__main__':main()
