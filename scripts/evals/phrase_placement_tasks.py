"""Hard two-turn fixtures for the Swift phrase-placement comparison."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import long_context_project as stockroom

ROOT = Path(__file__).resolve().parents[2]
STOCK_SEED = ROOT / '.marathon/diagnostics/pi-marathon-long-20260921/stockroom-marathon-2/checkpoint'
BOUNDARY = '\nUse only the Python standard library. Work only in this workspace, no network or delegation. Add and run tests. Do not commit.'
PROMPTS = {
 'ledger': [
  'Extend this existing inventory project. Its binding earlier contracts follow:\n' + '\n'.join(stockroom.PROMPTS[:3]) + '\nNEW FEATURE:\n' + stockroom.PROMPTS[3] + BOUNDARY,
  stockroom.PROMPTS[4] + BOUNDARY,
 ],
 'scheduler': [
  '''Implement solve(jobs) in scheduler.py as an EXACT optimizer, not a heuristic. Up to 8 jobs, one machine, nonpreemptive integer-time execution starting at time 0. Each job is a dict with id (nonempty string), duration (positive int), release (nonnegative int), deadline (nonnegative int), profit (any int, including negative), deps (list of job IDs). Booleans are not integers for validation. You may omit any job, but a selected job requires ALL its dependencies selected and completed before it starts. Jobs start no earlier than release and must finish <= deadline. Idle time is allowed. Maximize total profit, then minimize final completion time, then choose lexicographically smallest tuple of job IDs in execution order. Empty selection is allowed, with profit 0 and finish 0. Return exactly {"profit": int, "finish": int, "order": list[str]}. Reject duplicate IDs, unknown dependency IDs, any dependency cycle (even in jobs you would omit), and invalid fields with ValueError. Do not mutate input. Dependencies can have negative profit; globally profitable chains must still be considered. Repeated dependencies can be treated as one. Add regression tests with counterexamples to greedy selection.''' + BOUNDARY,
  '''Extend solve(jobs, blackouts=()) while preserving ALL earlier contracts. blackouts is a sequence of pairs (start,end) of nonnegative integers with start < end; reject booleans and invalid intervals with ValueError. Intervals may overlap, nest, touch, or arrive unsorted. No execution can overlap a blackout; jobs are nonpreemptive, so a job that cannot fit before a blackout must wait until after it, and that wait may encounter more blackouts. Touching endpoints is allowed. Deadlines and releases still apply. The exact profit / finish / lexicographic objective is unchanged. Validate malformed blackouts even when jobs is empty. Preserve both inputs. Add tests covering jobs fitting exactly in gaps and dependencies made infeasible by blackout-induced delays.''' + BOUNDARY,
 ]
}

def prepare(workspace,case):
 if case=='ledger':shutil.copytree(STOCK_SEED,workspace,dirs_exist_ok=True)
 else:(workspace/'scheduler.py').write_text('def solve(jobs):\n    raise NotImplementedError("Exact scheduling not implemented")\n')

# Independent exhaustive oracle and test generation, never put in agent workspaces.
SCHEDULER_GRADER = r'''
import copy,json,random
from scheduler import solve
stage=STAGE
checks=0;failures=[]
def record(label,fn):
 global checks
 checks+=1
 try:fn()
 except Exception as e:failures.append([label,type(e).__name__+': '+str(e)])
def job(id,duration=2,release=0,deadline=12,profit=3,deps=()):
 return dict(id=id,duration=duration,release=release,deadline=deadline,profit=profit,deps=list(deps))
def oracle(jobs,windows):
 best=(0,0,())
 def walk(order,t,profit):
  nonlocal best
  candidate=(-profit,t,tuple(order))
  if candidate<best:best=candidate
  for j in jobs:
   if j['id'] in order or not set(j['deps'])<=set(order):continue
   start=max(t,j['release'])
   while True:
    hits=[b for a,b in windows if start<b and start+j['duration']>a]
    if not hits:break
    start=max(hits)
   end=start+j['duration']
   if end<=j['deadline']:walk(order+[j['id']],end,profit+j['profit'])
 walk([],0,0)
 return dict(profit=-best[0],finish=best[1],order=list(best[2]))
def valid(label,jobs,windows=()):
 def check():
  before=copy.deepcopy((jobs,windows));expected=oracle(jobs,windows)
  got=solve(jobs,blackouts=windows) if stage else solve(jobs)
  assert got==expected,(got,expected)
  assert (jobs,windows)==before,'mutated inputs'
 record(label,check)
def invalid(label,jobs,windows=()):
 def check():
  before=copy.deepcopy((jobs,windows))
  try:solve(jobs,blackouts=windows) if stage else solve(jobs)
  except ValueError:pass
  else:raise AssertionError('expected ValueError')
  assert (jobs,windows)==before,'mutated inputs'
 record(label,check)
valid('empty',[])
valid('negative prerequisite',[job('root',profit=-6),job('reward',profit=20,deps=['root']),job('greedy',duration=9,profit=12)])
valid('release and deadline',[job('b',release=3,deadline=6),job('a',duration=3,deadline=3),job('c',duration=6,deadline=6,profit=8)])
valid('lex tie',[job('b'),job('a'),job('zero',profit=0)])
valid('repeated dependency',[job('a'),job('b',deps=['a','a'])])
rng=random.Random(89123)
for ncase in range(32):
 jobs=[]
 for i in range(7):
  jobs.append(job(chr(97+i),duration=rng.randint(1,4),release=rng.randint(0,5),deadline=rng.randint(4,18),profit=rng.randint(-4,12),deps=[chr(97+k) for k in range(i) if rng.random()<.18]))
 rng.shuffle(jobs)
 windows=[(2,4),(8,10)] if stage and ncase%2 else []
 valid('oracle-'+str(ncase),jobs,windows)
invalid('duplicate',[job('a'),job('a')]);invalid('unknown',[job('a',deps=['missing'])]);invalid('cycle',[job('a',deps=['b']),job('b',deps=['a'])])
for field,value in [('id',''),('duration',0),('duration',True),('release',-1),('deadline',False),('profit',1.5)]:
 j=job('a');j[field]=value;invalid('invalid-'+field+repr(value),[j])
if stage:
 valid('gap exact',[job('a',duration=3,deadline=3),job('b',duration=2,deadline=7,deps=['a'])],[(3,5)])
 valid('merged windows',[job('a',duration=3,deadline=20),job('b',duration=2,deadline=19,deps=['a'])],[(6,9),(2,5),(4,7),(9,11)])
 valid('delay breaks dependency',[job('a',deadline=5),job('b',deadline=7,profit=50,deps=['a'])],[(1,5)])
 for windows in [[(1,1)],[(-1,3)],[(False,3)],[(2,1)],[(1,2.5)]]:invalid('bad-window-'+repr(windows),[],windows)
print(json.dumps(dict(checks=checks,passed_checks=checks-len(failures),failures=failures)))
'''

def grade(workspace,case,turn):
 if case=='ledger':return stockroom.grade(workspace,turn+3)
 code=SCHEDULER_GRADER.replace('stage=STAGE',f'stage={turn}')
 try:
  r=subprocess.run([sys.executable,'-c',code],cwd=workspace,capture_output=True,text=True,timeout=60)
  return json.loads(r.stdout.strip().splitlines()[-1])
 except (ValueError,IndexError,subprocess.TimeoutExpired) as e:
  return dict(checks=46+8*turn,passed_checks=0,failures=[['grader',str(e)]])
