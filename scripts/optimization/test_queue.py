"""CPU-only coordination checks. Never touches inference or the real queue."""
import contextlib, importlib.util, io, json, multiprocessing, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('optqueue',Path(__file__).with_name('queue.py'))
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  root=Path(self.tmp.name); self.state=patch.object(q,'STATE',root/'state');self.state.start();self.addCleanup(self.state.stop)
  src=root/'source';src.mkdir();(src/'fattn-mma-f16.cuh').write_text('baseline')
  cfg=root/'config';cfg.write_text('baseline config')
  for name,value in [('BASE',src),('CONFIG',cfg)]:
   p=patch.object(q,name,value);p.start();self.addCleanup(p.stop)
  with contextlib.redirect_stdout(io.StringIO()):q.initialize()
 def test_concurrent_claims_are_distinct_and_isolated(self):
  ctx=multiprocessing.get_context('fork');workers=[ctx.Process(target=q.claim) for _ in range(2)]
  for p in workers:p.start()
  for p in workers:p.join(10);self.assertEqual(p.exitcode,0)
  rows=q.db().execute('SELECT * FROM tasks WHERE status="claimed"').fetchall()
  self.assertEqual(len(rows),2);self.assertNotEqual(rows[0]['owner'],rows[1]['owner'])
  one=Path(rows[0]['workspace'])/'source/fattn-mma-f16.cuh';one.write_text('candidate')
  self.assertEqual((Path(rows[1]['workspace'])/'source/fattn-mma-f16.cuh').read_text(),'baseline')
 def test_wrong_owner_cannot_finish(self):
  with contextlib.redirect_stdout(io.StringIO()):q.claim()
  r=q.db().execute('SELECT * FROM tasks WHERE status="claimed"').fetchone()
  e=Path(r['workspace'])/'result.json';e.write_text('{}')
  with self.assertRaises(ValueError):q.finish(SimpleNamespace(task=r['id'],owner='wrong',evidence=str(e),outcome='rejected',summary='test'))
  self.assertEqual(q.db().execute('SELECT status FROM tasks WHERE id=?',(r['id'],)).fetchone()[0],'claimed')
 def test_registered_conflict_never_launches_benchmark(self):
  with contextlib.redirect_stdout(io.StringIO()):q.claim()
  r=q.db().execute('SELECT * FROM tasks WHERE status="claimed"').fetchone()
  a=SimpleNamespace(task=r['id'],owner=r['owner'],command=['false'],timeout=1,wait=0)
  with patch.object(q,'locked',lambda *a:contextlib.nullcontext()),patch.object(q,'conflicts',return_value={'busy'}),patch.object(q,'broker_running',return_value=[{'model':'busy'}]),patch.object(q.subprocess,'Popen') as launch:
   with self.assertRaises(SystemExit):q.run(a)
   launch.assert_not_called()
 def test_timeout_cleans_owned_process_and_container(self):
  with contextlib.redirect_stdout(io.StringIO()):q.claim()
  r=q.db().execute('SELECT * FROM tasks WHERE status="claimed"').fetchone()
  a=SimpleNamespace(task=r['id'],owner=r['owner'],command=['probe'],timeout=0,wait=0)
  with patch.object(q,'locked',lambda *a:contextlib.nullcontext()),patch.object(q,'conflicts',return_value=set()),patch.object(q,'broker_running',return_value=[]),patch.object(q.subprocess,'check_output',return_value='15'),patch.object(q.subprocess,'Popen') as launch,patch.object(q.subprocess,'run') as cleanup,patch.object(q.os,'killpg') as kill:
   launch.return_value.pid=123;launch.return_value.poll.return_value=None
   with self.assertRaises(SystemExit):q.run(a)
   self.assertTrue(kill.called);self.assertEqual(cleanup.call_args.args[0][:3],['docker','stop','--timeout'])
   self.assertTrue(cleanup.call_args.args[0][-1].startswith('marathon-opt-'))
 def test_bench_arguments_after_task(self):
  with patch.object(q.sys,'argv',['queue.py','bench','iq4-loads','--owner','abc','--timeout','20','--','python3','probe.py']),patch.object(q,'run') as run:
   q.main();a=run.call_args.args[0];self.assertEqual(a.owner,'abc');self.assertEqual(a.command,['python3','probe.py']);self.assertEqual(a.timeout,20)
 def test_exclusive_lock(self):
  with q.locked(q.STATE/'lock'):
   with self.assertRaises(RuntimeError):
    with q.locked(q.STATE/'lock'):pass
if __name__=='__main__':unittest.main()
