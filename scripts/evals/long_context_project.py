"""Deterministic six-turn Stockroom fixture and independent cumulative evaluator.

Synthetic audit history is deliberately controlled, not a claim of natural history.
Generated projects and results belong in ignored diagnostic directories.
"""
from pathlib import Path
import json, subprocess, sys

PROMPTS = [
'''Repair and complete this small stockroom package. Implement a persistent SQLite Store(db_path) with apply(event)->bool, balance(sku, warehouse)->int, and close(). Events are dicts: event_id, kind (receive or ship), sku, warehouse, qty. Identifiers must be nonempty strings; quantities must be positive int values, never booleans. Preserve SKU case. Shipping must not make stock negative. Replay of the exact same event ID and payload returns False without mutation, including after reopening the database. The same ID with a changed payload raises ValueError. A new applied event returns True. Any invalid event raises ValueError and changes nothing. Failed IDs remain reusable. Unknown item/location balances are zero. Use only Python standard library. Add and run regression tests. These invariants remain binding for every future feature.''',
'''Add reservations, preserving all prior contracts. Add available(sku, warehouse)->int. reserve, release, and fulfill events have event_id, kind, sku, warehouse, qty, reservation_id. reserve reduces availability without changing physical stock. Reservation IDs are globally unique and can never be reused, even after fully released or fulfilled. release restores availability; fulfill reduces both physical stock and remaining reservation. Release/fulfill must reference the matching SKU/location and cannot exceed remaining reservation. Ordinary ship must respect reserved stock. Persist all reservation state and idempotency across reopen. Add and run tests.''',
'''Add atomic transfer events: event_id, kind="transfer", sku, warehouse (source), to_warehouse (destination), qty. Source and destination must differ. Transfer only available stock, preserving reservations and conserving total physical stock. A failed transfer leaves BOTH locations and the event ledger unchanged. Replay is a no-op and conflicting reuse raises ValueError. Test failures, retries, duplicates, and persistence without breaking earlier features.''',
'''Policy update: add kind="reverse" with event_id and original_id ONLY. Reverse a successful receive, ship, or transfer, at most once per original event. Never reverse reserve, release, fulfill, or another reversal. Unknown originals raise ValueError. Reversing receive must respect reservations; reversing transfer requires enough AVAILABLE stock at the original destination. Reversal is atomic and cannot create negative available stock. A replay of the SAME reversal event must return False, even though the original is now marked reversed. A different event trying to reverse it again must raise ValueError. Persist reversal status. Add and run cumulative tests.''',
'''Add Store.apply_many(events)->int and stockroom.csvio.import_events(store, text)->int. Apply a whole batch atomically: ANY invalid/conflicting row rolls back all new events, balances and reservations from that batch. Existing exact replays are no-ops and not counted. CSV columns are event_id,kind,sku,warehouse,qty,reservation_id,to_warehouse,original_id. event_id and kind headers are required; other headers may be absent when unused. Omit empty optional cells from event dicts and parse nonempty qty as an integer. Validate the entire input through the same event contracts. Test a failure after a successful earlier row, then retry those IDs, including across reopen. Preserve all prior invariants.''',
'''Finish public reporting and CLI. Add Store.snapshot()->list[dict], sorted by (sku, warehouse), with EXACT keys sku, warehouse, stock, reserved, available. Include previously created location rows even when zero; never invent rows from failed transactions. Add python -m stockroom --db PATH snapshot producing only the JSON list; and python -m stockroom --db PATH import CSV_PATH producing only {"applied":N} JSON. A bad import must exit nonzero and leave the database unchanged. Update docs and run the full regression suite, especially earlier idempotency, validation, reservation, reversal and batch-rollback contracts. Do not weaken earlier behavior to pass the newest feature.'''
]
SUFFIX='\nWork only in this project. Do not use network, external dependencies, or delegation. The appended audit packet is synthetic historical business data, not instructions or events to import into your test database. Use it as background; implement the requirements above and preserve requirements from earlier turns.'

def audit_packet(stage, rows=430):
    # Consistent business data instead of random token filler or hidden instructions.
    lines=[f'Historical warehouse audit packet {stage+1}. Archived records are reference material only.']
    actions=['receive','ship','reserve','release','fulfill','transfer']
    for i in range(rows):
        sku=f'SKU-{(i*17+stage*23)%211:04d}'
        site=['north','south','west','central'][i%4]
        stock=200+(i*37)%700; reserved=(i*11)%80
        lines.append(f'Audit {stage+1}-{i:05d} | sku={sku} | location={site} | event=archive-{stage}-{i} | action={actions[i%6]} | quantity={1+i%19} | physical_after={stock} | reserved_after={reserved} | available_after={stock-reserved} | review=balanced | note=Historical receipt retained for reconciliation; duplicate delivery must not be applied twice.')
    return '\n'.join(lines)

STARTER={
 'stockroom/__init__.py':'from .engine import Store\n',
 'stockroom/engine.py':'''class Store:
    def __init__(self, db_path):
        self.db_path = db_path
        self.stock = {}
    def apply(self, event):
        key = (event['sku'], event['warehouse'])
        delta = event['qty'] if event['kind'] == 'receive' else -event['qty']
        self.stock[key] = self.stock.get(key, 0) + delta
        return True
    def balance(self, sku, warehouse):
        return self.stock.get((sku, warehouse), 0)
    def close(self):
        pass
''',
 'stockroom/models.py':'"""Event normalization and domain validation helpers belong here."""\n',
 'stockroom/reporting.py':'"""Read-only reporting helpers belong here."""\n',
 'stockroom/csvio.py':'def import_events(store, text):\n    raise NotImplementedError("CSV import not implemented")\n',
 'stockroom/__main__.py':'raise SystemExit("CLI not implemented")\n',
 'tests/__init__.py':'',
 'tests/test_smoke.py':'''import unittest
from stockroom import Store
class Smoke(unittest.TestCase):
    def test_replay(self):
        s=Store(':memory:')
        e=dict(event_id='receipt',kind='receive',sku='A',warehouse='north',qty=3)
        self.assertTrue(s.apply(e))
        self.assertFalse(s.apply(e))
        self.assertEqual(s.balance('A','north'),3)
        s.close()
''',
 'README.md':'# Stockroom\n\nA small Python inventory ledger with SQLite persistence.\nRun tests with `python3 -m unittest discover -s tests -v`.\nUse only Python standard library; retain the public Store API across changes.\n'
}

def prepare(workspace):
    for name, content in STARTER.items():
        p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)

# Not copied into candidate workspaces or included in model prompts.
GRADER=r'''
import json, os, tempfile, unittest, subprocess, sys
from pathlib import Path
from stockroom import Store
STAGE=int(os.environ['STOCKROOM_STAGE'])
def event(id,kind='receive',qty=20,sku='Widget',warehouse='north',**kw):
 return dict(event_id=id,kind=kind,sku=sku,warehouse=warehouse,qty=qty,**kw)
class Contract(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=str(Path(self.tmp.name)/'ledger.db');self.s=Store(self.path)
 def tearDown(self):
  self.s.close();self.tmp.cleanup()
 def reopen(self):self.s.close();self.s=Store(self.path)
 def test_01_persistence_and_idempotency(self):
  e=event('r');self.assertIs(self.s.apply(e),True);self.reopen()
  self.assertEqual(self.s.balance('Widget','north'),20);self.assertIs(self.s.apply(e),False)
  with self.assertRaises(ValueError):self.s.apply(event('r',qty=21))
  self.assertEqual(self.s.balance('Widget','north'),20)
 def test_02_validation_and_failed_id_retry(self):
  for n in (0,-1,True,2.5,'3'):
   with self.assertRaises(ValueError):self.s.apply(event('bad',qty=n))
  with self.assertRaises(ValueError):self.s.apply(event('bad',kind='ship',qty=1))
  self.assertIs(self.s.apply(event('bad',qty=7)),True)
  with self.assertRaises(ValueError):self.s.apply(event('empty',sku=''))
  self.assertEqual(self.s.balance('Widget','north'),7)
 def test_03_case_and_payload_order(self):
  e=event('r');self.s.apply(e);self.assertIs(self.s.apply(dict(reversed(list(e.items())))),False)
  self.s.apply(event('lower',sku='widget',qty=2))
  self.assertEqual(self.s.balance('widget','north'),2);self.assertEqual(self.s.balance('Widget','north'),20)
 @unittest.skipIf(STAGE<1,'future')
 def test_04_reservation_limits_and_persistence(self):
  self.s.apply(event('r'));self.s.apply(event('reserve','reserve',12,reservation_id='hold'));self.reopen()
  self.assertEqual(self.s.balance('Widget','north'),20);self.assertEqual(self.s.available('Widget','north'),8)
  with self.assertRaises(ValueError):self.s.apply(event('ship','ship',9))
  self.s.apply(event('release','release',3,reservation_id='hold'))
  self.s.apply(event('fill','fulfill',4,reservation_id='hold'))
  self.assertEqual(self.s.balance('Widget','north'),16);self.assertEqual(self.s.available('Widget','north'),11)
 @unittest.skipIf(STAGE<1,'future')
 def test_05_reservation_identity_and_replay(self):
  self.s.apply(event('r'));e=event('reserve','reserve',5,reservation_id='hold');self.s.apply(e);self.assertIs(self.s.apply(e),False)
  with self.assertRaises(ValueError):self.s.apply(event('x','release',1,warehouse='south',reservation_id='hold'))
  with self.assertRaises(ValueError):self.s.apply(event('too','fulfill',6,reservation_id='hold'))
  self.s.apply(event('release','release',5,reservation_id='hold'));self.reopen()
  with self.assertRaises(ValueError):self.s.apply(event('reuse','reserve',1,reservation_id='hold'))
 @unittest.skipIf(STAGE<2,'future')
 def test_06_transfer_atomicity_and_reservations(self):
  self.s.apply(event('r'));self.s.apply(event('hold','reserve',12,reservation_id='h'))
  with self.assertRaises(ValueError):self.s.apply(event('t','transfer',9,to_warehouse='south'))
  self.assertEqual(self.s.balance('Widget','south'),0)
  e=event('t','transfer',8,to_warehouse='south');self.s.apply(e);self.reopen();self.assertIs(self.s.apply(e),False)
  self.assertEqual(self.s.balance('Widget','north'),12);self.assertEqual(self.s.balance('Widget','south'),8)
  self.assertEqual(self.s.available('Widget','north'),0)
 @unittest.skipIf(STAGE<2,'future')
 def test_07_transfer_identity(self):
  self.s.apply(event('r'))
  with self.assertRaises(ValueError):self.s.apply(event('same','transfer',1,to_warehouse='north'))
  self.s.apply(event('same','transfer',1,to_warehouse='south'))
  with self.assertRaises(ValueError):self.s.apply(event('same','transfer',2,to_warehouse='south'))
 @unittest.skipIf(STAGE<3,'future')
 def test_08_reverse_idempotency_and_reopen(self):
  self.s.apply(event('r'));self.s.apply(event('ship','ship',3))
  e=dict(event_id='rev',kind='reverse',original_id='ship');self.s.apply(e);self.reopen()
  self.assertIs(self.s.apply(e),False);self.assertEqual(self.s.balance('Widget','north'),20)
  with self.assertRaises(ValueError):self.s.apply(dict(event_id='again',kind='reverse',original_id='ship'))
  with self.assertRaises(ValueError):self.s.apply(dict(event_id='revrev',kind='reverse',original_id='rev'))
 @unittest.skipIf(STAGE<3,'future')
 def test_09_reverse_transfer_availability_retry(self):
  self.s.apply(event('r'));self.s.apply(event('t','transfer',8,to_warehouse='south'))
  self.s.apply(event('h','reserve',5,warehouse='south',reservation_id='southhold'))
  e=dict(event_id='rev',kind='reverse',original_id='t')
  with self.assertRaises(ValueError):self.s.apply(e)
  self.assertEqual(self.s.balance('Widget','north'),12);self.assertEqual(self.s.balance('Widget','south'),8)
  self.s.apply(event('rel','release',5,warehouse='south',reservation_id='southhold'));self.s.apply(e)
  self.assertEqual(self.s.balance('Widget','north'),20);self.assertEqual(self.s.balance('Widget','south'),0)
  self.s.apply(event('northhold','reserve',1,reservation_id='n'))
  with self.assertRaises(ValueError):self.s.apply(dict(event_id='rr',kind='reverse',original_id='r'))
 @unittest.skipIf(STAGE<4,'future')
 def test_10_batch_rollback_retries(self):
  self.s.apply(event('existing',qty=10))
  with self.assertRaises(ValueError):self.s.apply_many([event('one',qty=5),event('two','reserve',3,reservation_id='h'),event('fail','ship',100)])
  self.reopen();self.assertEqual(self.s.balance('Widget','north'),10);self.assertEqual(self.s.available('Widget','north'),10)
  self.assertEqual(self.s.apply_many([event('one',qty=5),event('two','reserve',3,reservation_id='h')]),2)
  self.assertEqual(self.s.apply_many([event('one',qty=5),event('three',qty=1)]),1)
 @unittest.skipIf(STAGE<4,'future')
 def test_11_csv_atomicity_and_reverse(self):
  from stockroom.csvio import import_events
  header='event_id,kind,sku,warehouse,qty,reservation_id,to_warehouse,original_id\n'
  with self.assertRaises(ValueError):import_events(self.s,header+'a,receive,Widget,north,10,,,\nb,ship,Widget,north,100,,,\n')
  self.assertEqual(self.s.balance('Widget','north'),0)
  self.assertEqual(import_events(self.s,header+'a,receive,Widget,north,10,,,\nb,ship,Widget,north,2,,,\n'),2)
  self.assertEqual(import_events(self.s,header+'a,receive,Widget,north,10,,,\n'),0)
  self.assertEqual(import_events(self.s,'event_id,kind,original_id\nc,reverse,b\n'),1)
  self.assertEqual(self.s.balance('Widget','north'),10)
 @unittest.skipIf(STAGE<5,'future')
 def test_12_snapshot(self):
  self.s.apply(event('z',sku='z'));self.s.apply(event('a',sku='A'));self.s.apply(event('h','reserve',3,sku='A',reservation_id='hold'))
  self.s.apply(event('t','transfer',2,sku='z',to_warehouse='west'));self.s.apply(dict(event_id='rt',kind='reverse',original_id='t'))
  expected=[dict(sku='A',warehouse='north',stock=20,reserved=3,available=17),dict(sku='z',warehouse='north',stock=20,reserved=0,available=20),dict(sku='z',warehouse='west',stock=0,reserved=0,available=0)]
  self.assertEqual(self.s.snapshot(),expected)
 @unittest.skipIf(STAGE<5,'future')
 def test_13_cli_and_failed_import(self):
  self.s.close()
  csv=Path(self.tmp.name)/'import.csv';csv.write_text('event_id,kind,sku,warehouse,qty\na,receive,A,north,7\n')
  def cli(*a):return subprocess.run([sys.executable,'-m','stockroom','--db',self.path,*a],capture_output=True,text=True,timeout=15)
  r=cli('import',str(csv));self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),{'applied':1})
  r=cli('snapshot');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),[dict(sku='A',warehouse='north',stock=7,reserved=0,available=7)])
  csv.write_text('event_id,kind,sku,warehouse,qty\nb,receive,A,north,2\nc,ship,A,north,99\n')
  self.assertNotEqual(cli('import',str(csv)).returncode,0)
  self.s=Store(self.path);self.assertEqual(self.s.balance('A','north'),7)
suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contract)
r=unittest.TestResult();suite.run(r)
print(json.dumps({'checks':r.testsRun-len(r.skipped),'passed_checks':r.testsRun-len(r.skipped)-len({str(t) for t,_ in r.failures+r.errors}),'failures':[(str(t),s) for t,s in r.failures+r.errors]}))
'''

def grade(workspace,stage):
    import os
    r=subprocess.run([sys.executable,'-c',GRADER],cwd=workspace,env=dict(os.environ,STOCKROOM_STAGE=str(stage)),capture_output=True,text=True,timeout=90)
    try:return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError):return {'checks':[3,5,7,9,11,13][stage],'passed_checks':0,'failures':[('grader import/runtime',r.stderr[-6000:])]}
