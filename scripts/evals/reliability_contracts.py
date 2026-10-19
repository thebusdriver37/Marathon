"""Additional independently graded fixtures, separate from the earlier prompt sweep."""


def cases():
    return {
        "atomic_report": {
            "files": {
                "invreport/__init__.py": "from .core import build\n",
                "invreport/core.py": "def build(text):\n    return {'applied': 0, 'balances': {}}\n",
                "invreport/__main__.py": (
                    "import json, sys\nfrom pathlib import Path\nfrom .core import build\n"
                    "Path(sys.argv[2]).write_text(json.dumps(build(Path(sys.argv[1]).read_text())))\n"),
                "SPEC.md": (
                    "build(text) consumes CSV with exactly headers id,item,delta, in any order. "
                    "Reject duplicate/missing/extra headers and rows with missing/extra cells. "
                    "Ignore empty or whitespace-only physical lines. A row containing CSV delimiters "
                    "still requires valid cells. IDs are trimmed, nonempty, case-sensitive. "
                    "Items are trimmed, Unicode-casefolded, and nonempty. delta is a signed base-10 integer "
                    "written with ASCII digits only (optional + or -); trim surrounding whitespace. "
                    "An identical normalized ID/item/delta replay is ignored; reusing an ID with different "
                    "normalized item or delta raises ValueError. Process in input order; no item balance "
                    "may ever go negative. Zero deltas are valid and create a zero-balance item. "
                    "Return exactly {'applied': N, 'balances': {item: quantity}}, counting only unique events. "
                    "Invalid input raises ValueError. Use Python standard library.\n"
                    "CLI: python3 -m invreport INPUT_CSV OUTPUT_JSON. Success writes the build result "
                    "as JSON and exits 0. Any invalid CSV exits nonzero, reports the error on stderr, and "
                    "preserves an existing OUTPUT_JSON byte-for-byte (or leaves it absent if nonexistent). "
                    "Preserve input files.\n"),
                "test_report.py": (
                    "import unittest\nfrom invreport import build\nclass Tests(unittest.TestCase):\n"
                    "    def test_receipt(self):\n"
                    "        self.assertEqual(build('id,item,delta\\na,Widget,3\\n'), "
                    "{'applied': 1, 'balances': {'widget': 3}})\n"),
                "notes.txt": "Unrelated user draft: do not change.\n",
            },
            "protected": ["SPEC.md", "test_report.py", "notes.txt"],
            "prompt": "Implement SPEC.md across invreport's library and CLI. Do not edit SPEC.md, test_report.py, "
                      "or notes.txt; put new tests in a separate file. "
                      "Reproduce the failing behavior, add useful regression coverage, and verify both library and CLI. "
                      "Pay attention to replay conflicts, invalid input, and preserving existing output files. Do not commit.",
            "oracle": r'''import json, subprocess, sys
from pathlib import Path
from invreport import build
text='delta,item,id\n +7 , Straße ,A\n7,STRASSE,A\n-2,strasse,B\n0,Empty,Z\n\n'
expected={'applied':3,'balances':{'strasse':5,'empty':0}}
assert build(text)==expected, build(text)
assert build('id,item,delta\na,"x,y",3\nb,"x,y",-3\n')=={'applied':2,'balances':{'x,y':0}}
assert build('id,item,delta\na,X,2\nA,x,1\n')=={'applied':2,'balances':{'x':3}}
invalid=[
 'id,item,delta\na,x,2\na,x,3\n', 'id,item,delta\na,x,-1\nb,x,2\n',
 'id,item,delta\na,x,2\na,y,2\n', 'id,item,delta\na,x,1.0\n',
 'id,item,delta\na,x,1_000\n', 'id,item,delta\na,x,١\n',
 'id,item,delta\na,x,True\n', 'id,item,delta\na,x,\n',
 'id,item,delta\n ,x,2\n', 'id,item,delta\na, ,2\n',
 'id,item,delta\na,x\n', 'id,item,delta\na,x,2,extra\n',
 'id,item,delta,delta\na,x,2,2\n', 'id,item\na,x\n',
 'id,item,delta,other\na,x,2,z\n', 'id,item,delta\n,,\n', 'id,item,delta\n,,,\n']
for value in invalid:
 try: build(value)
 except ValueError: pass
 else: raise AssertionError(('accepted invalid input',value))
original=b'USER OUTPUT MUST SURVIVE\n'
Path('probe.csv').write_text(text)
r=subprocess.run([sys.executable,'-m','invreport','probe.csv','probe.json'],capture_output=True,text=True,timeout=5)
assert r.returncode==0, r.stderr
assert json.loads(Path('probe.json').read_text())==expected
assert Path('probe.csv').read_text()==text
for value in invalid[::2]:
 Path('probe.csv').write_text(value);Path('probe.json').write_bytes(original)
 r=subprocess.run([sys.executable,'-m','invreport','probe.csv','probe.json'],capture_output=True,text=True,timeout=5)
 assert r.returncode!=0 and r.stderr.strip(), (r.returncode,r.stdout,r.stderr)
 assert Path('probe.json').read_bytes()==original
 Path('probe.json').unlink()
 r=subprocess.run([sys.executable,'-m','invreport','probe.csv','probe.json'],capture_output=True,text=True,timeout=5)
 assert r.returncode!=0 and not Path('probe.json').exists()
''',
        },
    }
