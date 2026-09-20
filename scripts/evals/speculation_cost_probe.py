#!/usr/bin/env python3
"""Paired synthetic requests on an exclusively leased, initially unloaded worker.

Uses the registered configuration or an exclusively leased diagnostic copy;
never reads or restores production slot snapshots.
Zero proposals is a request-level reference with the drafter still loaded, not a
separate no-drafter process. Saves synthetic responses and numeric metric deltas.
"""
import argparse
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import signal
import re
import random
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from marathon_app.catalog import backends
from marathon_app.pool import acquire_pool_worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, choices=[1, 2], default=1)
    p.add_argument('--mode', choices=['paired', 'copy-peak', 'phase-cost', 'wide-copy', 'lookup-copy', 'mixed-copy'], default='paired')
    p.add_argument('--lookup-width', type=int, choices=[6, 7, 15, 31, 63, 127, 255], default=15,
                   help='Proposal length for lookup-copy; six isolates drafter removal.')
    p.add_argument('--quality', action='store_true', help='Run independent synthetic quality checks instead of copy peak.')
    p.add_argument('--long-quality', action='store_true', help='Run the synthetic near-capacity retrieval screen.')
    p.add_argument('--quality-case', choices=['merge_intervals'], help='Rerun a selected quality case with the recorded prompt.')
    p.add_argument('--server-library', type=Path, help='Isolated server library overlay; never alters the runtime image.')
    p.add_argument('--copy-fixture', choices=['registry', 'varied'], default='registry')
    p.add_argument('--copy-temperature', type=float, default=0.0)
    p.add_argument('--copy-padding-records', type=int, choices=[0, 7000], default=0,
                   help='Synthetic unrelated archive after the file, to exercise long-context lookup.')
    p.add_argument('--app-smoke', action='store_true', help='Run a fresh real-Marathon coding task against the diagnostic worker.')
    a = p.parse_args()
    if not 0 <= a.copy_temperature <= 2:
        p.error('Copy temperature must be between zero and two.')
    copy_width = a.lookup_width if a.mode in ('lookup-copy', 'mixed-copy') else (7 if a.mode == 'wide-copy' else 6)
    if a.server_library and a.mode not in ('lookup-copy', 'mixed-copy', 'phase-cost'):
        p.error('A server library requires a diagnostic container mode.')
    if a.app_smoke and a.mode not in ('lookup-copy', 'mixed-copy', 'phase-cost'):
        p.error('Application smoke testing requires an isolated diagnostic endpoint.')
    a.output.mkdir(parents=True, exist_ok=False)
    model = f'marathon-qwen3.8-27b-uncensored-{a.gpu}'
    backend = replace(backends()['llama-swap-qwen3.8-uncensored-pool'], pool_models=(model,))
    lease, _ = acquire_pool_worker(backend, Path('/run/user/1000/marathon'), 'synthetic-speculation-cost')
    env = Path('/home/deforest/Documents/DEV/qwen-inference/.env')
    key = next(l.split('=', 1)[1].strip().strip('\"\'') for l in env.read_text().splitlines()
               if l.startswith('HERMES_API_KEY='))
    base = 'http://127.0.0.1:9292'
    scratch = None
    def request(path, body=None, raw=False):
        if scratch:
            path = path.removeprefix('/upstream/' + model)
        req = urllib.request.Request(base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read().decode()
        return data if raw else (json.loads(data) if data else {})
    def save(name, data):
        (a.output / name).write_text(json.dumps(data, indent=2) + '\n')
    def metrics():
        text = request('/upstream/' + model + '/metrics', raw=True)
        return {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1])
                for line in text.splitlines() if line and not line.startswith('#')}
    owned = False
    def stop(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    try:
        running = request('/running')['running']
        if any(x['model'] == model for x in running):
            raise RuntimeError('Selected worker already loaded; refusing to touch its state.')
        memory, power = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
            '--query-gpu=memory.used,power.limit', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        assert int(memory) < 100 and float(power) <= 275
        config = Path('/home/deforest/Documents/DEV/gpu-control/llama-swap/config.yaml')
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        save('protocol.json', {'model': model, 'gpu': a.gpu, 'power_w':float(power),
            'config_sha256':digest, 'context_capacity':196000, 'privacy':'new synthetic fixtures only',
            'order':[6,0,0,6] if a.mode == 'paired' else [copy_width,copy_width], 'mode':a.mode,
            'sampling':{'reasoning':'medium','temperature':([0,0.7] if a.quality else (0.7 if a.app_smoke else a.copy_temperature))},
            'quality_suite':a.quality, 'long_quality':a.long_quality,
            'copy_fixture':a.copy_fixture,
            'copy_temperature':a.copy_temperature,
            'copy_padding_records':a.copy_padding_records,
            'limitation':'Different speculative paths may change floating-point outputs; compare hashes and completed-task checks.'})
        if a.mode in ('phase-cost', 'wide-copy', 'lookup-copy', 'mixed-copy'):
            import yaml
            entry=yaml.safe_load(config.read_text())['models'][model]
            name='marathon-synthetic-'+a.mode
            command=entry['cmd'].replace('${gpu}',str(a.gpu)).replace('${PORT}','19979').replace('${MODEL_ID}','synthetic-phase-cost')
            argv=shlex.split(command)
            argv[argv.index('--name')+1]=name
            argv[argv.index('--alias')+1]='synthetic-phase-cost'
            cache=a.output.resolve()/'slots';cache.mkdir()
            for i,value in enumerate(argv):
                if value.endswith(':/cache'):
                    argv[i]=str(cache)+':/cache'
            argv.insert(2,'--detach')
            if a.server_library:
                library=a.server_library.resolve(strict=True)
                argv[2:2]=['--volume',str(library)+':/app/libllama-server-impl.so:ro']
                save('server-library.json',{'path':str(library),'sha256':hashlib.sha256(library.read_bytes()).hexdigest()})
            argv.remove('--rm')  # Keep startup failure logs until our cleanup.
            argv+=['--log-verbosity','4' if a.mode == 'phase-cost' else '3']
            if a.mode in ('wide-copy', 'lookup-copy'):
                for flag in ('--spec-draft-n-max', '--spec-ngram-map-k-size-m'):
                    argv[argv.index(flag)+1]=str(copy_width)
            if a.mode == 'lookup-copy':
                argv[argv.index('--spec-type')+1]='ngram-map-k'
                for flag in ('--spec-draft-model', '--spec-draft-ngl', '--cache-type-k-draft', '--cache-type-v-draft'):
                    index=argv.index(flag)
                    del argv[index:index+2]
            if a.mode == 'mixed-copy':
                argv[argv.index('--spec-ngram-map-k-size-m')+1]=str(copy_width)
            save('scratch-command.json',argv)
            # The registered worker is unloaded and leased; only one copy runs.
            subprocess.run(argv,check=True,stdout=subprocess.DEVNULL)
            scratch=name
            base='http://127.0.0.1:19979'
            for _ in range(180):
                state=subprocess.check_output(['docker','inspect','--format','{{.State.Running}}',name],text=True).strip()
                if state != 'true':
                    raise RuntimeError('Diagnostic worker exited; see server.log.')
                try:
                    if request('/health').get('status')=='ok':
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('Diagnostic worker did not become healthy.')
        else:
            owned = True
        request('/v1/chat/completions', {'model':model,'messages':[{'role':'user','content':'Reply READY.'}], 'max_tokens':8})
        if a.app_smoke:
            from speculation_quality_cases import run_app
            run_app(a.output,base,save)
            assert hashlib.sha256(config.read_bytes()).hexdigest()==digest
            return
        if a.quality or a.long_quality:
            from speculation_quality_cases import run
            run(request, '', model, save, copy_width, long=a.long_quality, only=a.quality_case)
            assert hashlib.sha256(config.read_bytes()).hexdigest()==digest
            return
        records={f'station_{i:03d}':{'region':f'zone_{i%7}','quota':100+i,'enabled':True} for i in range(100)}
        if a.copy_fixture == 'varied':
            rng=random.Random(820631)
            for record in records.values():
                record.update(region=''.join(rng.choice('abcdefghjkmnpqrstuvwxy') for _ in range(12)),
                              quota=rng.randrange(10000,99999),enabled=bool(rng.randrange(2)))
        source = '\n'.join(f'    "{key}": {{"region": "{value["region"]}", "quota": {value["quota"]}, "enabled": {value["enabled"]}}},' for key,value in records.items())
        tasks = {
            'copy_edit': 'Return this Python file in full, changing only station_050 quota to999. No explanation.\nREGISTRY = {\n' + source + '\n}\n',
            'new_code': 'Write a complete Python function merge_intervals(items) that merges overlapping and touching closed integer intervals. Sort without modifying input. Include eight assert tests covering empty, negative, nested, unsorted, duplicate and touching intervals. Return one Python code block only.',
            'novel_prose': 'Write a350-word fictional scene about a night-shift archivist discovering that a shipment manifest lists tomorrow as its departure date. Use natural dialogue and an ambiguous ending. Do not explain the story.'}
        if a.copy_padding_records:
            filler='\n'.join(f'Archive record {i:04d}: region=blue; revision=2; code=obsolete-{i:04d}.' for i in range(a.copy_padding_records))
            tasks['copy_edit']+='\nThe following archive is unrelated to the Python file. Do not reproduce it.\n<archive>\n'+filler+'\n</archive>\nReturn only the original 100-record REGISTRY Python file, with station_050 quota changed to 999 and everything else preserved.'
        save('fixtures.json', tasks)
        if a.mode in ('copy-peak', 'wide-copy', 'lookup-copy', 'mixed-copy'):
            chat = {'model':model, 'messages':[{'role':'user','content':tasks['copy_edit']}],
                    'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'medium'}}
            prompt = request('/upstream/'+model+'/apply-template', chat)['prompt']
            for stage in range(2):
                body = {'prompt':prompt,'n_predict':8192,'temperature':a.copy_temperature,'seed':6100,
                        'stream':True,'return_tokens':True,'timings_per_token':True,
                        'cache_prompt':True,'speculative.n_max':copy_width}
                save(f'copy-{stage}-request.json', body)
                req = urllib.request.Request(base+('/completion' if scratch else '/upstream/'+model+'/completion'),
                    data=json.dumps(body).encode(), headers={'Authorization':'Bearer '+key,
                    'Content-Type':'application/json'})
                start=time.monotonic(); events=[]; text=[]; final=None
                with urllib.request.urlopen(req,timeout=600) as response:
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        data=json.loads(line[6:]); elapsed=time.monotonic()-start
                        events.append({'arrival_s':elapsed,'event':data})
                        if data.get('stop'):
                            final=data
                        else:
                            text.append(data.get('content',''))
                save(f'copy-{stage}-events.json',events)
                assert final is not None
                content=''.join(text).split('</think>')[-1].strip()
                blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',content,re.S)
                code=blocks[-1] if blocks else content
                passed=False
                try:
                    tree=ast.parse(code)
                    assert len(tree.body)==1 and isinstance(tree.body[0],ast.Assign)
                    assert isinstance(tree.body[0].targets[0],ast.Name) and tree.body[0].targets[0].id=='REGISTRY'
                    value=ast.literal_eval(tree.body[0].value)
                    expected=json.loads(json.dumps(records))
                    expected['station_050']['quota']=999
                    passed=value==expected
                except (AssertionError,SyntaxError,ValueError,TypeError):
                    pass
                samples=[(x['event'].get('tokens_predicted',0),x['arrival_s']) for x in events if not x['event'].get('stop')]
                windows=[]; left=0
                for right in range(len(samples)):
                    while (left+1<right and samples[right][0]-samples[left+1][0]>=512
                           and samples[right][1]-samples[left+1][1]>=2):
                        left+=1
                    dn=samples[right][0]-samples[left][0]; dt=samples[right][1]-samples[left][1]
                    if dn>=512 and dt>=2:
                        windows.append({'tokens':dn,'seconds':dt,'tps':dn/dt,
                                        'start_token':samples[left][0],'end_token':samples[right][0]})
                row={'stage':stage,'quality_pass':passed,'stop_type':final.get('stop_type'),
                     'timings':final.get('timings'),'wall_s':time.monotonic()-start,
                     'content_sha256':hashlib.sha256(content.encode()).hexdigest(),
                     'best_512_token_window':max(windows,key=lambda x:x['tps']) if windows else None}
                save(f'copy-{stage}-summary.json',row);print(json.dumps(row),flush=True)
            assert hashlib.sha256(config.read_bytes()).hexdigest()==digest
            return
        for case, prompt in tasks.items():
            body = {'model':model, 'messages':[{'role':'user','content':prompt}],
                'max_tokens':768, 'temperature':0, 'seed':6100, 'stream':False,
                'cache_prompt':True, 'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'medium'}}
            request('/v1/chat/completions', dict(body, max_tokens=8))
            for stage, width in enumerate([6,6] if a.mode == 'phase-cost' else [6,0,0,6]):
                before=metrics(); start=time.monotonic()
                data=request('/v1/chat/completions', dict(body, **{'speculative.n_max':width}))
                elapsed=time.monotonic()-start; after=metrics()
                save(f'{case}-{stage}-response.json', data)
                message=data['choices'][0]['message']
                row={'case':case,'stage':stage,'width':width,'wall_s':elapsed,
                     'timings':data.get('timings'),'usage':data.get('usage'),
                     'finish_reason':data['choices'][0]['finish_reason'],
                     'message_sha256':hashlib.sha256(json.dumps(message,sort_keys=True).encode()).hexdigest(),
                     'metrics':{k:after[k]-before.get(k,0) for k in after if 'spec_' in k}}
                with (a.output/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        assert hashlib.sha256(config.read_bytes()).hexdigest()==digest
    finally:
        if scratch:
            try:
                with (a.output/'server.log').open('w') as log:
                    subprocess.run(['docker','logs',scratch],stdout=log,stderr=subprocess.STDOUT,check=False)
                subprocess.run(['docker','stop','--time','60',scratch],check=True,stdout=subprocess.DEVNULL)
                subprocess.run(['docker','rm',scratch],check=True,stdout=subprocess.DEVNULL)
            finally:
                lease.close()
        elif owned:
            # Only this run could load this reserved worker after the initial checks.
            try:
                request('/api/models/unload/'+model,{},raw=True)
            finally:
                lease.close()
        else:
            lease.close()


if __name__ == '__main__':
    main()
