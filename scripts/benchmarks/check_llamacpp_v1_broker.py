#!/usr/bin/env python3
"""Qualify a pinned candidate through an isolated broker and synthetic Marathon app.

Uses a new disposable credential, fresh state and a free selected GPU. Never reads production
credentials, catalogs, conversations or snapshots. Production broker is untouched.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess
import time
from types import SimpleNamespace
import urllib.request
import yaml

CONTROL = Path('/home/deforest/Documents/DEV/gpu-control')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-queries', type=int, default=256)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    slots = a.output / 'slots'; slots.mkdir()
    name = f'marathon-v1-broker-check-gpu{a.gpu}'
    model = f'marathon-qwen3.8-27b-uncensored-{a.gpu if a.gpu in (1, 2, 3) else 1}'
    base = 'http://127.0.0.1:19293'
    upstream = base + '/upstream/' + model
    token = secrets.token_urlsafe(32)
    env_name = 'MARATHON_V1_QUALIFICATION_KEY'
    checks = []

    def check(label, ok, **details):
        checks.append(dict(check=label, passed=bool(ok), **details))
        (a.output / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        print(label, bool(ok), flush=True)
        if not ok:
            raise RuntimeError(label)

    def request(url, data=None):
        req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json'}, data=None if data is None else json.dumps(data).encode())
        with urllib.request.urlopen(req, timeout=180) as response:
            body = response.read().decode()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {'status': body}

    def running():
        return request(base + '/running')['running']

    def free():
        return not subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'], text=True).strip()

    lease = None
    if a.gpu in (1, 2, 3):
        pool = Path(f'/run/user/{os.getuid()}/marathon/backend-pools/llama-swap-qwen3.8-uncensored-pool')
        suffix = hashlib.sha256(model.encode()).hexdigest()[:12]
        lease = (pool / f'{model}-{suffix}.lock').open('a+')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check('gpu_initially_free', free())
    check('container_name_available', subprocess.run(['docker', 'inspect', name],
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0)
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 19293))
    config = yaml.safe_load((CONTROL / 'llama-swap/config.yaml').read_text())
    entry = dict(config['models'][model])
    production_ttl = entry['ttl']
    cmd = shlex.split(entry['cmd'])
    at = next(i for i, v in enumerate(cmd) if v.startswith('sha256:'))
    cmd[at] = a.image
    cmd[cmd.index('--name') + 1] = name
    cmd[cmd.index('--gpus') + 1] = f'"device={a.gpu}"'
    for i, v in enumerate(cmd):
        if v.endswith(':/cache'):
            cmd[i] = str(slots) + ':/cache'
    cmd[at:at] = ['--env', 'GGML_CUDA_MARATHON_Q4_MAX_QUERIES=' + str(a.max_queries)]
    if '--spec-draft-n-ubatch' not in cmd:
        cmd += ['--spec-draft-n-ubatch', '64']
    entry.update(cmd=shlex.join(cmd), cmdStop='docker stop --timeout 60 ' + name,
                 unlisted=False, ttl=15)
    entry.pop('macros', None)
    isolated = dict(healthCheckTimeout=900, logLevel='info', startPort=19980, globalTTL=15,
                    apiKeys=['${env.' + env_name + '}'], models={model: entry})
    config_path = a.output / 'broker.yaml'
    config_path.write_text(yaml.safe_dump(isolated))
    broker_env = dict(os.environ, **{env_name: token})
    os.environ[env_name] = token  # only the freshly created test credential
    log = (a.output / 'broker.log').open('w')
    broker = None
    agent = None

    def start():
        nonlocal broker
        proc = subprocess.Popen(['/home/deforest/.local/bin/llama-swap', '-config', str(config_path),
                                 '-listen', '127.0.0.1:19293'], env=broker_env, stdout=log, stderr=subprocess.STDOUT)
        broker = proc
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('Isolated broker exited')
            try:
                request(base + '/health')
                return proc
            except OSError:
                time.sleep(.2)
        proc.terminate(); proc.wait(timeout=10)
        raise RuntimeError('Isolated broker readiness timeout')

    try:
        broker = start()
        check('broker_discovery', bool(request(base + '/v1/models').get('data')))
        check('health_and_discovery_are_lazy', not running() and free())
        spec = importlib.util.spec_from_file_location('v1_broker_app', CONTROL / 'benchmarks/qwen38-short-agentic-ab.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

        def app_http(path, data=None):
            # Preserve a truly unloaded first app request.
            if path.startswith('/slots/') and not running():
                return {}
            return request(upstream + path, data)

        class ProbeAgent(module.ShortAgent):
            def start(self):
                super().start()
                check('router_discovery_is_lazy', not running() and free())

        root = a.output / 'app'; root.mkdir()
        study = SimpleNamespace(out=root, base=base, http=app_http)
        args = SimpleNamespace(gpu=a.gpu, label='v1-broker-check', context=262144,
            backend_model_id=model, api_key_env=env_name,
            model_path='/home/deforest/AI/models/gguf/swift15-heretic-ara-merge/Swift-1.5-Qwen3.8-27B-Heretic-ARA-Merge-IQ4_XS.gguf',
            cases=['cache'], stages=2, phase_timeout=180)
        agent = ProbeAgent(study, args)
        agent.run()
        for case in agent.results:
            for phase in case['phases']:
                grade = phase['grade']
                check('app_' + case['case'] + '_' + str(phase['stage']), phase['exit_code'] == 0 and
                      bool(grade.get('checks')) and grade.get('passed_checks') == grade['checks'], result=phase)
        host_uuid = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        visible = subprocess.check_output(['docker', 'exec', name, 'nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        check('worker_gpu_isolation', visible == host_uuid)
        agent.close(); agent = None
        deadline = time.monotonic() + 90
        while running() and time.monotonic() < deadline:
            time.sleep(1)
        check('idle_unload', not running() and free(), tested_ttl_s=15, production_ttl_s=production_ttl)
        broker.terminate(); broker.wait(timeout=90); broker = None
        broker = start()
        check('restart_is_healthy_and_lazy', bool(request(base + '/v1/models').get('data')) and not running() and free())
    finally:
        if agent:
            agent.close()
        if broker:
            broker.terminate()
            try:
                broker.wait(timeout=90)
            except subprocess.TimeoutExpired:
                broker.kill(); broker.wait()
        subprocess.run(['docker', 'stop', '--time', '60', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ.pop(env_name, None)
        log.close()
        for f in slots.iterdir():
            if f.is_file():
                f.unlink()
    check('gpu_released', free())


if __name__ == '__main__':
    main()
