"""Measure actual CLI Responses streams in the isolated synthetic app harness.

Records event types and timing only, never credentials or prompt/token content.
The existing restricted bridge and namespace retain their access boundaries.
"""
import json
import threading
import time
import subprocess
from pathlib import Path


def start_with_caches(agent, start, broker, model, key_env, slots):
    """Use production lazy-pool/cache code with a disposable catalog and state."""
    catalog = agent.root / 'probe-catalog.toml'
    source = Path('/home/deforest/Documents/DEV/Marathon/config/runtime_catalog.toml').read_text()
    settings = source.split('[[backends]]', 1)[0]
    catalog.write_text(settings + '\n[[backends]]\nid="session-probe"\n'
                       'display_name="Isolated session probe"\nkind="llama_swap_pool"\n'
                       f'proxy={json.dumps(broker)}\npool_models=[{json.dumps(model)}]\n'
                       f'api_key_env={json.dumps(key_env)}\n'
                       'model_alias="qwen3.8-27b-uncensored"\nsupports_slots=true\n')
    original = subprocess.Popen

    def launch(command, *args, **kwargs):
        if any(str(x).endswith('/scripts/routers/codex_local_router.py') for x in command):
            env = dict(kwargs['env'])
            env.update(MARATHON_CATALOG=str(catalog), MARATHON_USER_CATALOG=str(agent.root / 'absent-user-catalog.toml'),
                       MARATHON_LAZY_POOL_BACKEND='session-probe', MARATHON_POOL_RUNTIME_DIR=str(agent.root / 'pool-runtime'),
                       MARATHON_BACKEND_MODEL_ID='qwen3.8-27b-uncensored',
                       MARATHON_MODEL_PREWARM_SLUG='qwen3.8-27b-uncensored',
                       MARATHON_MODEL_READY_TIMEOUT_SECONDS='360',
                       MARATHON_STARTER_CACHE_ENABLED='1', MARATHON_SLOT_SNAPSHOTS_ENABLED='1',
                       MARATHON_SLOT_SAVE_ROOT=str(slots.parent), MARATHON_BACKEND_CACHE_ID='isolated-session-probe')
            kwargs['env'] = env
        return original(command, *args, **kwargs)

    # The helper starts only this run's router. The inference model and broker
    # remain outside the restricted frontend namespace, as in the base harness.
    subprocess.Popen = launch
    try:
        start()
    finally:
        subprocess.Popen = original


def instrument_agent(agent):
    path = agent.root / 'stream-timings.jsonl'
    lock = threading.Lock()
    current = {}

    def emit(row):
        with lock, path.open('a') as output:
            output.write(json.dumps(row) + '\n')

    original_phase = agent.phase

    def phase(folder, project, prompt, stage):
        current.update(case=folder.name, stage=stage, started=time.monotonic())
        emit(dict(event='phase_start', case=folder.name, stage=stage))
        result = original_phase(folder, project, prompt, stage)
        emit(dict(event='phase_end', case=folder.name, **result))
        return result

    agent.phase = phase
    original_factory = agent.delegate.bridge_handler

    def factory(port, token):
        base = original_factory(port, token)

        class Handler(base):
            def send_response(self, code, message=None):
                self.probe_status = code
                return super().send_response(code, message)

            def forward(self):
                if self.command != 'POST' or self.path != '/v1/responses':
                    return super().forward()
                started = time.monotonic()
                phase_info = dict(current)
                row = dict(event='response', case=phase_info.get('case'), stage=phase_info.get('stage'),
                           request_offset_s=started - phase_info.get('started', started))
                output = self.wfile
                pending = bytearray()

                class TimedWriter:
                    def __getattr__(self, key):
                        return getattr(output, key)

                    def write(self, data):
                        pending.extend(data)
                        while b'\n' in pending:
                            line, _, rest = pending.partition(b'\n')
                            pending[:] = rest
                            if not line.startswith(b'data:'):
                                continue
                            try:
                                event = json.loads(line[5:])
                            except (ValueError, UnicodeError):
                                continue
                            kind = event.get('type', '')
                            elapsed = time.monotonic() - started
                            if kind.endswith('.delta') and event.get('delta'):
                                row.setdefault('first_generation_s', elapsed)
                                row.setdefault('first_generation_type', kind)
                            if kind == 'response.output_text.delta' and event.get('delta'):
                                row.setdefault('first_text_s', elapsed)
                            if kind == 'response.failed':
                                row['failed_event'] = True
                        return output.write(data)

                self.wfile = TimedWriter()
                try:
                    super().forward()
                finally:
                    self.wfile = output
                    row['duration_s'] = time.monotonic() - started
                    row['http_status'] = getattr(self, 'probe_status', None)
                    emit(row)

        return Handler

    agent.delegate.bridge_handler = factory
