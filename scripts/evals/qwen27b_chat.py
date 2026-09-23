#!/usr/bin/env python3
"""Bare merged-27B chat using a reserved worker, without any agent context."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from marathon_app.catalog import backends
from marathon_app.pool import acquire_pool_worker
from flash_next_chat import main as chat


def main():
    backend = backends()['llama-swap-qwen3.8-uncensored-pool']
    # Avoid GPU 2, which can host the user's separate shared inference service.
    backend = replace(backend, pool_models=tuple(
        model for model in backend.pool_models if model.endswith(('-1', '-3'))))
    key = os.environ.get(backend.api_key_env, '')
    if not key:
        key = next(line.split('=', 1)[1].strip().strip('\"\'')
                   for line in Path(backend.api_key_file).read_text().splitlines()
                   if line.startswith(backend.api_key_env + '='))
    headers = {'Authorization': 'Bearer ' + key}
    lease, model = acquire_pool_worker(
        backend, Path(f'/run/user/{os.getuid()}/marathon'), 'bare-27b-chat')
    owned = False
    try:
        request = urllib.request.Request(backend.proxy + '/running', headers=headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            running = json.load(response)['running']
        # Refuse to displace Bonsai or any other broker-managed workload on GPU 3.
        ids = {row.get('model') for row in running}
        if model.endswith('-3') and ids.intersection({'ternary-bonsai-2-27b-pq2', 'qwen3.8-27b-dflash'}):
            raise RuntimeError('GPU 3 has another workload; retry when a worker is free.')
        owned = model not in ids
        print(f'Reserved {model}. First reply may need a model load.')
        chat(backend.proxy + '/v1/chat/completions', model, headers, 'Qwen 27B merge')
    finally:
        try:
            if owned:
                request = urllib.request.Request(
                    backend.proxy + '/api/models/unload/' + model,
                    data=b'', headers=headers, method='POST')
                with urllib.request.urlopen(request, timeout=90):
                    pass
        finally:
            lease.close()


if __name__ == '__main__':
    main()
