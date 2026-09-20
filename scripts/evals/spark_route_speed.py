#!/usr/bin/env python3
"""Compare real Marathon CLI traffic with an exact direct Spark replay.

Outputs stay in the requested diagnostic directory. Temporary client config is
removed automatically; production settings and inference services are untouched.
"""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
import time

from aiohttp import ClientSession, ClientTimeout, web


async def main(out):
    out.mkdir(parents=True, exist_ok=False)
    original = (Path.home() / '.config/marathon/catalog.toml').read_text()
    direct = 'http://127.0.0.1:18088'
    broker = 'http://127.0.0.1:9292'
    captures = []
    results = []

    def observe(raw, record, started):
        for line in raw.splitlines():
            if not line.startswith(b'data: '):
                continue
            try:
                event = json.loads(line[6:])
            except (ValueError, UnicodeError):
                continue
            kind = event.get('type', '')
            if kind.endswith('.delta') and event.get('delta'):
                record.setdefault('first_generation_s', time.monotonic() - started)
                if kind == 'response.output_text.delta':
                    record.setdefault('first_prose_s', time.monotonic() - started)
                    record['prose'] = record.get('prose', '') + event['delta']
            if kind in ('response.completed', 'response.incomplete', 'response.failed'):
                response = event.get('response', {})
                record['usage'] = response.get('usage')
                record['status'] = response.get('status', kind)
                record['incomplete_details'] = response.get('incomplete_details')

    async with ClientSession(timeout=ClientTimeout(total=240)) as client:
        async def proxy(request):
            body = await request.read()
            measured = request.method == 'POST' and request.path.endswith('/responses')
            record = {'route': 'marathon', 'path': request.path}
            if measured:
                captures.append(json.loads(body))
                (out / f'request-{len(captures)}.json').write_text(json.dumps(captures[-1], indent=2))
            started = time.monotonic()
            headers = {key: value for key, value in request.headers.items()
                       if key.lower() in ('authorization', 'content-type')}
            async with client.request(request.method, broker + request.path_qs,
                                      data=body, headers=headers) as upstream:
                response = web.StreamResponse(status=upstream.status,
                    headers={'Content-Type': upstream.headers.get('Content-Type', 'application/json')})
                await response.prepare(request)
                async for line in upstream.content:
                    if measured:
                        observe(line, record, started)
                    await response.write(line)
                await response.write_eof()
            if measured:
                record['total_s'] = time.monotonic() - started
                results.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != 'prose'}), flush=True)
            return response

        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.router.add_route('*', '/{tail:.*}', proxy)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory(prefix='marathon-spark-speed-') as temporary:
                catalog = Path(temporary) / 'catalog.toml'
                # Locate Spark explicitly; new menu entries may precede it.
                marker = 'base_url = "http://127.0.0.1:9292/v1"'
                identity = 'id = "qwen3.8-flash-next-spark"'
                prefix, spark = original.split(identity, 1)
                assert marker in spark.split('[[', 1)[0]
                catalog.write_text(prefix + identity + spark.replace(
                    marker, f'base_url = "http://127.0.0.1:{port}/v1"', 1))
                env = dict(os.environ, MARATHON_USER_CATALOG=str(catalog))
                prompt = ('Write a vivid 100-word description of a small bookstore during a rainstorm. '
                          'Return only the prose. Do not use tools or inspect files.')
                for trial in range(2):
                    started = time.monotonic()
                    command = [str(Path.home() / '.local/bin/marathon'), '--instance', 'spark-speed-check',
                               'exec', '--json', '--sandbox', 'read-only', '-m', 'qwen3.8-flash-next-spark',
                               '-c', 'approval_policy="never"', '-c', 'model_reasoning_effort="medium"', prompt]
                    process = await asyncio.create_subprocess_exec(*command, env=env,
                        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE)
                    stdout, stderr = await asyncio.wait_for(process.communicate(), 240)
                    (out / f'cli-{trial}.jsonl').write_bytes(stdout)
                    (out / f'cli-{trial}.stderr').write_bytes(stderr)
                    if process.returncode or not captures:
                        raise RuntimeError(f'CLI failed: {process.returncode}; see {out}')
                    print(json.dumps({'cli_wall_s': time.monotonic() - started, 'trial': trial}), flush=True)
                    body = copy.deepcopy(captures[-1])
                    body['model'] = body['model'].removeprefix('spark/')
                    record = {'route': 'direct', 'trial': trial}
                    started = time.monotonic()
                    async with client.post(direct + '/v1/responses', json=body) as response:
                        response.raise_for_status()
                        async for line in response.content:
                            observe(line, record, started)
                    record['total_s'] = time.monotonic() - started
                    results.append(record)
                    print(json.dumps({k: v for k, v in record.items() if k != 'prose'}), flush=True)
                    (out / 'results.json').write_text(json.dumps(results, indent=2))
        finally:
            await runner.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.output))
