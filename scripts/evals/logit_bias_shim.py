#!/usr/bin/env python3
"""Loopback shim that adds one logit_bias map to Marathon's requests.

Transparent forwarder in front of the llama-swap broker. It changes exactly one
key of a JSON request body: logit_bias. Everything else, including headers,
streaming and non-inference routes, is forwarded unchanged, so all arms share
one code path and the baseline arm simply passes an empty map.
"""
import argparse
import json
from pathlib import Path

import aiohttp
from aiohttp import web


async def proxy(request: web.Request) -> web.StreamResponse:
    state = request.app['state']
    session: aiohttp.ClientSession = request.app['session']
    raw = await request.read()
    body = None
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            body = dict(parsed)
            if state['bias']:
                body['logit_bias'] = state['bias']
            raw = json.dumps(body, separators=(',', ':')).encode()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in {'host', 'content-length', 'transfer-encoding'}}
    session: aiohttp.ClientSession = request.app['session']
    async with session.request(request.method, state['upstream'].rstrip('/') + request.path,
                               data=raw if raw else None, headers=headers) as upstream:
        payload = await upstream.read()
        response = web.StreamResponse(status=upstream.status)
        for key, value in upstream.headers.items():
            if key.lower() not in {'content-length', 'transfer-encoding', 'content-encoding'}:
                response.headers[key] = value
        await response.prepare(request)
        await response.write(payload)
        return response


async def _startup(app):
    app['session'] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800))


async def _cleanup(app):
    await app['session'].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--upstream', default='http://127.0.0.1:9292')
    parser.add_argument('--bias-file', type=Path)
    args = parser.parse_args()
    bias = json.loads(args.bias_file.read_text()) if args.bias_file else {}
    app = web.Application()
    app['state'] = {'upstream': args.upstream, 'bias': bias}
    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
    for route in ('/{tail:.*}',):
        app.router.add_route('*', route, proxy)
    web.run_app(app, host='127.0.0.1', port=args.port, print=None)


if __name__ == '__main__':
    main()
