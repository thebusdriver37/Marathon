#!/usr/bin/env python3
"""Check cold/warm greedy media repeatability on a reserved, idle local worker.

Does not manage workers, change routing, or write slot snapshots.
It replaces the selected worker's in-memory prompt cache.
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True, help='Local server or broker upstream URL, without /v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', type=Path, action='append', required=True)
    parser.add_argument('--prompt-file', type=Path, help='Optional long-context user prompt')
    parser.add_argument('--api-key-env', default='HERMES_API_KEY')
    parser.add_argument('--slot', type=int, default=0)
    parser.add_argument('--tokens', type=int, default=768)
    parser.add_argument('--window', type=int, default=6)
    args = parser.parse_args()
    address = urllib.parse.urlsplit(args.base_url)
    if address.scheme != 'http' or address.hostname not in ('localhost', '127.0.0.1', '::1'):
        parser.error('use an explicitly reserved worker on a loopback HTTP endpoint')
    if address.username or address.password or address.query or address.fragment:
        parser.error('do not put credentials, query parameters, or fragments in the URL')
    if args.tokens < 1 or args.slot < 0 or args.window < 1:
        parser.error('tokens and window must be positive; slot must be non-negative')

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {'Content-Type': 'application/json'}
    if key := os.environ.get(args.api_key_env):
        headers['Authorization'] = 'Bearer ' + key

    def request(path, body=None):
        req = urllib.request.Request(args.base_url.rstrip('/') + path,
                                     None if body is None else json.dumps(body).encode(), headers)
        with opener.open(req, timeout=900) as response:
            return json.load(response)

    def run(label, messages, window, cold=False):
        slots = request('/slots')
        if any(slot.get('is_processing') for slot in slots):
            raise RuntimeError('worker is active; do not interrupt another workload')
        result = request('/v1/chat/completions', {
            'model': args.model, 'messages': messages, 'temperature': 0, 'seed': 424242,
            'max_tokens': args.tokens, 'cache_prompt': not cold, 'id_slot': args.slot,
            'chat_template_kwargs': {'enable_thinking': False}, 'speculative.n_max': window,
            'return_tokens': True, 'verbose': True,
        })
        choice = result['choices'][0]
        message = choice['message']
        tokens = result.get('__verbose', {}).get('tokens')
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
            raise RuntimeError('backend did not return raw generated token IDs')
        timings = result.get('timings', {})
        if window == 0 and timings.get('draft_n', 0):
            raise RuntimeError('backend did not honor the zero-draft reference')
        if window > 0 and not timings.get('draft_n', 0):
            raise RuntimeError('media speculation did not activate on this configuration')
        print(json.dumps({
            'label': label, 'message_sha256': hashlib.sha256(
                json.dumps(message, sort_keys=True).encode()).hexdigest(),
            'tokens_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            'token_count': len(tokens), 'finish_reason': choice['finish_reason'],
            'timings': timings, 'usage': result.get('usage'),
        }), flush=True)
        return message, tokens, choice['finish_reason']

    prompt = args.prompt_file.read_text() if args.prompt_file else (
        'Describe these images and their historical significance in about 700 words.'
    )
    content = [{'type': 'text', 'text': prompt}]
    for path in args.image:
        mime = mimetypes.guess_type(path.name)[0]
        if not mime or not mime.startswith('image/'):
            parser.error('image files need a recognized image extension')
        data = 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()
        content.append({'type': 'image_url', 'image_url': {'url': data}})
    messages = [{'role': 'system', 'content': 'Answer directly and accurately.'},
                {'role': 'user', 'content': content}]
    reference = run('cold-reference', messages, args.window, cold=True)
    for repeat in range(2):
        if run('warm-repeat-' + str(repeat), messages, args.window) != reference:
            raise RuntimeError('greedy image output changed between cold and warm requests')
    messages.extend([reference[0], {'role': 'user', 'content':
                                'Explain what the main object in the first image is used for.'}])
    reference = run('followup-reference', messages, args.window)
    if run('followup-repeat', messages, args.window) != reference:
        raise RuntimeError('greedy image follow-up differs')
    print('PASS: all greedy image token IDs, messages, and stop reasons match, including the follow-up.')


if __name__ == '__main__':
    main()
