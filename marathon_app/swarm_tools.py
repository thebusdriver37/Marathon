"""Translate Codex namespaces to llama.cpp function names and restore replies."""

import copy
import json


def flatten_request(payload, names):
    if not isinstance(payload, dict):
        raise ValueError('Swarm requests must be JSON objects.')
    payload = copy.deepcopy(payload)
    tools = []
    for tool in payload.get('tools', []):
        if tool.get('type') != 'namespace':
            tools.append(tool)
            continue
        namespace = tool['name']
        for member in tool.get('tools', []):
            if member.get('type') != 'function':
                raise ValueError('Swarm namespaces currently support function tools only.')
            alias = f'{namespace}__{member["name"]}'
            names[alias] = (namespace, member['name'])
            tools.append({**member, 'name': alias})
    if 'tools' in payload:
        aliases = [tool.get('name') for tool in tools if tool.get('name')]
        if len(aliases) != len(set(aliases)):
            raise ValueError('Swarm tool names collide after namespace translation.')
        payload['tools'] = tools
    reverse = {value: key for key, value in names.items()}
    for index, item in enumerate(payload.get('input', [])):
        if isinstance(item, dict) and item.get('type') == 'agent_message':
            # V2 stores the model-supplied message in encrypted_content. For
            # this local-only provider that value is the original plain text.
            content = [{'type': 'input_text', 'text':
                        f'Agent message from {item["author"]} to {item["recipient"]}:\n'}]
            for part in item.get('content', []):
                if part.get('type') == 'input_text':
                    content.append(part)
                elif part.get('type') == 'encrypted_content':
                    content.append({'type': 'input_text', 'text': part['encrypted_content']})
                else:
                    raise ValueError('Unsupported local agent message content.')
            payload['input'][index] = {'type': 'message', 'role': 'user', 'content': content}
            continue
        if isinstance(item, dict) and item.get('type') == 'function_call':
            alias = reverse.get((item.get('namespace'), item.get('name')))
            if alias:
                item['name'] = alias
                item.pop('namespace', None)
    return payload


def restore_calls(value, names):
    if isinstance(value, dict):
        value = {key: restore_calls(item, names) for key, item in value.items()}
        if value.get('type') == 'function_call' and value.get('name') in names:
            value['namespace'], value['name'] = names[value['name']]
    elif isinstance(value, list):
        value = [restore_calls(item, names) for item in value]
    return value


async def translated_sse(content, names):
    pending = b''
    async for chunk in content.iter_any():
        pending += chunk
        if len(pending) > 64 * 1024**2:
            raise ValueError('Swarm SSE event exceeds 64 MiB.')
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            if line.startswith(b'data:'):
                data = line[5:].strip()
                if data and data != b'[DONE]':
                    line = b'data: ' + json.dumps(restore_calls(json.loads(data), names)).encode()
            yield line + b'\n'
    if pending:
        yield pending
