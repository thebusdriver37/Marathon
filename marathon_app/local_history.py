"""Translate saved local collaboration messages for llama.cpp replay."""

import copy


def normalize_local_history(items):
    if not isinstance(items, list):
        return items
    result = []
    for original in items:
        if not isinstance(original, dict):
            result.append(original)
            continue
        item = original
        if item.get('type') == 'agent_message':
            content = [{'type': 'input_text', 'text':
                        f'Agent message from {item["author"]} to {item["recipient"]}:\n'}]
            for part in item.get('content', []):
                if part.get('type') == 'input_text':
                    content.append(copy.deepcopy(part))
                elif part.get('type') == 'encrypted_content':
                    # Local Qwen puts its literal message here. This does not
                    # decrypt encrypted messages from cloud providers.
                    content.append({'type': 'input_text', 'text': part['encrypted_content']})
                else:
                    raise ValueError('Unsupported local agent message content.')
            item = {'type': 'message', 'role': 'user', 'content': content}
        elif item.get('type') == 'function_call' and item.get('namespace'):
            # Resume may no longer offer the historical collaboration tools.
            # Their call IDs and results must still remain paired in history.
            item = {**item, 'name': f'{item["namespace"]}__{item["name"]}'}
            item.pop('namespace')
        result.append(item)
    return result
