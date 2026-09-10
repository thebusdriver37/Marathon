"""Bound one observed failure mode: repeated literal echo-only shell calls."""

import json
import shlex


ECHO_LIMIT = 8
RECOVERY_NOTE = (
    'Marathon recovery: repeated echo-only shell calls were omitted from this model request. '
    'They printed markers; they did not start helpers or advance the task. '
    'Do not repeat that loop. To coordinate helpers, call the available '
    'collaboration__list_agents or collaboration__followup_task function directly. '
    'Use the saved helper identities in your instructions. '
    'Use followup_task to wake an idle helper; send_message only queues a message and does not start it. '
    'If a required tool is unavailable, report that instead of pretending to call it.'
)


def echo_marker(item):
    if item.get('type') != 'function_call' or item.get('name') != 'exec_command':
        return False
    try:
        arguments = json.loads(item.get('arguments', ''))
    except (ValueError, TypeError):
        return False
    if not isinstance(arguments, dict) or not isinstance(arguments.get('cmd'), str):
        return False
    cmd = arguments['cmd']
    # Literal echo only: exclude expansion, redirection, pipelines, command
    # substitution, flags, and long output. Quoted marker text is still a no-op.
    if len(cmd) > 240 or any(character in cmd for character in '$`\\\n*?['):
        return False
    try:
        tokens = list(shlex.shlex(cmd, posix=True, punctuation_chars=True))
    except ValueError:
        return False
    return (len(tokens) >= 2 and tokens[0] == 'echo'
            and not tokens[1].startswith('-')
            and not any(token and all(c in ';&|<>()' for c in token) for token in tokens[1:]))


def recover_echo_history(payload, *, include_note=True):
    """Return a request-only cleanup and the current turn's echo streak.

    Stored rollouts remain intact. Only complete call/result pairs in long runs
    are omitted; useful commands, short runs, and all user messages remain.
    Compaction uses the same cleanup without an instruction to contact helpers.
    """
    items = payload.get('input')
    if not isinstance(items, list):
        return payload, 0, 0
    runs, run = [], []
    streak = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if item.get('role') == 'user' or item.get('type') == 'agent_message':
            if run:
                runs.append(run)
                run = []
            streak = 0
        elif item.get('type') in ('function_call', 'custom_tool_call'):
            if echo_marker(item):
                run.append(index)
                streak += 1
            else:
                if run:
                    runs.append(run)
                    run = []
                streak = 0
    if run:
        runs.append(run)
    outputs = {item.get('call_id'): index for index, item in enumerate(items)
               if isinstance(item, dict) and item.get('type') == 'function_call_output'}
    omitted = set()
    omitted_calls = 0
    for run in runs:
        if len(run) < ECHO_LIMIT:
            continue
        for index in run:
            call_id = items[index].get('call_id')
            if call_id and call_id in outputs:
                omitted_calls += 1
                omitted.update((index, outputs[call_id]))
                previous = index - 1
                while previous >= 0 and isinstance(items[previous], dict) and items[previous].get('type') == 'reasoning':
                    omitted.add(previous)
                    previous -= 1
    if not omitted:
        return payload, 0, streak
    cleaned = [item for index, item in enumerate(items) if index not in omitted]
    # A single bounded correction at the end of the request, after the old
    # examples, is more useful than repeating a warning for every omitted call.
    if include_note:
        cleaned.append({'type': 'message', 'role': 'developer', 'content': RECOVERY_NOTE})
    return {**payload, 'input': cleaned}, omitted_calls, streak
