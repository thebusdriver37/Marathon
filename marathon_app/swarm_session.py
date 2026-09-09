"""Recover a swarm's launch settings without copying Codex's agent graph."""

import json
from pathlib import Path
import uuid


def resume_id(arguments):
    args = list(arguments)
    if args[:1] == ['exec']:
        args = args[1:]
    if args[:1] != ['resume']:
        return None
    if len(args) < 2:
        raise ValueError('Swarm resume requires the complete session UUID.')
    try:
        return str(uuid.UUID(args[1]))
    except ValueError as error:
        raise ValueError('Swarm resume requires the complete session UUID.') from error


def saved_swarm(home):
    """Recognize old event-only runs as well as new persisted launch settings."""
    output = Path(home).parent
    manifest = output / 'swarm.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get('version') != 1 or data.get('agents') not in (2, 3) or data.get('workers') not in (1, 2, 3):
            raise ValueError(f'Invalid swarm settings: {manifest}')
        if data['workers'] > data['agents']:
            raise ValueError(f'Invalid swarm worker count: {manifest}')
        return data
    events = output / 'events.jsonl'
    if Path(home).name != 'codex-home' or not events.is_file():
        return None
    with events.open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get('event') == 'swarm.started':
                # The original experiment defaulted to three agents. Its event
                # log recorded worker count but did not persist agent count.
                workers = len(event.get('data', {}).get('instances', []))
                if workers not in (1, 2, 3):
                    raise ValueError('Saved swarm has an invalid worker count.')
                return {'version': 1, 'agents': 3, 'workers': workers}
    return None


def saved_agent_paths(home, session_id):
    matches = [path for category in ('sessions', 'archived_sessions')
               for path in (Path(home) / category).rglob(f'*-{session_id}.jsonl')]
    if len(matches) != 1:
        raise ValueError(f'Expected one saved rollout for swarm session {session_id}.')
    with matches[0].open() as handle:
        metadata = json.loads(handle.readline()).get('payload', {})
    if metadata.get('id') != session_id or isinstance(metadata.get('source'), dict):
        raise ValueError('Resume the swarm lead, not an individual helper.')
    paths = {session_id: '/root'}
    for category in ('sessions', 'archived_sessions'):
        for path in (Path(home) / category).rglob('*.jsonl'):
            with path.open() as handle:
                item = json.loads(handle.readline()).get('payload', {})
            if item.get('parent_thread_id') == session_id and item.get('agent_path'):
                paths[item['id']] = item['agent_path']
    return paths
