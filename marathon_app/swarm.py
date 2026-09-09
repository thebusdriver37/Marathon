"""Local experiment: native Codex collaboration across leased Marathon workers."""

import argparse
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import shutil
import signal
import threading
import time
from types import SimpleNamespace
import uuid

from .catalog import backend_for, discover_models
from .frontends import run_codex
from .codex_home import session_home_for_id
from .swarm_session import resume_id, saved_swarm, saved_agent_paths
from .runtime import Runtime, load_selection
from .swarm_gateway import SwarmGateway


def worker_main(connection, model, profile, name):
    # Each Runtime owns its signals, router, lease, and cleanup in its own process.
    os.setsid()
    runtime = Runtime(model, profile, name)
    try:
        runtime.start(lambda message: connection.send({'progress': message}))
        connection.send({'ready': {
            'router_url': runtime.router_url, 'router_token': runtime.router_token,
            'catalog_file': str(runtime.catalog_file), 'instance': name,
        }})
        connection.recv()  # Parent shutdown or EOF releases only this worker.
    except EOFError:
        pass
    except Exception as error:
        with contextlib.suppress(BrokenPipeError, EOFError):
            connection.send({'error': str(error)})
    finally:
        runtime.cleanup()
        connection.close()


@contextlib.contextmanager
def leased_workers(model, profile, count, run_id):
    context = multiprocessing.get_context('spawn')
    processes = []
    try:
        ready = []
        for index in range(count):
            parent, child = context.Pipe()
            name = f'swarm-{run_id}-{index}'
            process = context.Process(target=worker_main, args=(child, model, profile, name))
            process.start()
            child.close()
            processes.append((process, parent))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if parent.poll(0.25):
                    try:
                        message = parent.recv()
                    except EOFError as error:
                        raise RuntimeError(f'Swarm worker {index + 1} closed during startup') from error
                    if 'error' in message:
                        raise RuntimeError(message['error'])
                    if 'ready' in message:
                        ready.append(message['ready'])
                        break
                    print(f'Worker {index + 1}: {message["progress"]}', flush=True)
                if not process.is_alive():
                    raise RuntimeError(f'Swarm worker {index + 1} exited during startup')
            else:
                raise RuntimeError(f'Swarm worker {index + 1} startup timed out')
        yield ready
    finally:
        for process, connection in processes:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                connection.send('stop')
            connection.close()
        for process, connection in processes:
            process.join(timeout=120)
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)


class SwarmFrontend:
    def __init__(self, model, profile, output):
        self.model = model
        self.profile = profile
        self.instance = SimpleNamespace(name=None)
        self.session_home = output / 'codex-home'
        self.router_token = secrets.token_urlsafe(32)
        self.output = output
        self.log_lock = threading.Lock()

    def record(self, event, data=None, level='info'):
        with self.log_lock, (self.output / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps({'time': time.time(), 'event': event,
                                     'data': data or {}, 'level': level}) + '\n')

    @contextlib.contextmanager
    def frontend_signals(self):
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)


def run_swarm(arguments, *, session_home=None):
    parser = argparse.ArgumentParser(prog='marathon swarm', description=__doc__)
    parser.add_argument('--agents', type=int, choices=(2, 3))
    parser.add_argument('--workers', type=int, choices=(1, 2, 3),
                        help='GPU workers; defaults to one per agent. Use 1 for a serial baseline.')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('codex_args', nargs=argparse.REMAINDER,
                        help='Optional Codex arguments, for example: exec "task"')
    args = parser.parse_args(arguments)
    session_id = resume_id(args.codex_args)
    saved = None
    agent_paths = {}
    if session_id:
        session_home = session_home or session_home_for_id(session_id)
        saved = saved_swarm(session_home)
        if saved is None:
            raise ValueError('This conversation was not created by marathon swarm.')
        agent_paths = saved_agent_paths(session_home, session_id)
    args.agents = args.agents or (saved['agents'] if saved else 3)
    worker_count = args.workers or (saved['workers'] if saved else args.agents)
    if worker_count > args.agents:
        parser.error('--workers cannot exceed --agents')
    from .ui import _initial_selection

    models = discover_models()
    if not models:
        raise ValueError('Install and select a Marathon model before starting a swarm.')
    selection = _initial_selection(models, load_selection())
    backend = backend_for(selection.model, selection.profile)
    if backend.kind != 'llama_swap_pool' or len(backend.pool_models) < worker_count:
        raise ValueError('This local experiment needs a selected llama-swap pool with enough workers.')
    run_id = uuid.uuid4().hex[:10]
    output = (args.output_dir or Path('.marathon/swarms') / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    settings = {'version': 1, 'agents': args.agents, 'workers': worker_count}
    (output / 'swarm.json').write_text(json.dumps(settings) + '\n')
    print(f'Experimental swarm evidence: {output}', flush=True)
    frontend = SwarmFrontend(selection.model, selection.profile, output)
    if session_home is not None:
        frontend.session_home = Path(session_home)
        print(f'Resuming swarm with {args.agents} agents across {worker_count} workers.', flush=True)
    guidance = (
        f'You lead a local team with {args.agents - 1} helper agents. '
        'Use the native multi-agent tools to delegate independent tasks when useful. '
        f'Spawn at most {args.agents - 1} helpers in this session. '
        'Contact helpers with followup_task: it starts idle helpers and delivers messages to running ones. '
        'Helpers must not spawn more agents. Keep all agents on the inherited model. '
        'Give helpers disjoint file ownership or separate git worktrees before parallel edits. '
        'Do useful work while helpers run, then inspect their results and run integration tests. '
        'You own the final result. Do not commit or merge helper changes without checking them. '
        'After resuming, use list_agents and reuse the saved helpers with followup_task. '
        'Spawn only if a helper slot is actually empty. '
        'Shell commands cannot spawn native helpers; use the collaboration tools directly.'
    )
    if session_id:
        helpers = sorted(path for path in agent_paths.values() if path != '/root')
        guidance += (
            f' Saved helper identities: {json.dumps(helpers)}. '
            'These helpers may be idle and absent from list_agents; try followup_task '
            'with their saved identities before attempting to spawn replacements.'
        )
    with leased_workers(selection.model, selection.profile, worker_count, run_id) as workers:
        frontend.catalog_file = output / 'model-catalog.json'
        shutil.copyfile(workers[0]['catalog_file'], frontend.catalog_file)
        frontend.record('swarm.started', {'instances': [worker['instance'] for worker in workers]})
        with SwarmGateway(workers, frontend.router_token, frontend.record, max_agents=args.agents,
                          agent_paths=agent_paths) as gateway:
            frontend.router_url = gateway.url
            overrides = [
                '-c', 'features.multi_agent=true',
                '-c', 'features.multi_agent_v2.enabled=true',
                '-c', f'features.multi_agent_v2.max_concurrent_threads_per_session={args.agents}',
                '-c', 'features.multi_agent_v2.expose_spawn_agent_model_overrides=false',
                '-c', 'features.multi_agent_v2.subagent_developer_instructions="Complete your assigned task and report to the lead. Do not spawn agents. Edit only files assigned to you. Keep reports concise and distinguish verified results from assumptions."',
                '-c', 'model_providers.marathon-local.supports_websockets=false',
                '-c', f'developer_instructions={json.dumps(guidance)}',
            ]
            return run_codex(frontend, [*overrides, *args.codex_args])
