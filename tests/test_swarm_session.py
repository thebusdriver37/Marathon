"""Swarm resumes retain their orchestration mode and saved agent identities."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from marathon_app import __main__ as main
from marathon_app.swarm_session import resume_id, saved_swarm, saved_agent_paths

ROOT = '123e4567-e89b-12d3-a456-426614174000'
HELPER = '123e4567-e89b-12d3-a456-426614174001'


class SwarmSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.home = self.output / 'codex-home'
        (self.home / 'sessions').mkdir(parents=True)
        self.rollout(ROOT, {'id': ROOT, 'source': 'exec'})
        (self.output / 'events.jsonl').write_text(json.dumps({
            'event': 'swarm.started', 'data': {'instances': ['one', 'two', 'three']}}) + '\n')
        environment = mock.patch.dict(os.environ, {'MARATHON_CODEX_HOME': str(self.home)})
        environment.start()
        self.addCleanup(environment.stop)

    def rollout(self, thread, metadata):
        (self.home / 'sessions' / f'rollout-{thread}.jsonl').write_text(
            json.dumps({'type': 'session_meta', 'payload': metadata}) + '\n')

    def test_ordinary_and_headless_resume_restore_swarm_before_leasing_single_worker(self):
        for arguments in (['resume', ROOT], ['exec', 'resume', ROOT, 'Continue']):
            with (
                self.subTest(arguments=arguments),
                mock.patch('marathon_app.swarm.run_swarm', return_value=0) as swarm,
                mock.patch.object(main, 'automatic_launch_instance') as single,
            ):
                self.assertEqual(main.main(arguments), 0)
                swarm.assert_called_once_with(arguments, session_home=self.home)
                single.assert_not_called()

    def test_legacy_worker_count_and_manifest_override(self):
        self.assertEqual(saved_swarm(self.home), {'version': 1, 'agents': 3, 'workers': 3})
        settings = {'version': 1, 'agents': 2, 'workers': 1}
        (self.output / 'swarm.json').write_text(json.dumps(settings))
        self.assertEqual(saved_swarm(self.home), settings)
        (self.output / 'swarm.json').write_text('{"version":1,"agents":2,"workers":3}')
        with self.assertRaisesRegex(ValueError, 'worker count'):
            saved_swarm(self.home)

    def test_restore_paths_from_metadata_not_last_message_recipient(self):
        self.rollout(HELPER, {'id': HELPER, 'source': {'subagent': {}},
                             'parent_thread_id': ROOT, 'agent_path': '/root/helper'})
        self.assertEqual(saved_agent_paths(self.home, ROOT), {ROOT: '/root', HELPER: '/root/helper'})
        with self.assertRaisesRegex(ValueError, 'lead'):
            saved_agent_paths(self.home, HELPER)

    def test_resume_requires_an_explicit_uuid(self):
        self.assertEqual(resume_id(['resume', ROOT]), ROOT)
        self.assertEqual(resume_id(['exec', 'resume', ROOT, 'Continue']), ROOT)
        self.assertIsNone(resume_id(['exec', 'New task']))
        for arguments in (['resume'], ['resume', '--last'], ['resume', 'partial-id']):
            with self.assertRaisesRegex(ValueError, 'complete session UUID'):
                resume_id(arguments)

    def test_swarm_picker_cannot_silently_start_single_agent_mode(self):
        for arguments in (['resume'], ['resume', '--last'], ['exec', 'resume', '--last']):
            with (self.subTest(arguments=arguments),
                  mock.patch.object(main, 'automatic_launch_instance') as single):
                self.assertEqual(main.main(arguments), 2)
                single.assert_not_called()
