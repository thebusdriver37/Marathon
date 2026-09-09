"""Keep useful shell work while removing the observed swarm echo feedback loop."""

import copy
import json
import unittest

from marathon_app.echo_recovery import ECHO_LIMIT, echo_marker, recover_echo_history


def loop(count=ECHO_LIMIT):
    items = []
    for index in range(count):
        items.extend([
            {'type': 'reasoning', 'content': [{'type': 'text', 'text': 'Calling helpers now.'}]},
            {'type': 'function_call', 'name': 'exec_command', 'call_id': f'echo-{index}',
             'arguments': json.dumps({'cmd': 'echo "calling helpers now"'})},
            {'type': 'function_call_output', 'call_id': f'echo-{index}', 'output': 'calling helpers now'},
        ])
    return items


class EchoRecoveryTests(unittest.TestCase):
    def test_old_loop_is_removed_without_touching_saved_input_or_useful_work(self):
        useful = {'type': 'function_call', 'name': 'exec_command', 'call_id': 'test',
                  'arguments': '{"cmd":"python3 -m unittest"}'}
        result = {'type': 'function_call_output', 'call_id': 'test', 'output': '3 tests passed'}
        task = {'type': 'message', 'role': 'user', 'content': 'Continue with the helpers'}
        payload = {'input': [useful, result, *loop(32), task]}
        original = copy.deepcopy(payload)
        cleaned, count, streak = recover_echo_history(payload)
        self.assertEqual((count, streak), (32, 0))
        self.assertEqual(cleaned['input'][:3], [useful, result, task])
        self.assertEqual(len(cleaned['input']), 4)
        self.assertIn('collaboration__followup_task', cleaned['input'][-1]['content'])
        self.assertEqual(payload, original)
        self.assertEqual(recover_echo_history(cleaned), (cleaned, 0, 0))

    def test_current_loop_is_detected_and_new_user_turn_resets_it(self):
        payload = {'input': [{'role': 'user', 'content': 'Use helpers'}, *loop()]}
        self.assertEqual(recover_echo_history(payload)[2], ECHO_LIMIT)
        payload['input'].append({'role': 'user', 'content': 'Try again'})
        self.assertEqual(recover_echo_history(payload)[2], 0)

    def test_real_tool_progress_breaks_the_echo_streak(self):
        payload = {'input': [*loop(ECHO_LIMIT - 1),
                            {'type': 'function_call', 'name': 'collaboration__list_agents',
                             'arguments': '{}'}, *loop(ECHO_LIMIT - 1)]}
        self.assertEqual(recover_echo_history(payload), (payload, 0, ECHO_LIMIT - 1))

    def test_commands_with_side_effects_or_expansions_are_not_markers(self):
        for cmd in ('echo $HOME', 'echo "$(run)"', 'echo hi > file', 'echo yes && run',
                    'echo --help', 'echo hi; run', 'echo hi\nrun', 'printf hi', 'echo -n ready'):
            with self.subTest(cmd=cmd):
                self.assertFalse(echo_marker({'type': 'function_call', 'name': 'exec_command',
                                              'arguments': json.dumps({'cmd': cmd})}))
        for cmd in ('echo go', 'echo "calling helpers"', "echo 'calling helpers'"):
            self.assertTrue(echo_marker({'type': 'function_call', 'name': 'exec_command',
                                         'arguments': json.dumps({'cmd': cmd})}))

    def test_incomplete_call_is_preserved(self):
        payload = {'input': loop()[:-1]}
        cleaned, count, _ = recover_echo_history(payload)
        self.assertIn(payload['input'][-1], cleaned['input'])
        self.assertEqual(count, ECHO_LIMIT - 1)
