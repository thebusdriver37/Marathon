"""The Studio example must remain selectable through the existing external route."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from marathon_app.catalog import external_models, load_catalog
from test_router_context import router_module as router

EXAMPLE = Path(__file__).resolve().parents[1] / 'config/examples/macstudio-deepseek.toml'


def _studio_profiles():
    configured = external_models(load_catalog(EXAMPLE))
    with mock.patch.object(router, 'external_models', return_value=configured):
        return router._external_model_profiles()


class StudioCatalogTests(unittest.TestCase):
    def test_example_reaches_frontend_catalog_without_local_worker(self):
        profiles = _studio_profiles()
        profile = profiles['macstudio-deepseek']
        self.assertTrue(profile.external)
        self.assertFalse(profile.supports_slots)
        self.assertEqual(profile.model_paths, ())
        self.assertEqual(profile.target, 'http://127.0.0.1:11235')
        self.assertEqual(profile.reasoning_transport, 'responses')
        self.assertEqual(profile.temperature, 1.0)
        self.assertEqual(profile.default_reasoning_level, 'high')
        self.assertTrue(profile.native_instruction_roles)
        state = object.__new__(router.RouterState)
        state.available_profiles = profiles
        state._refresh_profiles = lambda: profiles
        with mock.patch.object(router, '_base_instructions', return_value='Synthetic instructions'):
            payload = state.model_catalog()
        model, = payload['models']
        self.assertEqual(model['slug'], 'macstudio-deepseek')
        self.assertEqual(model['visibility'], 'list')
        self.assertEqual(model['context_window'], 262144)
        self.assertEqual(model['auto_compact_token_limit'], 229376)
        self.assertEqual(model['truncation_policy']['limit'], 221184)
        self.assertTrue(model['supports_parallel_tool_calls'])
        self.assertEqual(model['default_reasoning_level'], 'high')

    def test_default_request_enables_server_thinking(self):
        # mlx-serve leaves DeepSeek in chat mode unless a reasoning effort arrives.
        profile = _studio_profiles()['macstudio-deepseek']
        request = {'model': profile.alias, 'input': []}
        router.normalize_responses_request(request, profile)
        self.assertEqual(request['reasoning'], {'effort': 'high'})
        request = {'model': profile.alias, 'input': [], 'reasoning': {'effort': 'none'}}
        router.normalize_responses_request(request, profile)
        self.assertEqual(request['reasoning'], {'effort': 'none'})

    def test_native_instruction_roles_keep_developer_messages(self):
        profile = _studio_profiles()['macstudio-deepseek']
        developer = {'type': 'message', 'role': 'developer',
                     'content': [{'type': 'input_text', 'text': 'Base instructions'}]}
        user = {'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': 'Task'}]}
        request = {'model': profile.alias, 'input': [developer, user]}
        router.normalize_responses_request(request, profile)
        self.assertEqual([item['role'] for item in request['input']], ['developer', 'user'])
        self.assertEqual(request['input'][0]['content'][0]['text'], 'Base instructions')

        legacy = router.ModelProfile(**{**profile.__dict__, 'native_instruction_roles': False})
        request = {'model': profile.alias, 'input': [developer, user]}
        router.normalize_responses_request(request, legacy)
        self.assertEqual([item['role'] for item in request['input']], ['user', 'user'])
        self.assertTrue(request['input'][0]['content'][0]['text'].startswith(
            '<marathon_context role="developer">'))

    def test_output_cap_tools_and_instructions(self):
        profile = _studio_profiles()['macstudio-deepseek']
        self.assertEqual(router._max_output_tokens(profile), 32768)
        tools = [{'type': 'function', 'name': name, 'parameters': {'type': 'object', 'properties': {}}}
                 for name in ('exec_command', 'apply_patch', 'view_image', 'multi_agent_v1__spawn_agent', 'get_goal')]
        request = {'model': profile.alias, 'input': [], 'tools': tools}
        router.normalize_responses_request(request, profile)
        self.assertEqual([tool['name'] for tool in request['tools']], ['exec_command', 'apply_patch', 'get_goal'])
        base = router._profile_base_instructions('UPSTREAM CODEX PROMPT', profile)
        self.assertNotIn('UPSTREAM CODEX PROMPT', base)
        self.assertIn('You are a coding agent running in Marathon', base)
        self.assertIn(router.MARATHON_RUNTIME_INSTRUCTIONS, base)
        self.assertIn(router.MARATHON_PARALLEL_TOOL_INSTRUCTIONS, base)

    def test_recovery_disables_thinking_only_when_none_is_declared(self):
        profile = _studio_profiles()['macstudio-deepseek']
        request = {'reasoning': {'effort': 'high', 'summary': 'auto'}}
        router._disable_responses_reasoning_for_recovery(request, profile)
        self.assertEqual(request['reasoning'], {'effort': 'none', 'summary': 'auto'})
        no_none = router.ModelProfile(**{**profile.__dict__, 'supported_reasoning_levels': (('high', 'High'),)})
        request = {'reasoning': {'effort': 'high'}}
        router._disable_responses_reasoning_for_recovery(request, no_none)
        self.assertEqual(request['reasoning'], {'effort': 'high'})

    def test_other_profiles_keep_router_defaults(self):
        profile = router.ModelProfile(**{**_studio_profiles()['macstudio-deepseek'].__dict__,
                                         'max_output_tokens': None, 'disabled_tools': (),
                                         'instructions_file': None})
        self.assertEqual(router._max_output_tokens(profile), router.DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertEqual(router._profile_base_instructions('UPSTREAM', profile).split('\n\n')[0], 'UPSTREAM')

    def test_native_instruction_roles_require_responses_transport(self):
        text = EXAMPLE.read_text().replace(
            'reasoning_transport = "responses"', 'reasoning_transport = "chat_template"')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'catalog.toml'
            path.write_text(text)
            with self.assertRaises(ValueError):
                external_models(load_catalog(path))


if __name__ == '__main__':
    unittest.main()
