"""Inkling selection and template effort regressions; synthetic inputs only."""
import asyncio
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from marathon_app import catalog
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/routers'))
import codex_local_router as router


class InklingTests(unittest.TestCase):
    def profile(self):
        return router.ModelProfile(slug='inkling-2-mini', alias='inkling', display_name='Inkling',
            description='Synthetic', model_paths=(), target='http://127.0.0.1:19001',
            context_window=32768, auto_compact_token_limit=29491, truncation_limit=29491,
            reasoning_transport='chat_template_effort', default_reasoning_level='medium',
            supported_reasoning_levels=tuple((x, x) for x in ('none', 'low', 'medium', 'xhigh')))

    def test_family_selection_and_bounded_profiles(self):
        raw = catalog.tomllib.loads((ROOT / 'config/runtime_catalog.toml').read_text())
        family = catalog._family_for(Path('/models/Inkling-2-Mini-IQ4_NL.gguf'), catalog.families(raw))
        self.assertEqual(family.id, 'inkling-2-mini')
        self.assertEqual(family.reasoning_transport, 'chat_template_effort')
        self.assertEqual(family.default_profile, 'inkling-32k')
        self.assertEqual([(p.context, p.ubatch) for p in family.profiles], [(32768, 64), (262144, 64)])
        self.assertTrue(all(p.tool_thinking_budget is None for p in family.profiles))
        self.assertTrue(catalog._model_sidecar(Path('dspark-Inkling-2-Mini-Q8_0.gguf')))

    def test_all_efforts_survive_normalization_including_none(self):
        for effort in ('none', 'low', 'medium', 'xhigh'):
            request = {'reasoning': {'effort': effort},
                       'chat_template_kwargs': {'enable_thinking': True, 'keep': 'value'}}
            router._apply_reasoning_effort(request, self.profile())
            self.assertEqual(request['chat_template_kwargs'], {'reasoning_effort': effort, 'keep': 'value'})
        request = {}
        router._apply_reasoning_effort(request, self.profile())
        self.assertEqual(request['chat_template_kwargs'], {'reasoning_effort': 'medium'})

    def test_custom_and_external_profiles_preserve_transport(self):
        with mock.patch.dict(router.os.environ, {'MARATHON_MODEL_PATH': '/synthetic.gguf',
                'MARATHON_MODEL_REASONING_TRANSPORT': 'chat_template_effort'}, clear=True):
            self.assertEqual(router._custom_model_profile(ROOT).reasoning_transport, 'chat_template_effort')
        raw = {'id':'inkling', 'model':'inkling', 'base_url':'http://127.0.0.1:19001/v1',
               'context':32768, 'reasoning_transport':'chat_template_effort'}
        configured = catalog.external_models({'external_models':[raw]})
        with mock.patch.object(router, 'external_models', return_value=configured):
            self.assertEqual(router._external_model_profiles()['inkling'].reasoning_transport, 'chat_template_effort')
        for transport in ('typo', 'responses'):
            family = {'id':'bad','reasoning_transport':transport}
            with self.assertRaisesRegex(ValueError, 'reasoning_transport'):
                catalog.families({'families':[family]})

    def test_stalled_recovery_requests_lowest_inkling_effort(self):
        state = object.__new__(router.RouterState)
        state.web_search_settings = SimpleNamespace(max_iterations=3)
        state.telemetry = mock.Mock()
        stalled = {'output':[{'type':'reasoning'}], 'usage':{'input_tokens':100, 'output_tokens':8192}}
        recovered = {'output':[{'type':'function_call','name':'exec_command','call_id':'call1','arguments':'{}'}],
                     'usage':{'input_tokens':120,'output_tokens':30}}
        state._request_json = mock.AsyncMock(side_effect=[stalled,recovered])
        asyncio.run(state._run_responses_loop(profile=self.profile(), forward_request={
            'input':[], 'tools':[{'type':'function','name':'exec_command'}], 'max_output_tokens':8192,
            'chat_template_kwargs':{'reasoning_effort':'medium'}}, web_search_enabled=False))
        request = state._request_json.await_args_list[1].args[3]
        self.assertEqual(request['chat_template_kwargs'], {'reasoning_effort':'none'})
        self.assertEqual(request['tool_choice'], 'required')

if __name__ == '__main__': unittest.main()
