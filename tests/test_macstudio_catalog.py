"""The Studio example must remain selectable through the existing external route."""
from pathlib import Path
import unittest
from unittest import mock

from marathon_app.catalog import external_models, load_catalog
from test_router_context import router_module as router


class StudioCatalogTests(unittest.TestCase):
    def test_example_reaches_frontend_catalog_without_local_worker(self):
        path = Path(__file__).resolve().parents[1] / 'config/examples/macstudio-deepseek.toml'
        configured = external_models(load_catalog(path))
        with mock.patch.object(router, 'external_models', return_value=configured):
            profiles = router._external_model_profiles()
        profile = profiles['macstudio-deepseek']
        self.assertTrue(profile.external)
        self.assertFalse(profile.supports_slots)
        self.assertEqual(profile.model_paths, ())
        self.assertEqual(profile.target, 'http://127.0.0.1:11235')
        self.assertEqual(profile.reasoning_transport, 'responses')
        self.assertIsNone(profile.temperature)
        self.assertIsNone(profile.default_reasoning_level)
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
