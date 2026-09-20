"""Picker identities track configured weights, not stable routing aliases."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from marathon_app.catalog import Backend
from marathon_app.model_identity import pool_identity
from test_router_context import fixture_profile, router_module


class ModelIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "broker.yaml"
        self.backend = Backend("pool", "Pool", None, kind="llama_swap_pool",
                               pool_models=("one", "two"), identity_config=self.path)

    def configure(self, target="Swift-Merge-IQ4_XS.gguf", second=None):
        self.path.write_text(
            'models:\n  one: &worker\n    cmd: >-\n'
            f'      docker run --volume /weights:/models:ro --model /models/{target}'
            ' --spec-draft-model /draft/Marathon-R32.gguf\n'
            '  two:\n    <<: *worker\n' + (f'    cmd: server --model {second}\n' if second else '')
        )

    def test_identity_changes_with_weights_and_keeps_drafter_in_description(self):
        for name in ("Swift-Merge-IQ4_XS.gguf", "Original-Q8_0.gguf"):
            self.configure(name)
            label, description = pool_identity(self.backend)
            self.assertEqual(label, Path(name).stem.replace("-", " ") + " (Local)")
            self.assertIn("Marathon-R32.gguf", description)

    def test_invalid_and_mixed_config_do_not_claim_old_identity(self):
        self.configure(second="other.gguf")
        self.assertIn("mixed identities", pool_identity(self.backend)[0])
        self.path.write_text("models: [broken")
        self.assertIn("identity unavailable", pool_identity(self.backend)[0])
        self.path.unlink()
        self.assertIn("identity unavailable", pool_identity(self.backend)[0])

    async def test_http_picker_refreshes_names_and_hides_only_duplicate_pool_alias(self):
        self.configure()
        local = replace(fixture_profile(), slug="local")
        alias = replace(local, slug="old-alias", external=True)
        remote = replace(local, slug="spark", display_name="Flash Next NVFP4 (DGX Spark)", external=True)
        state = object.__new__(router_module.RouterState)
        state.pool_backend = self.backend
        state.pool_slugs = {"local", "old-alias"}
        state.pool_slug = "local"
        state.available_profiles = {p.slug: p for p in (local, alias, remote)}
        state._refresh_profiles = lambda: state.available_profiles
        app = web.Application()
        app["state"] = state
        app.router.add_get("/v1/models", router_module.handle_models)
        with mock.patch.object(router_module, "backends", return_value={"pool": self.backend}):
            async with TestClient(TestServer(app)) as client:
                for name in ("Swift-Merge-IQ4_XS.gguf", "Original-Q8_0.gguf"):
                    self.configure(name)
                    response = await client.get("/v1/models")
                    self.assertEqual(response.status, 200)
                    models = (await response.json())["models"]
                    visible = [m for m in models if m["visibility"] == "list"]
                    self.assertEqual([m["slug"] for m in visible], ["local", "spark"])
                    self.assertEqual(visible[0]["display_name"], Path(name).stem.replace("-", " ") + " (Local)")
                    self.assertEqual(visible[1]["display_name"], remote.display_name)
                    self.assertIn("old-alias", state.available_profiles)


if __name__ == "__main__":
    unittest.main()
