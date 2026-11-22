"""Different models must share card leases without changing Qwen ownership."""
import asyncio
import fcntl
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock
from marathon_app.catalog import Backend, backends, load_catalog
from marathon_app.pool import acquire_pool_worker, release_pool_worker
from marathon_app.lease_handoff import receive_lease
from test_router_context import fixture_profile, router_module

class SharedLeaseTests(unittest.TestCase):
    def test_worker_keeps_same_flock_after_client_exits(self):
        qwen=Backend('q','Q',None,kind='llama_swap_pool',pool_models=('q0',))
        ink=replace(qwen,id='i',pool_models=('i0',),pool_lease_backend='q',pool_lease_models=(('i0','q0'),))
        with tempfile.TemporaryDirectory() as d:
            handle,_=acquire_pool_worker(ink,Path(d),'owner')
            handle.seek(0); info=json.load(handle)
            with self.assertRaises(RuntimeError):receive_lease(info['handoff_socket'],'wrong-worker')
            fd=receive_lease(info['handoff_socket'],'i0')
            release_pool_worker(handle)
            try:
                with self.assertRaises(RuntimeError):acquire_pool_worker(qwen,Path(d),'contender')
            finally:os.close(fd)
            handle,_=acquire_pool_worker(qwen,Path(d),'next')
            release_pool_worker(handle)

    def test_same_lock_inode_excludes_other_model_and_survives_release(self):
        qwen=Backend('qwen-pool','Qwen',None,kind='llama_swap_pool',pool_models=('qwen-0','qwen-1'))
        ink=replace(qwen,id='inkling-pool',pool_models=('ink-0','ink-1'),pool_lease_backend=qwen.id,
                    pool_lease_models=(('ink-0','qwen-0'),('ink-1','qwen-1')))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            q,worker=acquire_pool_worker(qwen,root,'qwen')
            i,other=acquire_pool_worker(ink,root,'inkling')
            try:
                self.assertEqual((worker,other),('qwen-0','ink-1'))
                with self.assertRaises(RuntimeError):acquire_pool_worker(ink,root,'third')
                inode=Path(q.name).stat().st_ino;name=q.name
            finally:release_pool_worker(q);release_pool_worker(i)
            i,worker=acquire_pool_worker(ink,root,'next')
            try:
                self.assertEqual(worker,'ink-0');self.assertEqual(i.name,name)
                self.assertEqual(Path(i.name).stat().st_ino,inode)
            finally:release_pool_worker(i)

    def test_catalog_rejects_incomplete_shared_mapping(self):
        raw={'settings':load_catalog()['settings'],'backends':[
            {'id':'q','display_name':'Q','kind':'llama_swap_pool','pool_models':['q0','q1']},
            {'id':'i','display_name':'I','kind':'llama_swap_pool','pool_models':['i0','i1'],
             'pool_lease_backend':'q','pool_lease_models':{'i0':'q0'}}]}
        with self.assertRaisesRegex(ValueError,'shared pool lease mapping'):backends(raw)
        raw['backends'][1]['pool_lease_models']['i1']='q1'
        self.assertEqual(dict(backends(raw)['i'].pool_lease_models),{'i0':'q0','i1':'q1'})

class SharedRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_unloads_before_releasing_and_reuses_canonical_lease(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            qwen=Backend('qwen','Qwen',None,kind='llama_swap_pool',proxy='http://localhost:9292',
                         pool_models=('qwen-0',),model_alias='qwen')
            ink=replace(qwen,id='inkling',model_alias='inkling',pool_models=('inkling-0',),supports_slots=False,
                        pool_lease_backend='qwen',pool_lease_models=(('inkling-0','qwen-0'),))
            q=replace(fixture_profile(),slug='qwen',alias='qwen',target=qwen.proxy)
            i=replace(q,slug='inkling',alias='inkling',supports_slots=False,reasoning_transport='responses',external=True)
            with mock.patch.dict(os.environ,{'MARATHON_LAZY_POOL_BACKEND':'qwen','MARATHON_MODEL_SLUG':'qwen','MARATHON_POOL_RUNTIME_DIR':d}), \
                 mock.patch.object(router_module,'backends',return_value={'qwen':qwen,'inkling':ink}), \
                 mock.patch.object(router_module,'_available_profiles',side_effect=lambda:{'qwen':q,'inkling':i}):
                state=router_module.RouterState('qwen',root/'state',root/'logs')
                state.flush_conversation_checkpoints=mock.AsyncMock(return_value=[])
                unloads=[]
                async def unload(worker):
                    self.assertIsNotNone(state.pool_handle)
                    with self.assertRaises(RuntimeError):acquire_pool_worker(ink,root,'contender')
                    unloads.append(worker)
                state._unload_pool_worker=unload
                try:
                    async with state.pool_request('qwen'):
                        lock=state.pool_handle.name
                        self.assertEqual(state.pool_model,'qwen-0')
                    async with state.pool_request('inkling'):
                        self.assertEqual(unloads,['qwen-0'])
                        self.assertEqual(state.pool_handle.name,lock)
                        self.assertEqual(state.pool_model,'inkling-0')
                        self.assertIn('/upstream/inkling-0',state.resolve_model('inkling').target)
                        self.assertFalse(state.resolve_model('inkling').supports_slots)
                    async with state.pool_request('qwen'):
                        self.assertEqual(unloads,['qwen-0','inkling-0'])
                        self.assertEqual(state.pool_handle.name,lock)
                        self.assertEqual(state.default_model,'qwen')
                        self.assertTrue(state.resolve_model('qwen').supports_slots)
                finally:
                    await state._cancel_pool_release();state._release_pool_worker()

class SharedPoolReasoningReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_delta_replay_translates_every_prior_reasoning_item(self):
        from test_router_transport import serving, scripted_backend, answer
        requests=[]
        reasoning={"type":"reasoning","summary":[],"content":[
            {"type":"reasoning_text","text":"A synthetic plan."}]}
        backend=scripted_backend(lambda data,count:[reasoning,answer()],requests)
        async with serving(backend,alias="inkling-2-mini",reasoning_transport="responses") as (state,http,server):
            previous=None
            for _ in range(3):
                payload={"model":"inkling-2-mini","input":[{"role":"user","content":"Continue the synthetic task."}]}
                if previous:payload["previous_response_id"]=previous
                async with http.post(server.make_url('/v1/responses'),json=payload) as response:
                    body=await response.json();self.assertEqual(response.status,200,body)
                previous=body['id']
            self.assertEqual(len(requests),3)
            for index,request in enumerate(requests):
                prior=[item for item in request['input'] if item.get('type')=='reasoning']
                self.assertEqual(len(prior),index)
                for item in prior:
                    self.assertNotIn('encrypted_content',item)
                    self.assertEqual(item['content'],reasoning['content'])
