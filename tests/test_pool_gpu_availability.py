"""Pool selection must account for jobs outside Marathon's lease registry."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from marathon_app.catalog import Backend
from marathon_app.pool import acquire_pool_worker, release_pool_worker


class GPUAvailabilityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        config = self.root / 'broker.yaml'
        config.write_text('''models:
  worker-1: &worker
    macros: {gpu: 1}
    cmd: docker run --name llama-swap-${MODEL_ID} --gpus '"device=${gpu}"'
  worker-2:
    <<: *worker
    macros: {gpu: 2}
''')
        self.backend = Backend('test', 'Test', None, pool_models=('worker-1', 'worker-2'),
                               identity_config=config, pool_min_free_vram_mib=23500)

    def choose(self, processes='', free='24500', running='', pids='PID\n'):
        def output(args):
            if '--query-gpu=index,uuid,memory.free' in args:
                return f'1, GPU-one, {free}\n2, GPU-two, 24500\n'
            if '--query-compute-apps=gpu_uuid,pid,used_memory' in args:
                return processes
            if args[1] == 'ps':
                return running
            if args[1] == 'top':
                return pids
            self.fail(str(args))
        with patch('marathon_app.pool._output', side_effect=output):
            handle, worker = acquire_pool_worker(self.backend, self.root, None)
        release_pool_worker(handle)
        return worker

    def test_training_job_is_skipped_even_with_free_memory(self):
        self.assertEqual(self.choose('GPU-one, 42, 100\n'), 'worker-2')

    def test_insufficient_vram_without_reported_process_is_skipped(self):
        self.assertEqual(self.choose(free='16000'), 'worker-2')

    def test_warm_worker_memory_is_reusable(self):
        self.assertEqual(self.choose('GPU-one, 42, 23900\n', '600',
                                    'llama-swap-worker-1\n', 'PID\n42\n'), 'worker-1')

    def test_foreign_job_alongside_warm_worker_is_skipped(self):
        self.assertEqual(self.choose('GPU-one, 42, 22000\nGPU-one, 43, 100\n',
                                    '2400', 'llama-swap-worker-1\n', 'PID\n42\n'), 'worker-2')

    def test_unavailable_gpu_probe_fails_closed(self):
        with patch('marathon_app.pool._output', side_effect=OSError('missing')):
            with self.assertRaisesRegex(RuntimeError, 'cannot verify GPU availability'):
                acquire_pool_worker(self.backend, self.root, None)

    def test_independent_availability_probes_run_concurrently(self):
        probes = threading.Barrier(3, timeout=5)

        def output(args):
            # Each probe must start before any can complete. This verifies
            # concurrency without depending on machine-specific latency.
            probes.wait()
            if '--query-gpu=index,uuid,memory.free' in args:
                return '1, GPU-one, 24500\n2, GPU-two, 24500\n'
            if '--query-compute-apps=gpu_uuid,pid,used_memory' in args:
                return 'GPU-one, 42, 100\n'
            return ''

        with patch('marathon_app.pool._output', side_effect=output):
            handle, worker = acquire_pool_worker(self.backend, self.root, None)
        try:
            self.assertEqual(worker, 'worker-2')
        finally:
            release_pool_worker(handle)

    def test_all_busy_reports_jobs(self):
        with self.assertRaisesRegex(RuntimeError, 'other jobs'):
            self.choose('GPU-one, 42, 100\nGPU-two, 43, 100\n')

    def test_bad_placement_skips_worker(self):
        self.backend.identity_config.write_text('models: {worker-1: {cmd: bad}, worker-2: {cmd: bad}}')
        with self.assertRaisesRegex(RuntimeError, 'placement'):
            self.choose()
