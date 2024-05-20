import unittest
from functools import partial

from gpuparallel import GPUParallel


class TestGPUParallel(unittest.TestCase):
    def test_init(self):
        def init(worker_id=None, gpu_id=None):
            global result
            result = worker_id

        def perform(worker_id=None, gpu_id=None):
            global result
            return result

        results = GPUParallel(n_gpu=2, init_fn=init)(perform for _ in range(100))
        self.assertEqual(set(results), {0, 1})

    def test_results(self):
        def perform(idx, worker_id=None, gpu_id=None):
            return idx

        true_seq = list(range(100))
        results = GPUParallel(n_gpu=2)(partial(perform, idx) for idx in true_seq)
        self.assertEqual(sorted(results), true_seq)

    def test_multicall(self):
        def perform(idx, worker_id=None, gpu_id=None):
            return idx

        true_seq1, true_seq2 = list(range(100)), list(range(100, 200))
        gp = GPUParallel(n_gpu=2)
        results = gp(partial(perform, idx) for idx in true_seq1)
        self.assertEqual(sorted(results), true_seq1)
        results = gp(partial(perform, idx) for idx in true_seq2)
        self.assertEqual(sorted(results), true_seq2)


if __name__ == '__main__':
    unittest.main()
