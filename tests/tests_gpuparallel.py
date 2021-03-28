import unittest
from functools import partial

from gpuparallel import GPUParallel


class TestGPUParallel(unittest.TestCase):
    def test_init(self):
        print('Run Test: test_init')

        def init(worker_id=None, gpu_id=None):
            global result
            result = worker_id

        def perform(worker_id=None, gpu_id=None):
            global result
            return result

        results = GPUParallel(n_gpu=2, init_fn=init)(perform for _ in range(10))
        self.assertEqual(set(results), {0, 1})

    def test_results(self):
        print('Run Test: test_results')

        def perform(idx, worker_id=None, gpu_id=None):
            print(f'Perform {idx}')
            return idx

        true_seq = list(range(10))
        results = GPUParallel(n_gpu=2)(partial(perform, idx) for idx in true_seq)
        self.assertEqual(sorted(results), true_seq)

    def test_multicall(self):
        print('Run Test: test_multicall')

        def perform(idx, worker_id=None, gpu_id=None):
            print(f'Perform {idx}')
            return idx

        true_seq1, true_seq2 = list(range(10)), list(range(10, 20))
        gp = GPUParallel(n_gpu=2)
        results = gp(partial(perform, idx) for idx in true_seq1)
        self.assertEqual(sorted(results), true_seq1)
        results = gp(partial(perform, idx) for idx in true_seq2)
        self.assertEqual(sorted(results), true_seq2)


if __name__ == '__main__':
    unittest.main()
