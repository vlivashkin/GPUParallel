import unittest
from typing import Generator

from gpuparallel import GPUParallel, delayed, log_to_stderr

log_to_stderr()


def test_init__init(worker_id=None, **kwargs):
    global result
    result = worker_id


def test_init__perform(**kwargs):
    global result
    return result


def test_results__perform(idx, **kwargs):
    return idx


def test_multicall__perform(idx, **kwargs):
    return idx


class TestGPUParallel(unittest.TestCase):
    def test_init(self):
        results = GPUParallel(n_gpu=2, init_fn=test_init__init)(test_init__perform for _ in range(10))
        self.assertEqual(set(results), {0, 1})

    def test_results(self):
        true_seq = list(range(10))
        results = GPUParallel(n_gpu=2, progressbar=False)(delayed(test_results__perform)(idx) for idx in true_seq)
        self.assertEqual(sorted(results), true_seq)

    def test_multicall(self):
        true_seq1, true_seq2 = list(range(10)), list(range(10, 20))
        gpup = GPUParallel(n_gpu=2, progressbar=False)
        results = gpup(delayed(test_multicall__perform)(idx) for idx in true_seq1)
        self.assertEqual(sorted(results), true_seq1)
        results = gpup(delayed(test_multicall__perform)(idx) for idx in true_seq2)
        self.assertEqual(sorted(results), true_seq2)

    def test_debug_mode(self):
        true_seq = list(range(10))
        results = GPUParallel(n_gpu=0, progressbar=False)(delayed(test_results__perform)(idx) for idx in true_seq)
        self.assertEqual(sorted(results), true_seq)

    def test_generator(self):
        true_seq = list(range(10))
        gpup = GPUParallel(n_gpu=2, progressbar=False, return_generator=True)
        results = gpup(delayed(test_results__perform)(idx) for idx in true_seq)
        self.assertTrue(isinstance(results, Generator))
        self.assertEqual(sorted(list(results)), true_seq)


if __name__ == '__main__':
    unittest.main()
