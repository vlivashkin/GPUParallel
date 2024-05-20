import unittest
from typing import Generator

from gpuparallel import GPUParallel, delayed, log_to_stderr

log_to_stderr()


def test_init__init(worker_id=None, **kwargs):
    global result
    result = worker_id


def test_init__task(**kwargs):
    global result
    return result


def task_return_identity(value, **kwargs):
    return value


def task_return_device_id(device_id, **kwargs):
    return device_id


class TestGPUParallel(unittest.TestCase):
    def test_init(self):
        results = GPUParallel(n_gpu=2, init_fn=test_init__init)(test_init__task for _ in range(10))
        self.assertEqual(set(results), {0, 1})

    def test_results(self):
        true_seq = list(range(10))
        results = GPUParallel(n_gpu=2, progressbar=False)(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(sorted(results), true_seq)

    def test_multicall(self):
        true_seq1, true_seq2 = list(range(10)), list(range(10, 20))
        gpup = GPUParallel(n_gpu=2, progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq1)
        self.assertEqual(sorted(results), true_seq1)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq2)
        self.assertEqual(sorted(results), true_seq2)

    def test_debug_mode(self):
        true_seq = list(range(10))
        gpup = GPUParallel(n_gpu=1, progressbar=False, debug=True)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(sorted(results), true_seq)

    def test_generator(self):
        true_seq = list(range(10))
        gpup = GPUParallel(n_gpu=2, progressbar=False, return_generator=True)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertTrue(isinstance(results, Generator))
        self.assertEqual(sorted(list(results)), true_seq)

    def test_device_ids(self):
        true_device_ids = ['cuda:3', 'cuda:4']
        gpup = GPUParallel(device_ids=true_device_ids, progressbar=False)
        results = gpup(delayed(task_return_device_id)() for _ in range(100))
        self.assertEqual(set(results), set(true_device_ids))


if __name__ == '__main__':
    unittest.main()
