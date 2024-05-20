import random
import unittest
from time import sleep
from typing import Generator

from gpuparallel import GPUParallel, delayed, log_to_stderr

log_to_stderr(log_level="INFO")


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


def task_wait_random_time(idx, **kwargs):
    time_to_sleep = random.random()
    sleep(time_to_sleep)
    return idx


class TestGPUParallel(unittest.TestCase):
    def test_init(self):
        true_set = {0, 1}
        results = GPUParallel(n_gpu=2, init_fn=test_init__init)(test_init__task for _ in range(10))
        self.assertEqual(true_set, set(results))

    def test_results(self):
        true_seq = list(range(10))
        results = GPUParallel(n_gpu=2, progressbar=False)(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(true_seq, sorted(results))

    def test_multicall(self):
        true_seq1, true_seq2 = list(range(10)), list(range(10, 20))
        gpup = GPUParallel(n_gpu=2, progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq1)
        self.assertEqual(true_seq1, sorted(results))
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq2)
        self.assertEqual(true_seq2, sorted(results))

    def test_debug_mode(self):
        true_seq = list(range(10))
        gpup = GPUParallel(n_gpu=1, progressbar=False, debug=True)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(true_seq, sorted(results))

    def test_generator(self):
        true_seq = list(range(10))
        gpup = GPUParallel(n_gpu=2, progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertTrue(isinstance(results, Generator))
        self.assertEqual(true_seq, sorted(list(results)))

    def test_device_ids(self):
        true_device_ids = ["cuda:3", "cuda:4"]
        gpup = GPUParallel(device_ids=true_device_ids, progressbar=False)
        results = gpup(delayed(task_return_device_id)() for _ in range(100))
        self.assertEqual(set(true_device_ids), set(results))

    def test_preserve_order(self):
        true_sequence = list(range(100))
        gpup = GPUParallel(n_gpu=20, preserve_order=True, progressbar=False)
        results = gpup(delayed(task_wait_random_time)(idx) for idx in true_sequence)
        self.assertEqual(true_sequence, list(results))


if __name__ == "__main__":
    unittest.main()
