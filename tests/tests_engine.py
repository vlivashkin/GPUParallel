import unittest

from gpuparallel import GPUParallel, delayed


def task_return_identity(value, **kwargs):
    return value


class TestBatchGPUParallel(unittest.TestCase):
    def test_multiprocessing(self):
        true_seq = list(range(100))

        gpup = GPUParallel(n_gpu=2, engine="multiprocessing", progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(true_seq, sorted(results))

    def test_billiard(self):
        true_seq = list(range(100))

        gpup = GPUParallel(n_gpu=2, engine="billiard", progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(true_seq, sorted(results))
