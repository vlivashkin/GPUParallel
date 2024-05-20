import unittest

from gpuparallel import GPUParallel, delayed
from gpuparallel.exceptions import GPUPPoolException
from gpuparallel.utils import log_to_stderr

log_to_stderr(log_level="INFO")


def task_return_identity(value, **kwargs):
    return value


def task_return_exception(value, **kwargs):
    raise RuntimeError("Exception!")


class TestBatchGPUParallel(unittest.TestCase):

    def setup_method(self, method):
        print(f"\n{type(self).__name__}:{method.__name__}")

    def test_multiprocessing(self):
        true_seq = list(range(100))
        gpup = GPUParallel(n_gpu=2, engine="multiprocessing", progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(true_seq, sorted(results))
        del gpup

    def test_multiprocessing_ignore_exception(self):
        input_seq = list(range(3))
        gpup = GPUParallel(n_gpu=2, engine="multiprocessing", ignore_errors=True, progressbar=False)
        results = gpup(delayed(task_return_exception)(idx) for idx in input_seq)
        results = list(results)
        self.assertEqual(len(input_seq), len(results))
        self.assertTrue(all(x is None for x in results))
        del gpup

    def test_multiprocessing_raise_exception(self):
        input_seq = list(range(3))
        gpup = GPUParallel(n_gpu=2, engine="multiprocessing", ignore_errors=False, progressbar=False)
        result = gpup(delayed(task_return_exception)(idx) for idx in input_seq)
        with self.assertRaises(GPUPPoolException):
            _ = list(result)
        del gpup

    def test_multiprocessing_raise_and_run_another(self):
        input_seq = list(range(3))
        gpup = GPUParallel(n_gpu=2, engine="multiprocessing", ignore_errors=False, progressbar=False)
        result = gpup(delayed(task_return_exception)(idx) for idx in input_seq)
        with self.assertRaises(GPUPPoolException):
            _ = list(result)
        del gpup

        true_seq = list(range(100))
        gpup = GPUParallel(n_gpu=2, engine="multiprocessing", progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(true_seq, sorted(results))
        del gpup

    def test_billiard(self):
        true_seq = list(range(100))
        gpup = GPUParallel(n_gpu=2, engine="billiard", progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(true_seq, sorted(results))
        del gpup

    def test_billiard_ignore_exception(self):
        input_seq = list(range(3))
        gpup = GPUParallel(n_gpu=2, engine="billiard", ignore_errors=True, progressbar=False)
        results = gpup(delayed(task_return_exception)(idx) for idx in input_seq)
        results = list(results)
        self.assertEqual(len(input_seq), len(results))
        self.assertTrue(all(x is None for x in results))
        del gpup

    def test_billiard_raise_exception(self):
        input_seq = list(range(3))
        gpup = GPUParallel(n_gpu=2, engine="billiard", ignore_errors=False, progressbar=False)
        result = gpup(delayed(task_return_exception)(idx) for idx in input_seq)
        with self.assertRaises(GPUPPoolException):
            _ = list(result)
        del gpup

    def test_billiard_raise_and_run_another(self):
        input_seq = list(range(3))
        gpup = GPUParallel(n_gpu=2, engine="billiard", ignore_errors=False, progressbar=False)
        result = gpup(delayed(task_return_exception)(idx) for idx in input_seq)
        with self.assertRaises(GPUPPoolException):
            _ = list(result)
        del gpup

        true_seq = list(range(100))
        gpup = GPUParallel(n_gpu=2, engine="billiard", progressbar=False)
        results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)
        self.assertEqual(true_seq, sorted(results))
        del gpup
