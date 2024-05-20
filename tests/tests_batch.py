import unittest

from gpuparallel import GPUParallel, BatchGPUParallel, delayed


def task_return_identity(value, **kwargs):
    return value


class TestBatchGPUParallel(unittest.TestCase):
    def test_batched_results_equal(self):
        gpup = GPUParallel(n_gpu=2, progressbar=False)
        non_batched_results = gpup(delayed(task_return_identity)(idx) for idx in range(100))

        bgpup = BatchGPUParallel(task_fn=task_return_identity, batch_size=9, n_gpu=2, progressbar=False)
        batched_results = bgpup([([idx], {}) for idx in range(100)])
        self.assertEqual(len(batched_results), 12)

        batched_results_ravel = [item for sublist in batched_results for item in sublist]
        self.assertEqual(sorted(batched_results_ravel), sorted(non_batched_results))

    def test_args_flat_result(self):
        true_seq = list(range(10))

        bgpup = BatchGPUParallel(task_fn=task_return_identity, batch_size=3, flat_result=True,
                                 n_gpu=2, progressbar=False)
        batched_results = bgpup([([idx], {}) for idx in true_seq])
        self.assertEqual(sorted(batched_results), true_seq)

    def test_kwargs_flat_result(self):
        true_seq = list(range(10))

        bgpup = BatchGPUParallel(task_fn=task_return_identity, batch_size=3, flat_result=True,
                                 n_gpu=2, progressbar=False)
        batched_results = bgpup([([], {'value': idx}) for idx in true_seq])
        self.assertEqual(sorted(batched_results), true_seq)
