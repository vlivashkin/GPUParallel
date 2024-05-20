import unittest

from gpuparallel import GPUParallel, BatchGPUParallel, delayed, log_to_stderr

log_to_stderr(log_level='INFO')


def task_return_identity(value, **kwargs):
    return value


def task_return_all_kwargs(value, **kwargs):
    return value, kwargs


class TestBatchGPUParallel(unittest.TestCase):
    def test_batched_results_equal(self):
        true_seq = list(range(100))

        gpup = GPUParallel(n_gpu=2, progressbar=False)
        non_batched_results = gpup(delayed(task_return_identity)(idx) for idx in true_seq)

        bgpup = BatchGPUParallel(task_fn=task_return_identity, batch_size=9, n_gpu=2, progressbar=False)
        batched_results = list(bgpup(true_seq))
        self.assertEqual(len(batched_results), 12)

        batched_results_ravel = [item for sublist in batched_results for item in sublist]
        self.assertEqual(sorted(non_batched_results), sorted(batched_results_ravel))

    def test_args_flat_result(self):
        true_seq = list(range(10))

        bgpup = BatchGPUParallel(task_fn=task_return_identity, batch_size=3, flat_result=True,
                                 n_gpu=2, progressbar=False)
        batched_results = list(bgpup(true_seq))
        self.assertEqual(true_seq, batched_results)

    def test_kwargs_flat_result(self):
        true_seq = list(range(10))

        bgpup = BatchGPUParallel(task_fn=task_return_identity, batch_size=3, flat_result=True,
                                 n_gpu=2, progressbar=False)
        batched_results = list(bgpup(value=true_seq))
        self.assertEqual(true_seq, batched_results)

    def test_nonbatched_args(self):
        true_seq = list(range(10))
        true_first_batch = (
            [0, 1, 2], {
                'batched_value': [0, 1, 2],
                'nonbatched_sequence': 'test',
                'nonbatched_value': 3,
                'device_id': 'cuda:0',
                'worker_id': 0
            }
        )

        bgpup = BatchGPUParallel(task_fn=task_return_all_kwargs, batch_size=3, flat_result=False,
                                 n_gpu=1, progressbar=False)
        batched_results = list(bgpup(true_seq, batched_value=true_seq, nonbatched_sequence='test', nonbatched_value=3))
        self.assertEqual(true_first_batch, batched_results[0])
