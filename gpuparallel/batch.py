import math
from typing import List, Union, Generator, Callable, Tuple, Dict, Sequence

from .gpuparallel import GPUParallel
from .utils import delayed


class BatchGPUParallel(GPUParallel):
    def __init__(self, task_fn: Callable, batch_size, flat_result=False, *args, **kwargs):
        """
        Parallel execution of ``task_fn`` with parameters given to ``__call__``.
        Tasks are batched: every arg and kwarg turns into list.

        :param task_fn: Task to be executed
        :param batch_size: Batch size
        :param flat_result: Either unbatch results or not.
            Param assumes that result of ``task_fn`` is an Iterable with len equal ``batch_size``.
        """
        super().__init__(*args, **kwargs)

        self.task_fn = task_fn
        self.batch_size = batch_size
        self.flat_result = flat_result

    def __call__(self, tasks: Sequence[Tuple[List, Dict]]) -> Union[List, Generator]:
        """
        :param tasks: List of parameters for ``task_fn``. Every element is (List of *args, Dict of **kwargs).
        :return: Batched result
        """
        n_batches = math.ceil(len(tasks) / self.batch_size)
        batches = []
        for batch_idx in range(n_batches):
            tasks_of_batch = tasks[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
            collated_task = ([], {})
            for arg_idx in range(len(tasks_of_batch[0][0])):
                collated_arg = [tasks_of_batch[idx][0][arg_idx] for idx in range(len(tasks_of_batch))]
                collated_task[0].append(collated_arg)
            for kwarg_key in tasks_of_batch[0][1].keys():
                collated_kwarg = [tasks_of_batch[idx][1][kwarg_key] for idx in range(len(tasks_of_batch))]
                collated_task[1][kwarg_key] = collated_kwarg
            batches.append(delayed(self.task_fn)(*collated_task[0], **collated_task[1]))

        result = super().__call__(batches)
        if self.flat_result:
            result = [item for sublist in result for item in sublist]
        return result
