import math
import multiprocessing as mp
from typing import Generator, Callable, Sequence

from .gpuparallel import GPUParallel
from .utils import delayed

log = mp.get_logger()


class BatchGPUParallel(GPUParallel):
    def __init__(self, task_fn: Callable, batch_size, flat_result=False, *args, **kwargs):
        """
        Parallel execution of ``task_fn`` with parameters given to ``__call__``.
        Tasks are batched: every arg and kwarg turns into list.

        :param task_fn: Task to be executed
        :param batch_size: Batch size
        :param flat_result: Unbatch results. Works only for single tensor output.
        """
        super().__init__(*args, **kwargs)

        self.task_fn = task_fn
        self.batch_size = batch_size
        self.flat_result = flat_result

    def __call__(self, *args, **kwargs) -> Generator:
        """
        All input parameters should have equal first axis to be batched.
        First arg/kwarg is used to determine size of the dataset.
        Inputs with other shape (or not Sequence typed) will be copied to every worker without batching.
        :return: Batched result
        """
        n_samples = len(args[0]) if len(args) > 0 else len(kwargs[list(kwargs.keys())[0]])
        n_batches = math.ceil(n_samples / self.batch_size)

        will_be_batched_args, will_be_batched_kwargs = set(), set()
        wont_be_batched_args, wont_be_batched_kwargs = set(), set()
        is_batched = lambda arg: hasattr(arg, "__len__") and len(arg) == n_samples
        for arg_idx, arg in enumerate(args):
            (will_be_batched_args if is_batched(arg) else wont_be_batched_args).add(arg_idx)
        for kwarg_key, kwarg_value in kwargs.items():
            (will_be_batched_kwargs if is_batched(kwarg_value) else wont_be_batched_kwargs).add(kwarg_key)
        log.info(f"Args: {will_be_batched_args} will be batched, {wont_be_batched_args} will be copied")
        log.info(f"Kwargs: {will_be_batched_kwargs} will be batched, {wont_be_batched_kwargs} will be copied")
        log.info(f"Total samples: {n_samples}, batches: {n_batches}")

        batches = []
        for batch_idx in range(n_batches):
            slce = slice(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)
            batch_args_kwargs = ([], {})
            for arg_idx, arg in enumerate(args):
                batch_arg = arg[slce] if arg_idx in will_be_batched_args else arg
                batch_args_kwargs[0].append(batch_arg)
            for kwarg_key, kwarg_value in kwargs.items():
                batch_kwarg = kwarg_value[slce] if kwarg_key in will_be_batched_kwargs else kwarg_value
                batch_args_kwargs[1][kwarg_key] = batch_kwarg
            batches.append(delayed(self.task_fn)(*batch_args_kwargs[0], **batch_args_kwargs[1]))

        result = super().__call__(batches)
        for batch in result:
            if self.flat_result:
                for item in batch:
                    yield item
            else:
                yield batch
