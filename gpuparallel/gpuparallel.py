import logging
from functools import partial
from multiprocessing import Pool, Manager, Queue, active_children
from typing import List, Iterable, Optional, Callable, Union, Generator

from gpuparallel.utils import log, import_tqdm


def _init_worker(gpu_queue: Queue, init_fn: Optional[Callable] = None):
    global worker_id, device_id

    worker_id, device_id = gpu_queue.get()
    if init_fn is not None:
        init_fn(worker_id=worker_id, device_id=device_id)

    if len(log.handlers) > 0:
        fmt = logging.Formatter(f"[%(levelname)s/Worker-{worker_id}({device_id})]:%(message)s")
        log.handlers[0].setFormatter(fmt)

    log.debug(f"Worker #{worker_id} with GPU{device_id} initialized.")


def _run_task(func: Callable, task_idx, result_queue: Queue, ignore_errors=True):
    global worker_id, device_id

    try:
        result = func(worker_id=worker_id, device_id=device_id)
        result_queue.put((task_idx, result))
    except Exception as e:
        log.error(f"Error during task #{task_idx}", exc_info=True)
        if ignore_errors:
            log.warning(f"Exception will be ignored according to ignore_errors flag")
            result_queue.put((task_idx, None))  # __call__ expects to get number of results equal to number of tasks
        else:
            raise e


class GPUParallel:
    def __init__(
        self,
        device_ids: Optional[List[str]] = None,
        n_gpu: Optional[Union[int, str]] = None,
        n_workers_per_gpu=1,
        init_fn: Optional[Callable] = None,
        preserve_order=True,
        progressbar=True,
        pbar_description=None,
        ignore_errors=False,
        debug=False,
    ):
        """
        Parallel execution of functions passed to ``__call__``.

        :param device_ids:
            List of gpu ids to use, e.g. ``['cuda:3', 'cuda:4']``. The library doesn't check if GPUs really available,
            it simply provides consistent ``worker_id`` and ``device_id`` to both ``init_fn`` and task functions.
        :param n_gpu:
            Number of GPUs to use, shortcut for ``device_ids=[f'cuda:{i}' for i in range(n_gpu)]``.
            Both parameters ``n_gpu`` and ``device_ids`` can't be filled.
            If neither of them filled, single ``cuda:0`` will be chosen.
        :param n_workers_per_gpu: Number of workers on every GPU.
        :param init_fn:
            Function which will be called during worker init.
            Function must have parameters ``worker_id`` and ``device_id`` (or ``**kwargs``).
            Helpful to init all common stuff (e.g. neural networks) here.
        :param preserve_order: Return values with the same order as input.
        :param progressbar: Allow to use tqdm progressbar.
        :param ignore_errors: Either ignore errors inside tasks or raise them.
        :param debug: When this parameter is True, parameters n_gpu and device_ids are ignored.
            Class creates only one worker ([device_id='cuda:0']) and run it in the same process (for better debugging).

        """
        assert not (n_gpu is not None and device_ids is not None), "Both 'n_gpu' and 'device_ids' cannot de filled"

        self.n_workers_per_gpu = n_workers_per_gpu
        self.preserve_order = preserve_order
        self.progressbar = progressbar
        self.pbar_description = pbar_description
        self.ignore_errors = ignore_errors
        self.debug_mode = debug

        if device_ids is not None:
            assert len(device_ids) > 0, "len(device_ids) must be > 0"
            self.n_gpu = len(device_ids)
            self.device_ids = device_ids
        else:
            n_gpu = n_gpu if n_gpu is not None else 1
            assert n_gpu > 0, "n_gpu must be > 0"
            self.n_gpu = n_gpu
            self.device_ids = [f"cuda:{idx}" for idx in range(n_gpu)]

        if not self.debug_mode:
            self._manager = Manager()
            self.gpu_queue = self._manager.Queue()
            for device_idx in range(self.n_gpu):
                for idx in range(self.n_workers_per_gpu):
                    worker_id = device_idx * self.n_workers_per_gpu + idx
                    self.gpu_queue.put((worker_id, self.device_ids[device_idx]))

            initializer = partial(_init_worker, gpu_queue=self.gpu_queue, init_fn=init_fn)
            self.pool = Pool(
                processes=self.n_gpu * self.n_workers_per_gpu, initializer=initializer, maxtasksperchild=None
            )

            self.result_queue = self._manager.Queue()
        else:  # debug mode; run init in the same process
            log.warning("Debug mode. All tasks will be run in main process for debug purposes.")
            if init_fn is not None:
                init_fn(worker_id=0, device_id=self.device_ids[0])

    def __del__(self):
        """
        Created pool will be freed only during this destructor.
        This allows to use ``__call__`` multiple times with the same initialized workers.
        """
        if not self.debug_mode:
            try:
                self._manager.shutdown()
                self.pool.close()
                self.pool.join()
            except Exception as e:
                log.warning("Can't close and join process pool.", exc_info=True)

        children = active_children()
        if children:
            log.warning(f"{len(children)} child processes are still alive after closing the pool.")

    def _call_sync(self, tasks: Iterable) -> List:
        log.warning(f"Debug mode is turned on. All tasks will be run in the main process.")

        tasks = list(tasks)
        if self.progressbar:
            from tqdm.auto import tqdm

            tasks = tqdm(tasks)
        for task in tasks:
            yield task(worker_id=0, device_id=self.device_ids[0])

    def _call_async(self, tasks: Iterable) -> Generator:
        # Submit all tasks to pool
        n_tasks = 0
        for task_idx, task in enumerate(tasks):
            self.pool.apply_async(_run_task, (task, task_idx, self.result_queue, self.ignore_errors))
            n_tasks += 1
        log.debug(f"Submitted {n_tasks} tasks")

        # Wait for all tasks to be performed
        tqdm = import_tqdm(self.progressbar)
        if self.preserve_order:
            with tqdm(total=n_tasks, desc=self.pbar_description) as pbar:
                result_cache = {}
                for return_task_idx in range(n_tasks):
                    while return_task_idx not in result_cache:
                        log.debug(f"{return_task_idx} not in cached {list(result_cache.keys())}...")
                        task_idx, result = self.result_queue.get()
                        result_cache[task_idx] = result
                        pbar.update(1)
                    log.debug(f"Found {return_task_idx} in cache!")
                    yield result_cache[return_task_idx]
                    del result_cache[return_task_idx]
        else:
            with tqdm(total=n_tasks, desc=self.pbar_description) as pbar:
                for return_task_idx in range(n_tasks):
                    task_idx, result = self.result_queue.get()
                    yield result
                    pbar.update(1)
        log.debug("All results are received!")

    def __call__(self, tasks: Iterable[Callable]) -> Generator:
        """
        Function which submits tasks for pool and collects the results of computations.

        :param tasks:
            List or generator with callable functions to be executed.
            Functions must have parameters ``worker_id`` and ``device_id`` (or ``**kwargs``).
        :return: List of results or generator
        """

        return self._call_async(tasks) if not self.debug_mode else self._call_sync(tasks)
