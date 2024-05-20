import multiprocessing as mp
import traceback
from functools import partial
from multiprocessing import Pool, Manager, Queue
from typing import List, Iterable, Optional, Callable


def _init_worker(gpu_queue: Queue, init_fn: Optional[Callable] = None, verbose=True):
    global worker_id, gpu_id

    worker_id, gpu_id = gpu_queue.get()
    if init_fn is not None:
        init_fn(worker_id=worker_id, gpu_id=gpu_id)

    if verbose:
        mp.get_logger().info(f'Worker #{worker_id} with GPU{gpu_id} initialized.')


def _run_task(func: Callable, result_queue: Queue, ignore_errors=True):
    global worker_id, gpu_id

    try:
        result = func(worker_id=worker_id, gpu_id=gpu_id)
        result_queue.put(result)
    except Exception as e:
        mp.get_logger().error(traceback.format_exc())
        result_queue.put(None)  # __call__ expects to get number of results equal to number of tasks
        if not ignore_errors:
            raise


class GPUParallel:
    def __init__(self, n_gpu=1, n_workers_per_gpu=1, init_fn: Optional[Callable] = None, verbose=True,
                 progressbar=True, ignore_errors=True):
        """
        :param n_gpu:
            Number of GPUs to use. The library doesn't check if GPUs really available, it is simply provide
            consistent ``worker_id`` and ``gpu_id`` to both ``init_fn`` and task functions.
        :param n_workers_per_gpu: Number of workers on every GPU.
        :param init_fn:
            Function which will be called during worker init.
            Function must have parameters ``worker_id`` and ``gpu_id`` (or ``**kwargs``).
            Helpful to init all common stuff (e.g. neural networks) here.
        :param verbose:
            Allow additional messages.
            Note that messages inside ``perform`` should be sent to ``mp.get_logger().info(message)``.
            To make them visible, you need to run ``mp.log_to_stderr(); mp.get_logger().setLevel('INFO')``
        :param progressbar: Allow to use tqdm progressbar.
        :param ignore_errors: Either ignore errors inside tasks or raise them.
        """
        self.n_gpu = n_gpu
        self.n_workers_per_gpu = n_workers_per_gpu
        self.init_fn = init_fn
        self.verbose = verbose
        self.progressbar = progressbar
        self.ignore_errors = ignore_errors

        m = Manager()
        self.gpu_queue = m.Queue()
        for gpu_id in range(self.n_gpu):
            for idx in range(self.n_workers_per_gpu):
                worker_id = gpu_id * self.n_workers_per_gpu + idx
                self.gpu_queue.put((worker_id, gpu_id))

        initializer = partial(_init_worker, gpu_queue=self.gpu_queue, init_fn=init_fn, verbose=self.verbose)
        self.pool = Pool(processes=self.n_gpu * self.n_workers_per_gpu, initializer=initializer,
                         maxtasksperchild=None)

        self.result_queue = m.Queue()

    def __del__(self):
        """
        Created pool will be freed only during this destructor.
        This allows to use ``__call__`` multiple times with the same initialized workers.
        """
        self.pool.close()
        self.pool.join()

    def __call__(self, tasks: Iterable) -> List:
        """
        Function which submits tasks for pool and collects the results of computations.

        :param tasks:
            List or generator with callable functions to be executed.
            Functions must have parameters ``worker_id`` and ``gpu_id`` (or ``**kwargs``).
        :return: List of results
        """
        n_tasks = 0
        for task in tasks:
            self.pool.apply_async(_run_task, (task, self.result_queue, self.ignore_errors))
            n_tasks += 1

        if self.verbose:
            print(f'Submitted {n_tasks} tasks')

        results = []
        if self.progressbar:
            from tqdm.auto import tqdm
            with tqdm(total=n_tasks) as pbar:
                for idx in range(n_tasks):
                    result = self.result_queue.get()
                    results.append(result)
                    pbar.update(1)
        else:
            for _ in range(n_tasks):
                result = self.result_queue.get()
                results.append(result)

        if self.verbose:
            print('All results are received!')

        return results
