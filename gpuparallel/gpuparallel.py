import multiprocessing
import traceback
from functools import partial
from multiprocessing import Pool, Manager

from tqdm import tqdm


def run_task(func, result_queue):
    global worker_id, gpu_id

    result = None
    try:
        result = func(worker_id=worker_id, gpu_id=gpu_id)
    except Exception as e:
        multiprocessing.get_logger().error(traceback.format_exc())
    #         raise

    result_queue.put(result)


class GPUParallel:
    def __init__(self, n_gpu=1, n_workers_per_gpu=1, init_fn=None, verbose=True, tqdm=True):
        self.n_gpu = n_gpu
        self.n_workers_per_gpu = n_workers_per_gpu
        self.verbose = verbose
        self.tqdm = tqdm
        self.init_fn = init_fn

        m = Manager()
        self.gpu_queue = m.Queue()
        for gpu_id in range(self.n_gpu):
            for idx in range(self.n_workers_per_gpu):
                worker_id = gpu_id * self.n_workers_per_gpu + idx
                self.gpu_queue.put((worker_id, gpu_id))

        multiprocessing.log_to_stderr()
        self.pool = Pool(processes=self.n_gpu * self.n_workers_per_gpu,
                         initializer=partial(self.init_worker, init_fn=init_fn),
                         maxtasksperchild=None)

        self.result_queue = m.Queue()

    def init_worker(self, init_fn=None):
        global worker_id, gpu_id
        worker_id, gpu_id = self.gpu_queue.get()
        if init_fn is not None:
            init_fn(worker_id=worker_id, gpu_id=gpu_id)

    def __call__(self, tasks):
        n_tasks = 0
        for task in tasks:
            self.pool.apply_async(run_task, (task, self.result_queue))
            n_tasks += 1

        if self.verbose:
            print(f'Submitted {n_tasks} tasks')

        results = []
        if self.tqdm:
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

        self.pool.close()
        self.pool.join()

        return results
