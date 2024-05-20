import logging
from typing import Optional, Callable

from gpuparallel.exceptions import GPUPWorkerNotInitializedException, GPUPWorkerException
from gpuparallel.utils import log


def _init_worker(gpu_queue: "Queue", init_fn: Optional[Callable] = None):
    global worker_id, device_id, is_broken

    # In case that some worker is crashes, Poll will recreate the worker (and fail to do it properly)
    # We will allow this, but this worker will always return GPUPWorkerNotInitializedException
    # This allows not to hang into endless cycle and fail from the main process
    is_broken = False
    try:
        is_empty_queue = gpu_queue.empty()
    except:
        is_empty_queue = True
    if is_empty_queue:
        log.error("GPU queue is empty, likely Pool tries to recreate failed workers")
        log.error("Current worker is in broken state, it will return GPUPWorkerNotInitializedException for every call")
        is_broken = True
        return

    worker_id, device_id = gpu_queue.get()
    if init_fn is not None:
        init_fn(worker_id=worker_id, device_id=device_id)

    if len(log.handlers) > 0:
        fmt = logging.Formatter(f"[%(levelname)s/Worker-{worker_id}({device_id})]:%(message)s")
        log.handlers[0].setFormatter(fmt)

    log.info(f"Worker #{worker_id} with {device_id} initialized.")


def _run_task(func: Callable, task_idx, result_queue: "Queue"):
    global worker_id, device_id, is_broken

    if is_broken:
        log.error("Current worker is in broken state, it will return GPUPWorkerNotInitializedException")
        result_queue.put((task_idx, GPUPWorkerNotInitializedException))
        return

    log.debug(f"Start task {task_idx}")
    try:
        result = func(worker_id=worker_id, device_id=device_id)
        result_queue.put((task_idx, result))
    except Exception as e:  # We need to return result un any case
        log.error(f"Error during task #{task_idx}", exc_info=True)
        result_queue.put((task_idx, GPUPWorkerException(e, worker_id, device_id)))
    log.debug(f"Result returned for task {task_idx}")
