class GPUPWorkerException:
    """
    Pickable container for worker exception
    """

    def __init__(self, message, worker_id: int, device_id: int):
        self.message = message
        self.worker_id = worker_id
        self.device_id = device_id


class GPUPWorkerNotInitializedException:
    pass


class GPUPPoolException(Exception):
    pass
