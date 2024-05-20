import multiprocessing as mp
from functools import partial

log = mp.get_logger()


def log_to_stderr(log_level='INFO'):
    """
    Shortcut allowing to display logs from workers.
    """
    mp.log_to_stderr()
    log.setLevel(log_level)


def delayed(func):
    """
    Analogue of joblib's delayed.
    """

    def wrapper(*args, **kwargs):
        return partial(func, *args, **kwargs)

    return wrapper
