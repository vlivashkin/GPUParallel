import multiprocessing as mp
from functools import partial

log = mp.get_logger()


def log_to_stderr(log_level='INFO'):
    """
    Shortcut allowing to display logs from workers.

    :param log_level: Set the logging level of this logger.
    """
    mp.log_to_stderr()
    log.setLevel(log_level)


def delayed(func):
    """
    Decorator used to capture the arguments of a function.
    Analogue of joblib's delayed.

    :param func: Function to be captured.
    """

    def wrapper(*args, **kwargs):
        return partial(func, *args, **kwargs)

    return wrapper
