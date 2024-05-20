import multiprocessing as mp
from functools import partial

log = mp.get_logger()


def log_to_stderr(log_level='INFO', force=False):
    """
    Shortcut allowing to display logs from workers.

    :param log_level: Set the logging level of this logger.
    :param force: Add handler even there are other handlers already.
    """
    if not log.handlers or force:
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
