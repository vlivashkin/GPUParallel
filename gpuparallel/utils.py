import multiprocessing as mp
import os
import signal
import subprocess
from functools import partial

log = mp.get_logger()


def log_to_stderr(log_level="INFO", force=False):
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


class TqdmStub:
    """
    Stub for the case if tqdm is disabled.
    """

    def __init__(self, total=None, *args, **kwargs):
        self.total = total

    def __enter__(self):
        self.current = 0
        return self

    def update(self, increment):
        self.current += increment

    def __exit__(self, type, value, traceback):
        pass


def import_tqdm(progressbar=True):
    if progressbar:
        try:
            from tqdm.auto import tqdm

            return tqdm
        except ImportError:
            log.warning("Can't load tqdm")
    return TqdmStub