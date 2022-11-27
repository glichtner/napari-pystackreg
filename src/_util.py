import sys
import threading
import types

try:  # pragma: no cover
    import pydevd  # noqa: F401

    DEBUGGING = True
except ImportError:
    DEBUGGING = False


def _work_qthread_aware_coverage(worker):  # pragma: no cover
    """
    Helper function to patch the worker's work method to be aware
    of the QThread it is running in.
    See patch_worker_for_coverage().

    :param worker: superqt.utils.WorkerBase
        (e.g. from @thread_worker decorator)
    :return: patched work method
    """
    original_func = worker.work

    def patched_func(_):
        sys.settrace(threading._trace_hook)
        return original_func()

    return types.MethodType(patched_func, worker)


def patch_worker_for_coverage(worker):  # pragma: no cover
    """
    Patch the worker's work method to be aware of the QThread
    it is running in. This is necessary to make coverage.py
    work with QThreads.
    See https://github.com/nedbat/coveragepy/issues/686

    :param worker: superqt.utils.WorkerBase
        (e.g. from @thread_worker decorator)
    :return: worker with patched work() method
    """
    if not DEBUGGING:
        worker.work = _work_qthread_aware_coverage(worker)
