"""Small timing helper used by benchmarks and algorithm results."""

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimerResult:
    elapsed_s: float = 0.0


@contextmanager
def timer(result: TimerResult | None = None):
    """Measure elapsed wall-clock time in seconds.

    Example:
        result = TimerResult()
        with timer(result):
            do_work()
        print(result.elapsed_s)
    """
    result = result or TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_s = time.perf_counter() - start
