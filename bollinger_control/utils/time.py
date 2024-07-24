import time


def sleep_s(
    s: float,
):
    """

    Sleep for s seconds.

    Parameters
    ----------
    s : float
        time in seconds to sleep

    """

    start = time.perf_counter_ns()
    if s > 0.001:
        while time.perf_counter_ns() - start < (s * 1e9 * 0.9):
            time.sleep(s / 10)

    # Sleep for the remaining time
    while time.perf_counter_ns() - start < s * 1e9:
        pass
