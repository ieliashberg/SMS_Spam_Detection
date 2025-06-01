import time
import threading
import psutil


class CPUTracker:
    """
    Tracks CPU usage over time in a background thread.
    Use start() before the block to measure, and stop() after to retrieve data.
    """

    def __init__(self):
        self._timestamps = []
        self._cpu_usage = []
        self._start_time = None
        self._flag = False
        self._thread = None

    def _track(self):
        while self._flag:
            self._cpu_usage.append(psutil.cpu_percent(interval=0.1))
            self._timestamps.append(time.time() - self._start_time)

    def start(self):
        """Begin tracking CPU usage."""
        self._timestamps.clear()
        self._cpu_usage.clear()
        self._start_time = time.time()
        self._flag = True
        self._thread = threading.Thread(target=self._track, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop tracking and return (timestamps, cpu_usage) lists."""
        self._flag = False
        if self._thread is not None:
            self._thread.join()
        return self._timestamps, self._cpu_usage
