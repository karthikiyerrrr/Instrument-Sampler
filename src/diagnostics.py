"""Diagnostics module — periodic logging of queue depths, timing, and overflow flags.

Runs in a daemon thread and logs a summary line at a configurable interval
so the operator can spot back-pressure, audio dropouts, or processing
bottlenecks without attaching a debugger.
"""

import logging
import queue
import threading
from typing import Callable

from src.live.analyzer import LatencyTracker

logger = logging.getLogger(__name__)


class DiagnosticsMonitor:
    """Collects and logs pipeline health metrics on a timer.

    Attributes:
        _queues: Named queues to monitor for depth.
        _overflow_fn: Callable that returns ``True`` when an audio
            input overflow has occurred since the last check.
        _latency_tracker: Shared tracker for peak aubio hop latency.
        _stop_event: Set by the main thread to signal shutdown.
    """

    def __init__(
        self,
        queues: dict[str, queue.Queue],
        overflow_fn: Callable[[], bool],
        latency_tracker: LatencyTracker,
    ) -> None:
        """Initialise the monitor.

        Args:
            queues: Mapping of human-readable names to the queues to
                monitor (e.g. ``{"live": live_queue, ...}``).
            overflow_fn: A zero-arg callable returning whether an audio
                overflow occurred (typically ``stream.check_overflow``).
            latency_tracker: Shared peak-latency tracker from the live
                analyzer.
        """
        self._queues = queues
        self._overflow_fn = overflow_fn
        self._latency_tracker = latency_tracker
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the monitor loop to exit."""
        self._stop_event.set()

    def run(self, interval_s: float = 5.0) -> None:
        """Periodically log pipeline diagnostics until stopped.

        Intended to be used as a ``threading.Thread`` target.

        Args:
            interval_s: Seconds between diagnostic log lines.
        """
        while not self._stop_event.wait(timeout=interval_s):
            parts: list[str] = []

            for name, q in self._queues.items():
                size = q.qsize()
                maxsize = q.maxsize
                pct = (size / maxsize * 100) if maxsize > 0 else 0.0
                parts.append(f"{name}={size}/{maxsize} ({pct:.0f}%)")

            overflow = self._overflow_fn()
            parts.append(f"overflow={'YES' if overflow else 'no'}")

            peak_us = self._latency_tracker.read_and_reset()
            parts.append(f"aubio_peak={peak_us:.0f}us")

            logger.info("DIAG  %s", "  ".join(parts))
