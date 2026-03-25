"""Fan-out dispatcher that copies each audio frame to all subscriber queues.

The Dispatcher sits between the ``sd.InputStream`` callback and the consumer
threads (live analyzer, WAV recorder).  Each subscriber gets its own bounded
``queue.Queue``.  When a queue is full the oldest frame is evicted so the
audio callback never blocks.
"""

import queue
import threading

import numpy as np


class Dispatcher:
    """Thread-safe audio frame fan-out with drop-oldest back-pressure.

    Example::

        dispatcher = Dispatcher()
        live_q = dispatcher.subscribe(maxsize=64)
        rec_q  = dispatcher.subscribe(maxsize=128)
        # In the audio callback:
        dispatcher.dispatch(indata.copy())
    """

    def __init__(self) -> None:
        self._subscribers: list[queue.Queue[np.ndarray | None]] = []
        self._lock = threading.Lock()

    def subscribe(self, maxsize: int = 64) -> queue.Queue[np.ndarray | None]:
        """Create and register a new bounded subscriber queue.

        Args:
            maxsize: Maximum number of frames the queue can hold before
                drop-oldest eviction kicks in.

        Returns:
            A new ``queue.Queue`` that will receive every dispatched frame.
        """
        q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[np.ndarray | None]) -> None:
        """Remove a subscriber queue so it no longer receives frames.

        Args:
            q: The queue previously returned by :meth:`subscribe`.
        """
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def dispatch(self, frame: np.ndarray) -> None:
        """Fan out *frame* to every subscriber queue.

        If a subscriber queue is full the oldest frame is silently dropped to
        make room.  This method is called from the audio callback and must
        never block.

        Args:
            frame: Audio buffer (typically float32, mono).
        """
        for q in self._subscribers:
            try:
                q.put_nowait(frame)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait(frame)
