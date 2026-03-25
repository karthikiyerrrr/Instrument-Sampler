"""Audio input stream built on ``sounddevice``.

Creates a callback-based ``sd.InputStream`` whose only job is to copy
each audio block into the :class:`Dispatcher` for downstream consumers.
"""

import logging
import threading

import numpy as np
import sounddevice as sd

from src.config import AppConfig
from src.ingestion.dispatcher import Dispatcher

logger = logging.getLogger(__name__)

_overflow_flag = threading.Event()


def _make_callback(
    dispatcher: Dispatcher,
) -> sd.RawInputStream:
    """Return a sounddevice callback that feeds *dispatcher*.

    Args:
        dispatcher: The fan-out dispatcher to receive copied frames.

    Returns:
        A callback function compatible with ``sd.InputStream(callback=...)``.
    """

    def audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status.input_overflow:
            _overflow_flag.set()
        dispatcher.dispatch(indata.copy())

    return audio_callback


def create_stream(
    dispatcher: Dispatcher,
    config: AppConfig,
) -> sd.InputStream:
    """Build and return an ``sd.InputStream`` wired to *dispatcher*.

    The stream is created but **not** started — call ``stream.start()``
    after all consumers have subscribed.

    Args:
        dispatcher: Fan-out dispatcher for audio frames.
        config: Application configuration (sample rate, block size, etc.).

    Returns:
        A configured ``sd.InputStream`` ready to be started.
    """
    callback = _make_callback(dispatcher)

    stream = sd.InputStream(
        samplerate=config.samplerate_hz,
        channels=config.channels,
        dtype=config.dtype,
        blocksize=config.blocksize,
        latency="low",
        device=config.device,
        callback=callback,
    )
    return stream


def check_overflow() -> bool:
    """Return ``True`` and clear the flag if an input overflow was detected.

    Returns:
        Whether an overflow occurred since the last call.
    """
    if _overflow_flag.is_set():
        _overflow_flag.clear()
        return True
    return False
