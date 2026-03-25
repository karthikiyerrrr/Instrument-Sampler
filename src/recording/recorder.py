"""WAV recorder consumer — writes audio frames from the Dispatcher to disk.

Runs in a dedicated thread, subscribing to the Dispatcher's queue and
writing every frame to a standard 16-bit PCM WAV file.  The recorded file
becomes the input for post-processing (basic-pitch, RAVE/DDSP).
"""

import logging
import os
import queue
import wave

import numpy as np

logger = logging.getLogger(__name__)


def wav_recorder(
    q: queue.Queue[np.ndarray | None],
    path: str,
    samplerate_hz: int = 44100,
    channels: int = 1,
) -> None:
    """Drain *q* and write every audio frame to a WAV file.

    Converts float32 samples (range -1.0 .. 1.0) to 16-bit signed PCM.
    Send ``None`` on the queue to stop recording and close the file.

    Args:
        q: Bounded queue of float32 audio frames from the Dispatcher.
        path: Destination file path (e.g. ``recordings/session.wav``).
        samplerate_hz: Sample rate for the WAV header.
        channels: Number of audio channels (1 = mono).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    frames_written: int = 0

    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate_hz)

        while True:
            frame = q.get()
            if frame is None:
                break
            pcm_16 = (frame * 32767).astype(np.int16)
            wf.writeframes(pcm_16.tobytes())
            frames_written += pcm_16.shape[0]

    duration_s = frames_written / samplerate_hz
    logger.info(
        "WAV recorder stopped — %d frames (%.1f s) written to %s",
        frames_written,
        duration_s,
        path,
    )
