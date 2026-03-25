"""Real-time onset detection and monophonic pitch tracking via aubio.

Runs in a dedicated consumer thread that reads audio frames from the
Dispatcher queue, feeds hop-sized chunks to aubio, and emits
``mido.Message`` objects for detected note events.
"""

import logging
import queue
import threading
import time

import aubio
import mido
import numpy as np

logger = logging.getLogger(__name__)

HOP_SIZE: int = 256
ONSET_BUF_SIZE: int = 512
PITCH_BUF_SIZE: int = 2048
MIN_MIDI_NOTE: int = 1
MIN_CONFIDENCE: float = 0.5
DEFAULT_VELOCITY: int = 100


class LatencyTracker:
    """Thread-safe tracker for peak aubio hop processing latency.

    The live analyzer updates this every hop.  The diagnostics monitor
    reads and resets it each reporting cycle.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._peak_us: float = 0.0

    def record(self, elapsed_us: float) -> None:
        """Record a hop latency sample, keeping the peak.

        Args:
            elapsed_us: Processing time for one hop in microseconds.
        """
        with self._lock:
            if elapsed_us > self._peak_us:
                self._peak_us = elapsed_us

    def read_and_reset(self) -> float:
        """Return the peak latency since the last reset, then reset to zero.

        Returns:
            Peak hop latency in microseconds.
        """
        with self._lock:
            peak = self._peak_us
            self._peak_us = 0.0
            return peak


def live_analyzer(
    audio_queue: queue.Queue[np.ndarray | None],
    midi_queue: queue.Queue[mido.Message | None],
    samplerate_hz: int = 44100,
    latency_tracker: LatencyTracker | None = None,
) -> None:
    """Consume audio frames, detect onsets/pitch, and emit MIDI messages.

    Reads from *audio_queue* (fed by the Dispatcher), accumulates samples
    into hop-sized chunks, and runs aubio onset + YIN pitch detection on
    each hop.  When an onset is detected with sufficient pitch confidence,
    a ``note_on`` message is placed on *midi_queue* (preceded by a
    ``note_off`` for the previous note).

    Send ``None`` on *audio_queue* to signal shutdown.

    Args:
        audio_queue: Bounded queue of float32 mono audio frames from the
            Dispatcher.
        midi_queue: Output queue for ``mido.Message`` objects forwarded
            to the virtual MIDI port.
        samplerate_hz: Sample rate matching the audio stream (default
            44100).
        latency_tracker: Optional shared tracker that receives per-hop
            timing measurements for the diagnostics monitor.
    """
    onset_detector = aubio.onset(
        "default",
        buf_size=ONSET_BUF_SIZE,
        hop_size=HOP_SIZE,
        samplerate=samplerate_hz,
    )
    pitch_detector = aubio.pitch(
        "yin",
        buf_size=PITCH_BUF_SIZE,
        hop_size=HOP_SIZE,
        samplerate=samplerate_hz,
    )
    pitch_detector.set_unit("midi")
    pitch_detector.set_tolerance(0.8)

    buffer = np.empty(0, dtype=np.float32)
    current_note: int | None = None

    while True:
        frame = audio_queue.get()
        if frame is None:
            break

        buffer = np.append(buffer, frame.flatten())

        while len(buffer) >= HOP_SIZE:
            hop = buffer[:HOP_SIZE].astype(np.float32)
            buffer = buffer[HOP_SIZE:]

            t_start = time.perf_counter_ns()
            is_onset = onset_detector(hop)
            midi_pitch = int(round(pitch_detector(hop)[0]))
            confidence = pitch_detector.get_confidence()
            elapsed_us = (time.perf_counter_ns() - t_start) / 1_000
            if latency_tracker is not None:
                latency_tracker.record(elapsed_us)

            if is_onset and midi_pitch >= MIN_MIDI_NOTE and confidence >= MIN_CONFIDENCE:
                if current_note is not None:
                    msg_off = mido.Message(
                        "note_off", note=current_note, velocity=0, channel=0,
                    )
                    midi_queue.put_nowait(msg_off)
                current_note = midi_pitch
                msg_on = mido.Message(
                    "note_on",
                    note=current_note,
                    velocity=DEFAULT_VELOCITY,
                    channel=0,
                )
                midi_queue.put_nowait(msg_on)
                logger.info(
                    "ONSET  note=%d  confidence=%.2f  aubio_us=%.0f",
                    current_note,
                    confidence,
                    elapsed_us,
                )

    if current_note is not None:
        midi_queue.put_nowait(
            mido.Message("note_off", note=current_note, velocity=0, channel=0),
        )
    logger.info("Live analyzer shut down")
