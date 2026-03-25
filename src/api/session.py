"""Session lifecycle manager for the audio pipeline.

Wraps the same Dispatcher / live-analyzer / WAV-recorder / MIDI-sender
pipeline that ``src.main`` runs from the CLI, but exposes it as
start/stop methods callable from the FastAPI layer.

A ``MidiFanout`` thread sits between the live analyzer and the MIDI
sender so that every MIDI event is also placed on an ``asyncio.Queue``
readable by the WebSocket handler.
"""

import asyncio
import datetime
import logging
import os
import queue
import threading
import time
from typing import Any, AsyncGenerator

import mido
import numpy as np
import sounddevice as sd

from src.bridge.midi_sender import midi_sender, open_midi_port
from src.config import AppConfig
from src.diagnostics import DiagnosticsMonitor
from src.ingestion.dispatcher import Dispatcher
from src.ingestion.stream import check_overflow, create_stream
from src.live.analyzer import LatencyTracker, live_analyzer
from src.recording.recorder import wav_recorder

logger = logging.getLogger(__name__)


def _midi_fanout(
    source: queue.Queue[mido.Message | None],
    sender_q: queue.Queue[mido.Message | None],
    broadcast_q: queue.Queue[dict[str, Any] | None],
    start_time: float,
) -> None:
    """Read MIDI messages from *source* and tee to two downstream queues.

    Converts each ``mido.Message`` into a JSON-friendly dict before
    placing it on *broadcast_q* so the WebSocket handler can serialise
    it directly.

    Args:
        source: Queue fed by the live analyzer.
        sender_q: Queue consumed by the MIDI sender thread.
        broadcast_q: Queue consumed by the WebSocket broadcaster.
        start_time: ``time.monotonic()`` at session start, used to
            compute ``time_ms`` offsets.
    """
    while True:
        msg = source.get()
        if msg is None:
            sender_q.put(None)
            broadcast_q.put(None)
            break
        sender_q.put(msg)
        event: dict[str, Any] = {
            "type": msg.type,
            "note": msg.note,
            "velocity": msg.velocity,
            "time_ms": round((time.monotonic() - start_time) * 1000, 1),
        }
        try:
            broadcast_q.put_nowait(event)
        except queue.Full:
            try:
                broadcast_q.get_nowait()
            except queue.Empty:
                pass
            broadcast_q.put_nowait(event)


class SessionManager:
    """Controls the audio pipeline lifecycle for the API layer.

    Only one session can be active at a time.  Call :meth:`start` to
    begin recording / analysis and :meth:`stop` to tear everything down.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._active = False
        self._wav_path: str | None = None
        self._start_time: float = 0.0

        self._stream: sd.InputStream | None = None
        self._dispatcher: Dispatcher | None = None
        self._threads: list[threading.Thread] = []
        self._midi_port: mido.ports.BaseOutput | None = None

        self._live_queue: queue.Queue[np.ndarray | None] | None = None
        self._recorder_queue: queue.Queue[np.ndarray | None] | None = None
        self._analyzer_midi_q: queue.Queue[mido.Message | None] | None = None
        self._sender_midi_q: queue.Queue[mido.Message | None] | None = None
        self._broadcast_q: queue.Queue[dict[str, Any] | None] | None = None

    @property
    def active(self) -> bool:
        """Whether a session is currently recording."""
        return self._active

    @property
    def wav_path(self) -> str | None:
        """Path to the WAV file for the current or most recent session."""
        return self._wav_path

    def start(self, device_index: int | None = None) -> str:
        """Start a new recording session.

        Args:
            device_index: PortAudio device index, or ``None`` for the
                system default.

        Returns:
            Path to the WAV file being recorded.

        Raises:
            RuntimeError: If a session is already active.
        """
        with self._lock:
            if self._active:
                raise RuntimeError("A session is already active")

            config = self._config
            if device_index is not None:
                config.device = device_index

            self._midi_port = open_midi_port(config.midi_port_name)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._wav_path = os.path.join(
                config.recording_dir, f"session_{timestamp}.wav",
            )

            self._dispatcher = Dispatcher()
            self._live_queue = self._dispatcher.subscribe(maxsize=64)
            self._recorder_queue = self._dispatcher.subscribe(maxsize=128)

            self._analyzer_midi_q = queue.Queue(maxsize=256)
            self._sender_midi_q = queue.Queue(maxsize=256)
            self._broadcast_q = queue.Queue(maxsize=512)

            self._start_time = time.monotonic()
            latency_tracker = LatencyTracker()

            threads: list[threading.Thread] = []

            threads.append(threading.Thread(
                target=live_analyzer,
                args=(
                    self._live_queue,
                    self._analyzer_midi_q,
                    config.samplerate_hz,
                    latency_tracker,
                ),
                name="live-analyzer",
                daemon=True,
            ))
            threads.append(threading.Thread(
                target=wav_recorder,
                args=(
                    self._recorder_queue,
                    self._wav_path,
                    config.samplerate_hz,
                    config.channels,
                ),
                name="wav-recorder",
                daemon=True,
            ))
            threads.append(threading.Thread(
                target=_midi_fanout,
                args=(
                    self._analyzer_midi_q,
                    self._sender_midi_q,
                    self._broadcast_q,
                    self._start_time,
                ),
                name="midi-fanout",
                daemon=True,
            ))
            threads.append(threading.Thread(
                target=midi_sender,
                args=(self._midi_port, self._sender_midi_q),
                name="midi-sender",
                daemon=True,
            ))

            diagnostics = DiagnosticsMonitor(
                queues={
                    "live": self._live_queue,
                    "recorder": self._recorder_queue,
                    "midi": self._analyzer_midi_q,
                },
                overflow_fn=check_overflow,
                latency_tracker=latency_tracker,
            )
            self._diagnostics = diagnostics
            threads.append(threading.Thread(
                target=diagnostics.run,
                args=(config.diagnostics_interval_s,),
                name="diagnostics",
                daemon=True,
            ))

            for t in threads:
                t.start()
            self._threads = threads

            self._stream = create_stream(self._dispatcher, config)
            self._stream.start()

            self._active = True

            device_info = sd.query_devices(
                config.device or sd.default.device[0],
            )
            logger.info(
                "Session started on '%s' | %d Hz | recording to %s",
                device_info["name"],
                config.samplerate_hz,
                self._wav_path,
            )
            return self._wav_path

    def stop(self) -> dict[str, Any]:
        """Stop the active session and return summary info.

        Returns:
            Dict with ``wav_path`` and ``duration_s``.

        Raises:
            RuntimeError: If no session is active.
        """
        with self._lock:
            if not self._active:
                raise RuntimeError("No active session to stop")

            self._stream.stop()
            self._stream.close()
            self._diagnostics.stop()

            self._live_queue.put(None)
            self._recorder_queue.put(None)

            for t in self._threads:
                t.join(timeout=5)

            self._active = False
            duration_s = round(
                (time.monotonic() - self._start_time), 2,
            )
            logger.info(
                "Session stopped — %.1f s — %s",
                duration_s,
                self._wav_path,
            )
            return {"wav_path": self._wav_path, "duration_s": duration_s}

    async def midi_events(self) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator yielding MIDI event dicts for WebSocket broadcast.

        Bridges the threaded ``_broadcast_q`` into the async world by
        polling in short intervals.  Yields until a ``None`` sentinel
        arrives (session stopped) or the session becomes inactive.
        """
        bq = self._broadcast_q
        if bq is None:
            return
        loop = asyncio.get_event_loop()
        while self._active:
            try:
                event = await loop.run_in_executor(None, bq.get, True, 0.1)
            except queue.Empty:
                continue
            if event is None:
                break
            yield event
