"""Instrument-Sampler entry point — wires ingestion, live analysis, and recording.

Usage::

    .venv/bin/python -m src
    .venv/bin/python -m src --device 2
    .venv/bin/python -m src --list-devices
    .venv/bin/python -m src --config my_config.yaml
    .venv/bin/python -m src --midi-port MyPort --blocksize 512
"""

import argparse
import datetime
import logging
import os
import queue
import threading

import mido
import sounddevice as sd

from src.bridge.midi_sender import midi_sender, open_midi_port
from src.config import load_config
from src.diagnostics import DiagnosticsMonitor
from src.ingestion.dispatcher import Dispatcher
from src.ingestion.stream import check_overflow, create_stream
from src.live.analyzer import LatencyTracker, live_analyzer
from src.recording.recorder import wav_recorder

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with audio, MIDI, and config attributes.
    """
    parser = argparse.ArgumentParser(
        description="Instrument-Sampler — live audio capture with aubio analysis and WAV recording.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="PortAudio input device index (omit for system default).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    parser.add_argument(
        "--midi-port",
        type=str,
        default=None,
        help="Name of the virtual MIDI output port (default: InstrumentSampler).",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=None,
        help="Frames per audio callback buffer (default: 1024).",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=None,
        help="Audio sample rate in Hz (default: 44100).",
    )
    parser.add_argument(
        "--recording-dir",
        type=str,
        default=None,
        help="Directory for WAV recordings (default: recordings/).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the live ingestion pipeline until Ctrl+C."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    config = load_config(args)
    midi_port = open_midi_port(config.midi_port_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(config.recording_dir, f"session_{timestamp}.wav")

    dispatcher = Dispatcher()
    live_queue = dispatcher.subscribe(maxsize=64)
    recorder_queue = dispatcher.subscribe(maxsize=128)
    midi_queue: queue.Queue[mido.Message | None] = queue.Queue(maxsize=256)
    latency_tracker = LatencyTracker()

    analyzer_thread = threading.Thread(
        target=live_analyzer,
        args=(live_queue, midi_queue, config.samplerate_hz, latency_tracker),
        name="live-analyzer",
        daemon=True,
    )
    recorder_thread = threading.Thread(
        target=wav_recorder,
        args=(recorder_queue, wav_path, config.samplerate_hz, config.channels),
        name="wav-recorder",
        daemon=True,
    )
    midi_sender_thread = threading.Thread(
        target=midi_sender,
        args=(midi_port, midi_queue),
        name="midi-sender",
        daemon=True,
    )

    diagnostics = DiagnosticsMonitor(
        queues={"live": live_queue, "recorder": recorder_queue, "midi": midi_queue},
        overflow_fn=check_overflow,
        latency_tracker=latency_tracker,
    )
    diagnostics_thread = threading.Thread(
        target=diagnostics.run,
        args=(config.diagnostics_interval_s,),
        name="diagnostics",
        daemon=True,
    )

    analyzer_thread.start()
    recorder_thread.start()
    midi_sender_thread.start()
    diagnostics_thread.start()

    stream = create_stream(dispatcher, config)
    stream.start()

    device_info = sd.query_devices(config.device or sd.default.device[0])
    logger.info(
        "Listening on '%s' | %d Hz | blocksize %d | recording to %s",
        device_info["name"],
        config.samplerate_hz,
        config.blocksize,
        wav_path,
    )
    logger.info("Press Ctrl+C to stop.")

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass

    logger.info("Stopping...")
    stream.stop()
    stream.close()
    diagnostics.stop()

    live_queue.put(None)
    recorder_queue.put(None)
    analyzer_thread.join(timeout=5)
    recorder_thread.join(timeout=5)

    midi_queue.put(None)
    midi_sender_thread.join(timeout=2)
    diagnostics_thread.join(timeout=2)

    logger.info("Session saved to %s", wav_path)


if __name__ == "__main__":
    main()
