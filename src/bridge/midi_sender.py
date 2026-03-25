"""Live MIDI bridge — opens a virtual MIDI port and forwards messages from the analyzer.

On macOS/Linux the port is created as a native virtual port via CoreMIDI/ALSA.
On Windows, the user must install loopMIDI and create a port with the
configured name before starting the application.
"""

import logging
import queue
import sys

import mido

logger = logging.getLogger(__name__)

DEFAULT_PORT_NAME: str = "InstrumentSampler"


def open_midi_port(port_name: str = DEFAULT_PORT_NAME) -> mido.ports.BaseOutput:
    """Open a named MIDI output port.

    On macOS and Linux a virtual port is created natively.  On Windows
    the port must already exist (e.g. via loopMIDI).

    Args:
        port_name: Display name for the virtual MIDI port.

    Returns:
        An open ``mido`` output port ready for ``port.send(msg)``.

    Raises:
        OSError: If the port cannot be opened (e.g. loopMIDI not running
            on Windows).
    """
    if sys.platform == "win32":
        port = mido.open_output(port_name)
        logger.info("Opened existing MIDI port '%s' (Windows / loopMIDI)", port_name)
    else:
        port = mido.open_output(port_name, virtual=True)
        logger.info("Created virtual MIDI port '%s'", port_name)
    return port


def midi_sender(
    port: mido.ports.BaseOutput,
    midi_queue: queue.Queue[mido.Message | None],
) -> None:
    """Drain *midi_queue* and forward each message to the MIDI port.

    Runs as a dedicated thread target.  Send ``None`` on the queue to
    signal shutdown; the port is closed on exit.

    Args:
        port: An open mido output port.
        midi_queue: Queue of ``mido.Message`` objects from the live
            analyzer.  ``None`` is the shutdown sentinel.
    """
    try:
        while True:
            msg = midi_queue.get()
            if msg is None:
                break
            port.send(msg)
            logger.debug("MIDI  %s", msg)
    finally:
        port.close()
        logger.info("MIDI port closed")
