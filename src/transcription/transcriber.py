"""Polyphonic MIDI transcription using basic-pitch with ONNX backend.

Runs on the complete WAV file after a session ends.  Must execute in a
background thread — never on the audio callback thread.
"""

import logging
import os

logger = logging.getLogger(__name__)


def transcribe_wav(wav_path: str, output_mid: str) -> str:
    """Transcribe a WAV file to polyphonic MIDI using basic-pitch.

    Uses the ONNX backend for inference (no TensorFlow dependency).

    Args:
        wav_path: Path to the input WAV file.
        output_mid: Path where the output ``.mid`` file will be written.

    Returns:
        The path to the written MIDI file.

    Raises:
        FileNotFoundError: If the input WAV does not exist.
        RuntimeError: If transcription fails.
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    os.makedirs(os.path.dirname(output_mid) or ".", exist_ok=True)

    from basic_pitch.inference import predict

    logger.info("Transcribing %s → %s", wav_path, output_mid)

    model_output, midi_data, note_events = predict(wav_path)

    midi_data.write(output_mid)
    logger.info(
        "Transcription complete: %d notes → %s",
        len(note_events),
        output_mid,
    )

    return output_mid
