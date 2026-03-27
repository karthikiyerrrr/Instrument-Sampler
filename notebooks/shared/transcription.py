"""Shared transcription utilities for Group 1 notebooks.

Provides basic-pitch ONNX transcription for monophonic instruments.
"""

import os


def transcribe_wav(wav_path: str, output_mid: str) -> tuple:
    """Transcribe a WAV file to polyphonic MIDI using basic-pitch.

    Uses the ONNX backend for inference (no TensorFlow dependency).

    Args:
        wav_path: Path to the input WAV file.
        output_mid: Path where the output ``.mid`` file will be written.

    Returns:
        Tuple of (model_output, midi_data, note_events).

    Raises:
        FileNotFoundError: If the input WAV does not exist.
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    os.makedirs(os.path.dirname(output_mid) or ".", exist_ok=True)

    from basic_pitch.inference import predict

    print(f"Transcribing {wav_path}")
    model_output, midi_data, note_events = predict(wav_path)

    midi_data.write(output_mid)
    print(f"Transcription complete: {len(note_events)} notes -> {output_mid}")

    return model_output, midi_data, note_events
