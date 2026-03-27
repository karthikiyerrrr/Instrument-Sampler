"""Post-recording polyphonic MIDI transcription via basic-pitch (ONNX backend)."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Onset / frame detection thresholds — tuned for acoustic instruments.
_ONSET_THRESHOLD: float = 0.5
_FRAME_THRESHOLD: float = 0.3
# Minimum note duration in milliseconds to suppress ghost notes.
_MIN_NOTE_LENGTH_MS: int = 58
# Frequency range spanning a full acoustic instrument (C1–C8).
_MIN_FREQUENCY_HZ: float = 32.7
_MAX_FREQUENCY_HZ: float = 4186.0


def transcribe_wav(wav_path: str, output_dir: str | None = None) -> str:
    """Transcribe a WAV recording to a polyphonic MIDI file.

    Uses ``basic-pitch`` with the ONNX backend so TensorFlow is never
    imported at inference time.  The entire WAV is processed in one pass;
    chunking or stitching is never performed.

    Args:
        wav_path: Path to the source WAV file produced by the WAV recorder.
        output_dir: Directory to write the ``.mid`` file.  Defaults to the
            same directory as ``wav_path``.

    Returns:
        Absolute path to the generated ``.mid`` file.

    Raises:
        ImportError: If ``basic-pitch[onnx]`` is not installed in the active
            environment.
        FileNotFoundError: If ``wav_path`` does not exist.
        RuntimeError: If the transcription subprocess raises an exception.
    """
    try:
        from basic_pitch.inference import predict  # type: ignore[import]
        from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "basic-pitch is not installed.  Run: pip install 'basic-pitch[onnx]'"
        ) from exc

    wav = Path(wav_path).resolve()
    if not wav.exists():
        raise FileNotFoundError(f"WAV file not found: {wav}")

    out_dir = Path(output_dir).resolve() if output_dir else wav.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_path = out_dir / wav.with_suffix(".mid").name
    logger.info("Transcribing %s → %s", wav, midi_path)

    try:
        _model_output, midi_data, note_events = predict(
            str(wav),
            ICASSP_2022_MODEL_PATH,
            onset_threshold=_ONSET_THRESHOLD,
            frame_threshold=_FRAME_THRESHOLD,
            minimum_note_length=_MIN_NOTE_LENGTH_MS,
            minimum_frequency=_MIN_FREQUENCY_HZ,
            maximum_frequency=_MAX_FREQUENCY_HZ,
        )
    except Exception as exc:
        raise RuntimeError(f"basic-pitch prediction failed: {exc}") from exc

    midi_data.write(str(midi_path))
    logger.info("Transcription complete: %d note events → %s", len(note_events), midi_path)
    return str(midi_path)
