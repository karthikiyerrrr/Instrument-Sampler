"""High-level timbre-cloning API for the main Python 3.11 process."""

import logging
from pathlib import Path

from src.timbre.runner import run_inference, run_training

logger = logging.getLogger(__name__)


def calibrate(
    wav_path: str,
    model_name: str,
    models_dir: str = "models",
    steps: int = 300,
) -> str:
    """Fine-tune a DDSP model on a calibration recording.

    Args:
        wav_path: Path to the calibration WAV file.  Ideally ≥ 3 minutes of
            clean, monophonic playing across the full pitch range.
        model_name: Human-readable identifier for this instrument model.
        models_dir: Base directory where model sub-directories are stored.
        steps: Number of gradient steps (increase for longer recordings).

    Returns:
        Path to the model directory containing ``decoder.onnx``.
    """
    out_dir = str(Path(models_dir) / model_name)
    logger.info("Calibrating '%s' from %s (%d steps) → %s", model_name, wav_path, steps, out_dir)
    return run_training(wav_path, out_dir, steps=steps)


def clone_timbre(
    wav_path: str,
    model_dir: str,
    output_dir: str | None = None,
) -> str:
    """Synthesise a timbre-cloned version of a session recording.

    Args:
        wav_path: Input session WAV recorded by the WAV recorder.
        model_dir: Path to the model directory produced by :func:`calibrate`.
        output_dir: Directory for the cloned WAV.  Defaults to the same
            directory as *wav_path*.

    Returns:
        Path to the cloned ``.wav`` file at 44.1 kHz / 16-bit PCM.
    """
    src = Path(wav_path)
    out_dir = Path(output_dir) if output_dir else src.parent
    out_path = str(out_dir / (src.stem + "_cloned.wav"))
    logger.info("Cloning timbre: %s → %s (model: %s)", wav_path, out_path, model_dir)
    return run_inference(wav_path, model_dir, out_path)
