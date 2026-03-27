"""Subprocess runner — launches ddsp_worker scripts inside .venv-ddsp (Python 3.10).

The main Python 3.11 process must never import TensorFlow or DDSP directly;
all timbre work happens via this runner.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (src/timbre/runner.py).
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_VENV_DDSP: Path = _PROJECT_ROOT / ".venv-ddsp"
_WORKER_DIR: Path = _PROJECT_ROOT / "ddsp_worker"

# Minimum WAV duration for reliable DDSP training.
MIN_CALIBRATION_SECS: float = 30.0
RECOMMENDED_CALIBRATION_SECS: float = 180.0  # 3 minutes


def _python() -> str:
    """Return the Python interpreter path inside .venv-ddsp.

    Raises:
        FileNotFoundError: If .venv-ddsp does not exist.
    """
    candidates = [
        _VENV_DDSP / "bin" / "python",       # macOS / Linux
        _VENV_DDSP / "Scripts" / "python.exe",  # Windows
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f".venv-ddsp not found at {_VENV_DDSP}. "
        "Create it with: python3.10 -m venv .venv-ddsp && "
        ".venv-ddsp/bin/pip install -r requirements-ddsp.txt"
    )


def run_training(
    wav_path: str,
    out_dir: str,
    steps: int = 300,
    frame_rate: int = 250,
) -> str:
    """Fine-tune a DDSP model on a calibration recording.

    Args:
        wav_path: Path to the calibration WAV file.
        out_dir: Directory where ``decoder.onnx`` and ``meta.json`` will be written.
        steps: Gradient update steps (default 300; increase for longer recordings).
        frame_rate: Feature frame rate in fps used during training.

    Returns:
        Path to the output model directory (same as *out_dir*).

    Raises:
        FileNotFoundError: If .venv-ddsp is not set up.
        RuntimeError: If the training subprocess exits with a non-zero code.
    """
    python = _python()
    script = str(_WORKER_DIR / "train.py")
    cmd = [
        python, script,
        "--wav", wav_path,
        "--out", out_dir,
        "--steps", str(steps),
        "--frame-rate", str(frame_rate),
    ]
    logger.info("Launching DDSP training: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info("[ddsp/train] %s", line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            logger.error("[ddsp/train] %s", line)
        raise RuntimeError(
            f"DDSP training failed (exit {result.returncode}). "
            f"stderr: {result.stderr[-500:]}"
        )
    return out_dir


def run_inference(
    wav_path: str,
    model_dir: str,
    out_path: str,
) -> str:
    """Synthesise a timbre-cloned WAV via the ONNX decoder.

    Args:
        wav_path: Input session WAV.
        model_dir: Directory containing ``decoder.onnx`` (output of :func:`run_training`).
        out_path: Destination path for the cloned WAV.

    Returns:
        Path to the generated cloned WAV file (same as *out_path*).

    Raises:
        FileNotFoundError: If .venv-ddsp is not set up.
        RuntimeError: If the inference subprocess exits with a non-zero code.
    """
    python = _python()
    script = str(_WORKER_DIR / "infer.py")
    cmd = [
        python, script,
        "--wav", wav_path,
        "--model", model_dir,
        "--out", out_path,
    ]
    logger.info("Launching DDSP inference: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info("[ddsp/infer] %s", line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            logger.error("[ddsp/infer] %s", line)
        raise RuntimeError(
            f"DDSP inference failed (exit {result.returncode}). "
            f"stderr: {result.stderr[-500:]}"
        )
    return out_path
