"""End-to-end test runner for the Group 1 (Continuous Mono) DDSP pipeline.

Runs all six pipeline stages in sequence with programmatic assertions.
Designed to execute in the .venv-ddsp virtual environment.

Usage (from project root):
    source .venv-ddsp/bin/activate
    cd notebooks
    python run_group1_e2e.py

Set SMOKE_TEST = True (default) for a fast 50-step fine-tune suitable for
CI validation. Set to False for a full 2000-step production run.
"""

import os
import sys
import wave
from pathlib import Path

# --------------- Configuration ---------------
SMOKE_TEST: bool = True          # True -> 50 steps; False -> 2000 steps
MODEL_NAME: str = "violin"       # Pretrained base: violin, flute, flute2, trumpet, tenor_saxophone
MODELS_DIR: str = "../models/ddsp_pretrained"
RECORDINGS_DIR: str = "../recordings"
CALIBRATIONS_DIR: str = "../calibrations"
OUTPUT_DIR: str = "../output"
# ---------------------------------------------

# Add notebooks/ to path for shared imports
sys.path.insert(0, str(Path(__file__).parent))

CALIBRATION_WAV: str = os.path.join(RECORDINGS_DIR, "calibration_synthetic.wav")
SESSION_WAV: str = os.path.join(RECORDINGS_DIR, "session_synthetic.wav")
CHECKPOINT_DIR: str = os.path.join(CALIBRATIONS_DIR, "finetune_output")
OUTPUT_WAV: str = os.path.join(OUTPUT_DIR, "cloned_output.wav")
OUTPUT_MID: str = os.path.join(OUTPUT_DIR, "transcription_group1.mid")

FINETUNE_STEPS: int = 50 if SMOKE_TEST else 2000


def _pass(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"[PASS] {label}{suffix}")


def _fail(label: str, reason: str) -> None:
    print(f"[FAIL] {label}  -- {reason}")
    sys.exit(1)


def _wav_duration_s(path: str) -> float:
    """Return duration in seconds of a WAV file."""
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---- Stage 1: Synthetic audio generation ----

def test_synthetic_audio_generation() -> None:
    from shared.test_audio import generate_test_wav

    # Calibration: 35s chromatic scale C4-C6 (25 notes, ~1.4s each)
    generate_test_wav(CALIBRATION_WAV, duration_s=35.0)
    # Session: 10s shorter scale for inference
    generate_test_wav(
        SESSION_WAV,
        duration_s=10.0,
        pitch_sequence=list(range(60, 73)),  # C4-C5
    )

    for path, expected_s in [(CALIBRATION_WAV, 35.0), (SESSION_WAV, 10.0)]:
        if not os.path.isfile(path):
            _fail("Synthetic audio generation", f"{path} not written")
        actual_s = _wav_duration_s(path)
        if abs(actual_s - expected_s) > 0.5:
            _fail(
                "Synthetic audio generation",
                f"{path}: expected ~{expected_s}s, got {actual_s:.1f}s",
            )

    _pass(
        "Synthetic audio generation",
        f"calibration_synthetic.wav: {_wav_duration_s(CALIBRATION_WAV):.1f}s, "
        f"session_synthetic.wav: {_wav_duration_s(SESSION_WAV):.1f}s",
    )


# ---- Stage 2: Preprocessing ----

def test_preprocessing() -> dict:
    from shared.ddsp import chunk_for_training, preprocess_audio

    features = preprocess_audio(CALIBRATION_WAV)

    for key in ("audio", "f0_hz", "f0_confidence", "loudness_db"):
        if key not in features:
            _fail("Preprocessing", f"Missing key '{key}' in features dict")
        if len(features[key]) == 0:
            _fail("Preprocessing", f"Feature '{key}' is empty")

    examples = chunk_for_training(features)
    if len(examples) == 0:
        _fail("Preprocessing", "chunk_for_training returned 0 examples (recording too short?)")

    n_frames = len(features["f0_hz"])
    _pass(
        "Preprocessing",
        f"{n_frames} frames, {len(examples)} training examples",
    )
    return {"features": features, "examples": examples}


# ---- Stage 3: Model download ----

def test_model_download_or_skip() -> str:
    from shared.ddsp import _checkpoint_exists, download_model

    ckpt_dir = os.path.join(MODELS_DIR, MODEL_NAME)
    if _checkpoint_exists(ckpt_dir):
        _pass("Model download", f"'{MODEL_NAME}' already present at {ckpt_dir}")
    else:
        download_model(MODEL_NAME, ckpt_dir)
        if not _checkpoint_exists(ckpt_dir):
            _fail("Model download", f"Checkpoint not found after download at {ckpt_dir}")
        _pass("Model download", f"'{MODEL_NAME}' downloaded to {ckpt_dir}")

    return ckpt_dir


# ---- Stage 4: Fine-tuning ----

def test_finetune(examples: list, base_ckpt_dir: str) -> None:
    from shared.ddsp import finetune

    finetune(
        base_checkpoint_dir=base_ckpt_dir,
        examples=examples,
        output_dir=CHECKPOINT_DIR,
        steps=FINETUNE_STEPS,
        learning_rate=0.0001,
        batch_size=4,
    )

    # Verify a checkpoint was written
    ckpt_files = [
        f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt")
    ] if os.path.isdir(CHECKPOINT_DIR) else []
    if not ckpt_files:
        _fail("Fine-tune", f"No checkpoint files found in {CHECKPOINT_DIR}")

    _pass(
        f"Fine-tune ({FINETUNE_STEPS} steps)",
        f"checkpoint written to {CHECKPOINT_DIR}/",
    )


# ---- Stage 5: Inference ----

def test_inference() -> None:
    from shared.ddsp import run_inference

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_inference(CHECKPOINT_DIR, SESSION_WAV, OUTPUT_WAV)

    if not os.path.isfile(OUTPUT_WAV):
        _fail("Inference", f"Output WAV not written: {OUTPUT_WAV}")

    out_size = os.path.getsize(OUTPUT_WAV)
    if out_size == 0:
        _fail("Inference", "Output WAV is empty (0 bytes)")

    out_duration_s = _wav_duration_s(OUTPUT_WAV)
    if out_duration_s < 1.0:
        _fail("Inference", f"Output WAV too short: {out_duration_s:.1f}s")

    _pass(
        "Inference",
        f"{OUTPUT_WAV}: {out_duration_s:.1f}s, 44100 Hz",
    )


# ---- Stage 6: Transcription ----

def test_transcription() -> None:
    from shared.transcription import transcribe_wav

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, midi_data, note_events = transcribe_wav(SESSION_WAV, OUTPUT_MID)

    if not os.path.isfile(OUTPUT_MID):
        _fail("Transcription", f"MIDI file not written: {OUTPUT_MID}")

    if len(note_events) == 0:
        _fail("Transcription", "No note events found in transcription output")

    _pass("Transcription", f"{OUTPUT_MID}: {len(note_events)} notes")


# ---- Main ----

def main() -> None:
    mode = "SMOKE TEST (50 steps)" if SMOKE_TEST else "FULL RUN (2000 steps)"
    print(f"=== Group 1 DDSP End-to-End Test — {mode} ===\n")

    # Stage 1
    test_synthetic_audio_generation()

    # Stage 2
    preprocess_result = test_preprocessing()

    # Stage 3
    base_ckpt_dir = test_model_download_or_skip()

    # Stage 4
    test_finetune(preprocess_result["examples"], base_ckpt_dir)

    # Stage 5
    test_inference()

    # Stage 6
    test_transcription()

    print("\nAll 6 stages PASSED.")


if __name__ == "__main__":
    main()
