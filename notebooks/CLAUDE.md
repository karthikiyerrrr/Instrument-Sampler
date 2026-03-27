# notebooks/

Standalone Jupyter notebooks for synthesis pipeline experimentation and testing. These run **outside** the main app — used for experimentation, not imported at runtime.

## Structure

- `preprocessing/` — DDSP-specific audio preprocessing
- `transcription/` — Basic Pitch transcription experiments
- `synthesis/` — DDSP fine-tuning and inference
- `shared/` — Helper modules reused across notebooks (`ddsp.py`, `transcription.py`)

## Dependencies

Install from project root: `pip install -r requirements-ddsp.txt` then `pip install --no-deps "ddsp>=3.5.0"`.

Requires Python 3.11 (3.10+ compatible). Uses TensorFlow (not PyTorch) — this is separate from the main app's torch-based synthesis.

---

## Pending Tasks

| # | Task | Status |
|---|------|--------|
| 10 | Set up DDSP in notebook venv (ddsp no-deps + torchcrepe shim) | Done |
| 11 | Implement calibration workflow (preprocessing + fine-tuning) | Done |
| 12 | Implement post-recording timbre cloning (DDSP inference) | Done |
| N1 | Test Group 1 (Continuous Mono) DDSP pipeline end-to-end | Done |
| N2 | Test Group 2 (Polyphonic Plucked/Struck) Karplus-Strong pipeline | Pending |
| N3 | Test Group 3 (Unpitched Percussion) WaveNet/NSynth pipeline | Pending |
