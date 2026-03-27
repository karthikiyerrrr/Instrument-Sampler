# src/

Core Python application — live audio capture, analysis, MIDI routing, offline transcription, preprocessing, synthesis, and FastAPI server. Rules for each module are in `.claude/rules/src/`.

## Structure

- `ingestion/` — sounddevice audio capture and Dispatcher fan-out
- `recording/` — WAV file writer consumer
- `live/` — aubio real-time onset detection and pitch tracking
- `bridge/` — mido virtual MIDI port sender
- `transcription/` — basic-pitch offline transcription (post-recording)
- `preprocessing/` — Instrument group classifier and group-specific audio cleaning
- `api/` — FastAPI server, REST routes, WebSocket MIDI broadcast, session lifecycle
- `config.py` — YAML + CLI config loader
- `diagnostics.py` — Queue depth / latency monitor
- `main.py` — CLI entry point

## Dependencies

| File | Purpose | Key Packages |
|------|---------|-------------|
| `requirements.txt` | Live runtime + API server | `sounddevice`, `numpy`, `aubio-ledfx`, `mido`, `python-rtmidi`, `scipy`, `PyYAML`, `fastapi`, `uvicorn` |
| `requirements-post.txt` | Offline transcription | `basic-pitch[onnx]`, `onnxruntime`, `librosa` |
| `requirements-preprocess.txt` | Preprocessing | `noisereduce`, `scipy`, `torch`, `librosa` |
| `requirements-synth.txt` | Synthesis (Groups 2+3) | `torch`, `torchaudio` |

All run in a single Python 3.11 venv.

---

## Pending Tasks

| # | Task | Status |
|---|------|--------|
| 13 | Integration testing and latency benchmarking | Pending |
| 18 | Implement instrument group classifier (`src/preprocessing/classifier.py`) | Pending |
| 19 | Implement preprocessing pipeline (Wiener, U-Net, HPSS, spectral gating) | Pending |
| 20 | Implement onset-only transcription for Group 3 | Pending |
| 21 | Implement Karplus-Strong differentiable synthesis (Group 2) | Pending |
| 22 | Implement WaveNet/NSynth synthesis (Group 3) | Pending |
| 23 | Add `instrument_group` config + CLI flag | Pending |
