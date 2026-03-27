# Instrument-Sampler

Capture live audio from an acoustic instrument, get real-time MIDI feedback, record the session, then classify the instrument into one of three groups and apply group-specific preprocessing, transcription (monophonic, polyphonic, or percussion onset), and timbre-cloned synthesis — all routable into FL Studio.

## How It Works

The system operates in two modes: a **live path** for low-latency feedback during performance, and a **post-recording path** for high-quality offline processing. The post-recording path classifies the instrument into one of three groups and applies group-specific preprocessing, transcription, and synthesis.

```
Mic → [Ingestion] → Dispatcher ─┬─→ [Live Analysis (aubio)] → MIDI → FL Studio
                                 └─→ [WAV Recorder] → .wav
                                          ↓
                                 [Instrument Classifier] → Group 1 / 2 / 3
                                          ↓
                                 [Preprocessing (group-specific)]
                                          ↓
                              ┌───────────┼───────────┐
                         Group 1     Group 2      Group 3
                      Basic Pitch  Basic Pitch  Onset Detection
                       (mono)    (poly, onset)  + Envelopes
                          ↓           ↓              ↓
                        DDSP     Karplus-Strong  WaveNet/NSynth
                          ↓           ↓              ↓
                       .wav/.mid → FL Studio (import)
```

### Live Path

1. **Ingestion** — `sounddevice` streams audio from the mic at 44.1 kHz / float32 / mono. A `Dispatcher` fans each buffer out to all subscribed consumers via thread-safe queues.
2. **Live Analysis** — `aubio` performs real-time onset detection and YIN pitch tracking. Detected notes are immediately emitted as MIDI messages. This path is monophonic and approximate — its purpose is performer feedback, not archival transcription.
3. **MIDI Bridge** — `mido` + `python-rtmidi` open a virtual MIDI output port (`InstrumentSampler`) that FL Studio or any DAW can read in real time. On Windows, use [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) to create the virtual port.
4. **WAV Recording** — Every audio frame is simultaneously written to a timestamped `.wav` file for post-processing.
5. **Diagnostics** — A monitoring thread periodically logs queue depths, overflow flags, and aubio processing latency.

### Post-Recording Path

After a session ends, the system classifies the instrument into one of three groups and applies group-specific processing:

- **Group 1 — Continuous Monophonic** (flute, sax, violin, voice): Wiener filter denoising + 1D U-Net de-reverberation → Basic Pitch monophonic transcription → DDSP (Harmonic + Noise) synthesis. DDSP pipeline is currently in `notebooks/synthesis/` for standalone testing.
- **Group 2 — Polyphonic Plucked/Struck** (guitar, piano, harp): Transient-preserving Wiener filter + HPSS → Basic Pitch polyphonic transcription (onset-tuned) → Differentiable Karplus-Strong synthesis (PyTorch, main process).
- **Group 3 — Unpitched Percussion** (snare, cymbals, kick): Spectral gating → Onset detection + amplitude envelope extraction (no pitch tracking) → WaveNet autoencoder / GAN synthesis (PyTorch, main process).

### Calibration

Before synthesis can match your instrument, you run a one-time **calibration** per instrument:

- **Group 1:** Record a chromatic scale (~30s). Fine-tunes a base DDSP model (e.g., violin, flute) to your instrument's timbre.
- **Group 2:** Record arpeggios across your range (~30s). Fine-tunes the Karplus-Strong body IR + damping parameters.
- **Group 3:** Record varied hits per drum type (~30s). Fine-tunes the WaveNet/NSynth encoder.

Calibration is available through both the CLI and the Web UI. Fine-tuning takes approximately 5–15 minutes on CPU; GPU accelerates this significantly.

### Web UI

A browser-based dashboard provides an alternative to the CLI:

- **Backend** — FastAPI (`src/api/`) exposes REST endpoints for device listing and session start/stop, plus a WebSocket that broadcasts live MIDI events.
- **Frontend** — Next.js 16 app (`web/`) with TypeScript, Tailwind CSS, and App Router. Components include a device selector, record/stop controls, and a canvas-based piano-roll MIDI visualizer.

## Quick Start

### Requirements

- Python 3.11 (all components run in a single venv)
- macOS (native virtual MIDI) or Windows (with [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html))
- Node.js 18+ (for the web UI)

### 1. Set Up the Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the CLI

```bash
# Start with system default mic
python -m src

# List available audio devices
python -m src --list-devices

# Use a specific device
python -m src --device 2

# Custom config file
python -m src --config my_config.yaml

# Override MIDI port name and block size
python -m src --midi-port MyPort --blocksize 512
```

Press **Ctrl+C** to stop. The recorded WAV file is saved to the `recordings/` directory.

### 3. Install Post-Processing Dependencies

```bash
pip install -r requirements-post.txt
```

For DDSP timbre cloning (Group 1 notebooks only):

```bash
pip install -r requirements-ddsp.txt
pip install --no-deps "ddsp>=3.5.0"
```

### 4. Run the Web UI

Start the API server and the Next.js frontend in separate terminals:

```bash
# Terminal 1 — API server
uvicorn src.api.server:app --reload --port 8000

# Terminal 2 — Next.js dev server
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to access the dashboard.

## Configuration

All settings live in `config.yaml` and can be overridden via CLI arguments:

| Setting | Default | Description |
|---------|---------|-------------|
| `samplerate_hz` | 44100 | Audio sample rate in Hz |
| `channels` | 1 | Input channels (1 = mono) |
| `blocksize` | 1024 | Frames per audio callback buffer |
| `recording_dir` | `recordings` | Directory for WAV session files |
| `midi_port_name` | `InstrumentSampler` | Virtual MIDI output port name |
| `diagnostics_interval_s` | 5.0 | Seconds between diagnostics log lines |
| `instrument_group` | `auto` | Instrument group: `auto`, `continuous-mono`, `polyphonic-plucked`, `unpitched-percussion` |

CLI arguments take the highest priority, followed by `config.yaml`, then built-in defaults.

## Project Structure

```
src/
├── main.py                 # CLI entry point
├── config.py               # YAML + CLI config loader
├── diagnostics.py          # Queue depth / latency monitor
├── ingestion/
│   ├── stream.py           # sounddevice InputStream wrapper
│   └── dispatcher.py       # Fan-out dispatcher to consumer queues
├── live/
│   └── analyzer.py         # aubio onset + pitch → MIDI messages
├── recording/
│   └── recorder.py         # WAV file writer consumer
├── bridge/
│   └── midi_sender.py      # mido virtual MIDI port sender
├── transcription/
│   └── transcriber.py      # basic-pitch polyphonic MIDI transcription
└── api/
    ├── server.py           # FastAPI app + CORS
    ├── routes.py           # REST endpoints (devices, session)
    ├── session.py          # SessionManager lifecycle
    └── websocket.py        # WebSocket MIDI broadcast

models/
└── ddsp_pretrained/            # Pre-trained DDSP checkpoints
    ├── violin/                 # Violin model (ckpt-40000)
    └── tenor_saxophone/        # Tenor sax model (ckpt-20000)

notebooks/                      # Standalone testing notebooks
├── shared/
│   ├── ddsp.py                 # DDSP model utilities (loading, training, inference)
│   └── transcription.py        # basic-pitch integration helpers
├── preprocessing/
│   └── group1_ddsp_preprocess.ipynb
├── transcription/
│   └── group1_basic_pitch.ipynb
└── synthesis/
    ├── group1_ddsp_finetune.ipynb
    └── group1_ddsp_inference.ipynb

web/                        # Next.js 16 frontend
├── app/
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── Dashboard.tsx
│   ├── DeviceSelector.tsx
│   ├── SessionControls.tsx
│   └── MidiVisualizer.tsx
└── lib/
    ├── api.ts              # Typed fetch helpers
    └── types.ts            # Shared TypeScript interfaces
```

## Dependencies

Dependencies are split by environment so you only install what you need:

| File | Environment | Purpose | Key Packages |
|------|-------------|---------|-------------|
| `requirements.txt` | Main (Python 3.11) | Live runtime + API server | `sounddevice`, `numpy`, `aubio-ledfx`, `mido`, `python-rtmidi`, `scipy`, `PyYAML`, `fastapi`, `uvicorn` |
| `requirements-post.txt` | Main (Python 3.11) | Offline transcription | `basic-pitch[onnx]`, `onnxruntime`, `librosa` |
| `requirements-ddsp.txt` | Notebooks (Python 3.11) | DDSP timbre cloning (Group 1) | `tensorflow`, `tensorflow-probability`, `torchcrepe`, `gin-config`, `ddsp` (no-deps) |
| `requirements-preprocess.txt` | Planned | Group-specific preprocessing | `noisereduce`, `scipy`, `torch`, `librosa` |
| `requirements-synth.txt` | Planned | Synthesis (Groups 2+3) | `torch`, `torchaudio` |

All app dependencies run in a single Python 3.11 venv. DDSP dependencies are separate (`requirements-ddsp.txt`) and only needed for the synthesis notebooks. The preprocessing and synthesis requirements files will be created when Milestone 5 implementation begins.

## Project Status

**Completed:**

- **Milestone 1** — Ingestion pipeline with Dispatcher fan-out, aubio live analyzer, WAV recorder.
- **Milestone 2** — Live MIDI bridge (virtual port via mido), YAML config loader with CLI overrides, diagnostics monitor.
- **Milestone 3** — Post-processing: polyphonic transcription via `basic-pitch` (ONNX). DDSP timbre cloning pipeline (calibration, fine-tuning, inference) moved to standalone Jupyter notebooks under `notebooks/` for testing. Pre-trained DDSP checkpoints available for violin and tenor saxophone.
- **Milestone 4** — Web UI: FastAPI backend with session lifecycle, device listing, WebSocket MIDI broadcast; Next.js 16 frontend with device selector, record/stop controls, and canvas piano-roll visualizer.

**In Progress:**

- **Notebook testing** — Group 1 DDSP pipeline (preprocessing, transcription, fine-tuning, inference) testing in progress via `notebooks/`. Shared utilities (`notebooks/shared/ddsp.py`, `notebooks/shared/transcription.py`) support the notebook workflows. Groups 2 and 3 notebook pipelines pending.

**Pending** (tracked in directory-specific `CLAUDE.md` files):

- Integration testing and latency benchmarking (Task 13) — see `src/CLAUDE.md`.
- **Milestone 5** — Preprocessing & Multi-Pipeline: instrument group classifier, group-specific preprocessing (`src/preprocessing/` — not yet implemented), onset-only transcription (Group 3), Karplus-Strong synthesis (Group 2), WaveNet/NSynth synthesis (Group 3), `instrument_group` config setting — see `src/CLAUDE.md`.

**Known Issues:**

- `aubio` has no official Python 3.11 wheels — use the `aubio-ledfx` fork instead.
- WaveNet/NSynth (Group 3) training requires GPU and a dataset of ≥500 isolated drum hits per class.
- `src/preprocessing/` directory does not exist yet — will be created as part of Milestone 5.
- `requirements-preprocess.txt` and `requirements-synth.txt` do not exist yet — will be created when Milestone 5 implementation begins.

## License

This project is not yet licensed. All rights reserved.
