# Instrument-Sampler

Capture live audio from an acoustic instrument, get real-time monophonic MIDI feedback, record the session, and produce full polyphonic MIDI transcription and timbre-cloned audio — all routable into FL Studio.

## How It Works

The system operates in two modes: a **live path** for low-latency feedback during performance, and a **post-recording path** for high-quality offline processing.

```
Mic → [Ingestion] → Dispatcher ─┬─→ [Live Analysis (aubio)] → MIDI → FL Studio
                                 └─→ [WAV Recorder] → .wav ─┬─→ [Transcription] → .mid → FL Studio
                                                              └─→ [Timbre Cloning] → .wav → FL Studio
```

### Live Path

1. **Ingestion** — `sounddevice` streams audio from the mic at 44.1 kHz / float32 / mono. A `Dispatcher` fans each buffer out to all subscribed consumers via thread-safe queues.
2. **Live Analysis** — `aubio` performs real-time onset detection and YIN pitch tracking. Detected notes are immediately emitted as MIDI messages. This path is monophonic and approximate — its purpose is performer feedback, not archival transcription.
3. **MIDI Bridge** — `mido` + `python-rtmidi` open a virtual MIDI output port (`InstrumentSampler`) that FL Studio or any DAW can read in real time. On Windows, use [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) to create the virtual port.
4. **WAV Recording** — Every audio frame is simultaneously written to a timestamped `.wav` file for post-processing.
5. **Diagnostics** — A monitoring thread periodically logs queue depths, overflow flags, and aubio processing latency.

### Post-Recording Path *(not yet implemented)*

After a session ends, two offline stages will process the recorded `.wav` file:

- **Transcription** — Spotify's `basic-pitch` (CNN, ONNX backend) produces a polyphonic `.mid` file with accurate note boundaries, velocities, and pitch bends.
- **Timbre Cloning** — RAVE (or DDSP as fallback) synthesizes audio that recreates the instrument's acoustic texture, outputting a new `.wav` file.

### Web UI

A browser-based dashboard provides an alternative to the CLI:

- **Backend** — FastAPI (`src/api/`) exposes REST endpoints for device listing and session start/stop, plus a WebSocket that broadcasts live MIDI events.
- **Frontend** — Next.js 16 app (`web/`) with TypeScript, Tailwind CSS, and App Router. Components include a device selector, record/stop controls, and a canvas-based piano-roll MIDI visualizer.

## Quick Start

### Requirements

- Python 3.11
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

### 3. Run the Web UI

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
├── transcription/          # (Milestone 3 — pending)
├── timbre/                 # (Milestone 3 — pending)
└── api/
    ├── server.py           # FastAPI app + CORS
    ├── routes.py           # REST endpoints (devices, session)
    ├── session.py          # SessionManager lifecycle
    └── websocket.py        # WebSocket MIDI broadcast

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

Dependencies are split into three files so you only install what you need:

| File | Purpose | Key Packages |
|------|---------|-------------|
| `requirements.txt` | Live runtime | `sounddevice`, `numpy`, `aubio-ledfx`, `mido`, `python-rtmidi`, `scipy`, `PyYAML`, `fastapi`, `uvicorn` |
| `requirements-post.txt` | Post-processing | `basic-pitch[onnx]`, `onnxruntime`, `librosa` |
| `requirements-train.txt` | Model training | `acids-rave`, `torch`, `torchaudio` |

## Project Status

**Completed:**

- **Milestone 1** — Ingestion pipeline with Dispatcher fan-out, aubio live analyzer, WAV recorder.
- **Milestone 2** — Live MIDI bridge (virtual port via mido), YAML config loader with CLI overrides, diagnostics monitor.
- **Milestone 4** — Web UI: FastAPI backend with session lifecycle, device listing, WebSocket MIDI broadcast; Next.js 16 frontend with device selector, record/stop controls, and canvas piano-roll visualizer.

**In Progress / Pending:**

- **Milestone 3** — Post-processing: polyphonic transcription via `basic-pitch`, timbre cloning via RAVE/DDSP, integration testing and latency benchmarking.

**Known Issues:**

- `aubio` has no official Python 3.11 wheels — use the `aubio-ledfx` fork instead.
- DDSP has TensorFlow compatibility issues on Python 3.11 — RAVE is the recommended alternative.

## License

This project is not yet licensed. All rights reserved.
