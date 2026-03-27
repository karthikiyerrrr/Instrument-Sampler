# Instrument-Sampler — Project Rules

**PERMANENT DIRECTIVE: The "Project Status" section of this file MUST be updated immediately whenever a task is completed, a new milestone is reached, or a technical blocker is encountered. No exceptions.**

**README DIRECTIVE: Whenever the "Project Status" section below is updated, review `README.md` (project root) and update its "Project Status", "Known Issues", and any other sections affected by the change so the README always reflects the current state of the project.**

---

## Technical Specification

Captures live audio from an acoustic instrument, provides real-time monophonic MIDI feedback, records the session, then produces polyphonic MIDI transcription and timbre-cloned audio as post-processing — all routed into FL Studio.

```
Mic → [Ingestion] → Dispatcher ─┬─→ [Live Analysis (aubio)] → MIDI → FL Studio (virtual MIDI port)
                                 └─→ [WAV Recorder] → .wav file ─┬─→ [Transcription (basic-pitch)] → .mid file → FL Studio
                                                                   └─→ [Timbre Cloning (DDSP subprocess)] → .wav file → FL Studio
                                                                              ↑
                                                                 Calibration recording + base model → fine-tuned model
```

Live path is streaming-first (< 50 ms latency). Post-processing operates on complete files offline. DDSP runs in an isolated Python 3.10 subprocess (`.venv-ddsp`) — no TensorFlow in the main 3.11 process.

### Dependencies

| Group | File | Environment | Packages |
|---|---|---|---|
| Live runtime | `requirements.txt` | Python 3.11 | `sounddevice`, `numpy`, `aubio` (`aubio-ledfx`), `mido`, `python-rtmidi`, `scipy` |
| Transcription | `requirements-post.txt` | Python 3.11 | `basic-pitch[onnx]`, `onnxruntime`, `librosa` |
| Timbre cloning | `requirements-ddsp.txt` | Python 3.10 (`.venv-ddsp`) | `ddsp`, `tensorflow`, `librosa` |

### Constraints

| Constraint | Value |
|---|---|
| Python (main) | 3.11 |
| Python (DDSP) | 3.10, separate venv, invoked as subprocess |
| Live-path latency | < 50 ms end-to-end |
| Primary OS | macOS; Windows via loopMIDI + VB-Cable |

---

## Project Status

- All milestones complete through task 17. Post-processing (tasks 9–12) implemented on `post-processing` branch.
- `.venv` active with live-path + transcription deps. `.venv-ddsp` must be created manually (Python 3.10 + `requirements-ddsp.txt`).
- DDSP timbre cloning via subprocess isolation. RAVE removed from plan.

| # | Task | Status |
|---|------|--------|
| 1 | Install core dependencies (`requirements.txt`) | Done |
| 2 | Scaffold project structure (`src/` package) | Done |
| 3 | Implement Dispatcher + ingestion stream | Done |
| 4 | Implement live aubio onset+pitch analyzer | Done |
| 5 | Implement WAV recorder consumer | Done |
| 6 | Implement live MIDI bridge (mido virtual port) | Done |
| 7 | Implement config loader and CLI (config.yaml, argparse) | Done |
| 8 | Implement diagnostics module (queue depths, timing, overflow flags) | Done |
| 9 | Implement post-recording transcription (basic-pitch) | Done |
| 10 | Set up DDSP subprocess isolation (Python 3.10 venv, subprocess runner) | Done |
| 11 | Implement calibration workflow (CLI + Web UI recording, DDSP fine-tuning) | Done |
| 12 | Implement post-recording timbre cloning (DDSP inference via subprocess) | Done |
| 13 | Integration testing and latency benchmarking | Pending |
| 14 | FastAPI API layer with SessionManager, REST routes, WebSocket MIDI stream | Done |
| 15 | Next.js scaffold with TypeScript, Tailwind, App Router, API proxy | Done |
| 16 | Frontend components: DeviceSelector, SessionControls, MidiVisualizer, Dashboard | Done |
| 17 | Wire page.tsx, .gitignore, status.mdc updates | Done |

### Known Blockers

- **aubio Python 3.11 wheels**: Use `aubio-ledfx` fork.
- **DDSP + Python 3.11**: Resolved via subprocess isolation in `.venv-ddsp`.
- **`.venv-ddsp` setup**: Must be created manually — `python3.10 -m venv .venv-ddsp && .venv-ddsp/bin/pip install -r requirements-ddsp.txt`.

---

## Python Code Style

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections.
- Full type hints on every function signature. Use `np.ndarray` for audio buffers.
- PEP 8: `snake_case` functions/variables, `PascalCase` classes.
- Functions < 50 lines. Absolute imports only.
- Specify units in variable names for time/frequency (e.g., `buffer_ms`, `pitch_hz`).
