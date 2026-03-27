# Instrument-Sampler — Project Rules

**PERMANENT DIRECTIVE: The "Project Status" section of this file MUST be updated immediately whenever a task is completed, a new milestone is reached, or a technical blocker is encountered. No exceptions.**

**README DIRECTIVE: Whenever the "Project Status" section below is updated, review `README.md` (project root) and update its "Project Status", "Known Issues", and any other sections affected by the change so the README always reflects the current state of the project.**

---

## Technical Specification

Captures live audio from an acoustic instrument, provides real-time monophonic MIDI feedback, records the session, then classifies the instrument into one of three groups and applies group-specific preprocessing, transcription, and timbre-cloned synthesis — all routed into FL Studio.

```
Mic → [Ingestion] → Dispatcher ─┬─→ [Live Analysis (aubio)] → MIDI → FL Studio (virtual MIDI port)
                                 └─→ [WAV Recorder] → .wav file
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
                     (.venv-ddsp)  (PyTorch)      (PyTorch)
                          ↓           ↓              ↓
                       .wav/.mid → FL Studio (import)
```

**Instrument groups:** (1) Continuous monophonic — flute, sax, violin, voice; (2) Polyphonic plucked/struck — guitar, piano, harp; (3) Unpitched percussion — snare, cymbals, kick.

Live path is streaming-first (< 50 ms latency). Post-processing operates on complete files offline. Synthesis engines (Karplus-Strong, WaveNet) will run in the main Python 3.11 process. DDSP (Group 1) has been moved to standalone notebooks under `notebooks/synthesis/` for testing.

### Dependencies

| Group | File | Environment | Packages |
|---|---|---|---|
| Live runtime | `requirements.txt` | Python 3.11 | `sounddevice`, `numpy`, `aubio` (`aubio-ledfx`), `mido`, `python-rtmidi`, `scipy` |
| Transcription | `requirements-post.txt` | Python 3.11 | `basic-pitch[onnx]`, `onnxruntime`, `librosa` |
| Preprocessing | `requirements-preprocess.txt` | Python 3.11 | `noisereduce`, `scipy`, `torch`, `librosa` |
| Synthesis (Groups 2+3) | `requirements-synth.txt` | Python 3.11 | `torch`, `torchaudio` |
| Timbre cloning (Group 1) | `requirements-ddsp.txt` | Python 3.11 (notebooks only) | `ddsp` (no-deps), `tensorflow`, `torchcrepe` |

### Constraints

| Constraint | Value |
|---|---|
| Python | 3.11 (main app); DDSP notebooks may use 3.10+ |
| Instrument groups | 3: continuous-mono, polyphonic-plucked, unpitched-percussion |
| Live-path latency | < 50 ms end-to-end |
| Primary OS | macOS; Windows via loopMIDI + VB-Cable |

---

## Project Status

- Milestones 1–4 code complete (DDSP pipeline moved from app to `notebooks/` for standalone testing). Integration testing (Task 13) pending.
- Milestone 5 planned: instrument-group classification, preprocessing, group-specific transcription and synthesis.
- `.venv` active with live-path deps. Post-processing deps in `requirements-post.txt`. DDSP deps in `requirements-ddsp.txt` (notebooks only).

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
| 14 | FastAPI API layer with SessionManager, REST routes, WebSocket MIDI stream | Done |
| 15 | Next.js scaffold with TypeScript, Tailwind, App Router, API proxy | Done |
| 16 | Frontend components: DeviceSelector, SessionControls, MidiVisualizer, Dashboard | Done |
| 17 | Wire page.tsx, .gitignore, status.mdc updates | Done |

Incomplete tasks tracked in directory-specific `CLAUDE.md` files (`src/`, `notebooks/`).

### Known Blockers

- **aubio Python 3.11 wheels**: Use `aubio-ledfx` fork.
- **WaveNet/NSynth (Group 3)**: Training requires GPU and a dataset of ≥500 isolated drum hits per class.

---

## Python Code Style

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections.
- Full type hints on every function signature. Use `np.ndarray` for audio buffers.
- PEP 8: `snake_case` functions/variables, `PascalCase` classes.
- Functions < 50 lines. Absolute imports only.
- Specify units in variable names for time/frequency (e.g., `buffer_ms`, `pitch_hz`).
