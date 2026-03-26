# src/ — Module-Level Rules

## Audio Ingestion (`ingestion/`, `recording/`)

- Use `sounddevice` (PortAudio). Never PyAudio.
- All capture MUST use the non-blocking callback API (`sd.InputStream` + `callback`). Never `sd.rec()` or `sd.playrec()`.
- The callback's only job: copy the audio block and hand it to the `Dispatcher`. Never block, never allocate large objects, never do I/O or inference inside the callback.
- Dispatcher fans out each frame to all subscriber queues with **drop-oldest** semantics (get-then-put on `queue.Full`). Never use bare `put_nowait()` without drop-oldest. Never share a single queue between consumers.
- Always set `maxsize` on queues. Never use unbounded queues.
- Use `sd.query_devices()` for device discovery. Never hardcode device index.
- WAV recorder uses `None` sentinel to stop cleanly.
- Stream defaults: 44100 Hz, mono, float32, blocksize 1024 (~23 ms), `latency="low"`.

**Latency targets:** callback < 5 ms, ingestion-to-queue < 25 ms. Monitor `sd.CallbackFlags` for `input_overflow`.

## Live Analysis (`live/`)

- Use `aubio` (`aubio-ledfx` for Python 3.11+ wheels) for real-time onset detection and monophonic YIN pitch tracking.
- aubio runs in a **consumer thread** reading from the dispatcher queue — never inside the audio callback.
- `onset("default")` with HFC method, `pitch("yin")` with unit `"midi"` and tolerance 0.8.
- Hop size 256 samples. Buffer size 512 (onset) / 2048 (pitch).
- Discard low-confidence pitches (< 0.5) to avoid phantom notes.
- Every `note_on` must be balanced by a `note_off` before the next onset or on shutdown.
- Live path is monophonic, fixed velocity (100), no pitch bends. Sub-semitone detail comes from post-recording transcription.

**Latency budget:** aubio < 1 ms/hop, total live path < 30 ms.

## Post-Recording Transcription (`transcription/`)

- Use `basic-pitch` with **ONNX backend** (`pip install "basic-pitch[onnx]"`). Never load TensorFlow when ONNX is available.
- Use the `predict()` API on the complete WAV file. Never manually chunk audio or stitch results.
- Output is a `pretty_midi.PrettyMIDI` object → write to `.mid` for FL Studio import.
- Run in a background thread or subprocess — never on the main/audio thread.

## MIDI Bridge (`bridge/`)

- Use `mido` + `python-rtmidi`. Virtual port named `InstrumentSampler`.
- macOS/Linux: `virtual=True`. Windows: requires loopMIDI — open by name without `virtual=True`. Always check `sys.platform`.
- MIDI sender runs in a **dedicated thread** consuming from a queue. Never send MIDI from the audio callback thread.
- Send each event individually as it arrives. Never batch MIDI messages.
- Enumerate ports with `mido.get_output_names()`. Never hardcode port names.

## Timbre Cloning (`timbre/`)

Timbre cloning is **offline only** — process the complete WAV in one pass after the session. Never attempt real-time frame-by-frame inference.

**Primary engine: RAVE** — PyTorch-based, Python 3.11 compatible, 48 kHz native, 20x real-time on CPU. Install: `acids-rave`, `torch`, `torchaudio`. Export trained models to TorchScript (`.ts`). Resample output to 44.1 kHz for DAW import. Store models in `models/`, reference from `config.yaml`.

**Fallback: DDSP** — TensorFlow-based, requires isolated Python 3.10 venv (`.venv-ddsp`), invoked as subprocess. Never `import tensorflow` at inference time — export to ONNX/TFLite during training, use `onnxruntime` at inference. DDSP operates at 16 kHz; upsample to 44.1 kHz with `scipy.signal.resample_poly(audio_16k, up=441, down=160)`.

**Anti-patterns:** Never mix RAVE and DDSP deps in the same environment. Never train on < 3 min of audio.
