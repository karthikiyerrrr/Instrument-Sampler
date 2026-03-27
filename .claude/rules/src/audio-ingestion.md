# Audio Ingestion (`ingestion/`, `recording/`)

- Use `sounddevice` (PortAudio). Never PyAudio.
- All capture MUST use the non-blocking callback API (`sd.InputStream` + `callback`). Never `sd.rec()` or `sd.playrec()`.
- The callback's only job: copy the audio block and hand it to the `Dispatcher`. Never block, never allocate large objects, never do I/O or inference inside the callback.
- Dispatcher fans out each frame to all subscriber queues with **drop-oldest** semantics (get-then-put on `queue.Full`). Never use bare `put_nowait()` without drop-oldest. Never share a single queue between consumers.
- Always set `maxsize` on queues. Never use unbounded queues.
- Use `sd.query_devices()` for device discovery. Never hardcode device index.
- WAV recorder uses `None` sentinel to stop cleanly.
- Stream defaults: 44100 Hz, mono, float32, blocksize 1024 (~23 ms), `latency="low"`.

**Latency targets:** callback < 5 ms, ingestion-to-queue < 25 ms. Monitor `sd.CallbackFlags` for `input_overflow`.
