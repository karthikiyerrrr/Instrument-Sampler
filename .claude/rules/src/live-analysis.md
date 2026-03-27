# Live Analysis (`live/`)

- Use `aubio` (`aubio-ledfx` for Python 3.11+ wheels) for real-time onset detection and monophonic YIN pitch tracking.
- aubio runs in a **consumer thread** reading from the dispatcher queue — never inside the audio callback.
- `onset("default")` with HFC method, `pitch("yin")` with unit `"midi"` and tolerance 0.8.
- Hop size 256 samples. Buffer size 512 (onset) / 2048 (pitch).
- Discard low-confidence pitches (< 0.5) to avoid phantom notes.
- Every `note_on` must be balanced by a `note_off` before the next onset or on shutdown.
- Live path is monophonic, fixed velocity (100), no pitch bends. Sub-semitone detail comes from post-recording transcription.

**Latency budget:** aubio < 1 ms/hop, total live path < 30 ms.
