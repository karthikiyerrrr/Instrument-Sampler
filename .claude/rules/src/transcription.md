# Post-Recording Transcription (`transcription/`)

Transcription approach varies by instrument group. All run offline on the preprocessed WAV.

- **Group 1 (Continuous Mono):** `basic-pitch` with ONNX backend. Monophonic output expected. Use `predict()` on the complete file. Output: `pretty_midi.PrettyMIDI` → `.mid`.
- **Group 2 (Polyphonic Plucked/Struck):** `basic-pitch` with ONNX backend, tuned for onset sensitivity on HPSS-preprocessed audio. Polyphonic output with per-note onset, offset, pitch, velocity. Output: `pretty_midi.PrettyMIDI` → `.mid`.
- **Group 3 (Unpitched Percussion):** No pitch tracking. `librosa.onset.onset_detect()` with spectral flux for onset times + RMS amplitude envelope extraction per onset. Optional hit classification CNN on mel-spectrograms. Output: list of (onset_time, amplitude_envelope) tuples, NOT standard MIDI.
- Run in a background thread or subprocess — never on the main/audio thread.
