# Preprocessing (`preprocessing/`)

Preprocessing runs **offline** on the complete recorded WAV file before transcription and synthesis. The classifier determines the instrument group from `config.yaml` (`instrument_group` setting) or automatic audio feature analysis.

- **Group 1 (Continuous Mono):** `noisereduce.reduce_noise()` (standard Wiener filtering) to establish noise profile from room tone, then 1D U-Net de-reverberation (torch model) to map reverberant signal to dry audio.
- **Group 2 (Polyphonic Plucked/Struck):** Transient-preserving Wiener filter with oversubtraction factor (α) and spectral floor (β) to avoid smearing attack transients. Then `librosa.decompose.hpss()` for Harmonic-Percussive Source Separation — isolates strike from resonant tail.
- **Group 3 (Unpitched Percussion):** `noisereduce.reduce_noise()` with spectral gating focused on minimizing noise floor between discrete hits.
- Output is a cleaned WAV passed to transcription and synthesis. The recording consumer must NOT apply any filtering — deliver raw captured signal.

**Anti-patterns:** Never preprocess inside the audio callback. Never apply standard Wiener to Group 2 (destroys transients). Never skip preprocessing — it materially affects downstream model accuracy.
