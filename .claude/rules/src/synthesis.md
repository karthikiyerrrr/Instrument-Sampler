# Synthesis / Timbre Cloning

Synthesis is **offline only** — process the complete preprocessed WAV in one pass. The engine is selected by instrument group. All synthesis pipelines are currently being developed and tested in `notebooks/synthesis/` before integration into the app.

- **Group 1 (Continuous Mono) — DDSP (Harmonic + Noise):** Moved to `notebooks/synthesis/group1_ddsp_*.ipynb` for standalone testing. Uses TensorFlow + DDSP with `torchcrepe` for f0 extraction. DDSP operates at 16 kHz; upsamples to 44.1 kHz. Calibration: >= 30s chromatic scale, fine-tune from pretrained base. See `requirements-ddsp.txt` for dependencies.
- **Group 2 (Polyphonic Plucked/Struck) — Differentiable Karplus-Strong:** PyTorch, main process. Neural network predicts per-note physical string parameters (excitation, damping coefficient, delay-line length, body resonance) from polyphonic MIDI. Renders each note independently then sums. Calibration: >= 30s arpeggios -> fine-tune body IR + damping network.
- **Group 3 (Unpitched Percussion) — WaveNet Autoencoder / GAN (NSynth-like):** PyTorch, main process. Mel-spectrogram encoder -> latent embedding per hit -> autoregressive/GAN decoder conditioned on onset times + amplitude envelopes. Calibration: >= 30s varied hits per drum type (>= 500 hits for training from scratch).

**Anti-patterns:** Never use DDSP harmonic synthesizer for unpitched percussion. Never use Karplus-Strong for sustained wind/voice. Never attempt real-time frame-by-frame synthesis.
