"""Synthetic audio generation for Group 1 DDSP pipeline smoke tests.

Generates monophonic WAV files that simulate a continuous instrument
(flute/violin) without requiring a real recording. Suitable for end-to-end
pipeline validation.
"""

import os
import wave
from typing import Optional

import numpy as np


def generate_test_wav(
    output_path: str,
    duration_s: float = 35.0,
    sample_rate_hz: int = 44100,
    pitch_sequence: Optional[list[float]] = None,
) -> str:
    """Generate a synthetic monophonic WAV simulating a flute/violin.

    Produces a chromatic scale with vibrato and per-note ADSR envelopes.
    Each note contains the fundamental and a 3rd harmonic at -12 dB.
    Suitable for DDSP preprocessing, fine-tuning, and transcription.

    Args:
        output_path: Destination path for the output WAV file.
        duration_s: Total audio duration in seconds.
        sample_rate_hz: Sample rate of the output WAV.
        pitch_sequence: Optional list of MIDI note numbers to render.
            Defaults to a two-octave chromatic scale (C4–C6, MIDI 60–84).

    Returns:
        Absolute path to the written WAV file.
    """
    if pitch_sequence is None:
        # Two-octave chromatic scale: C4 (60) to C6 (84)
        pitch_sequence = list(range(60, 85))

    n_notes = len(pitch_sequence)
    note_duration_s = duration_s / n_notes

    # ADSR parameters
    attack_s = 0.03
    decay_s = 0.02
    sustain_level = 0.8
    release_s = 0.05

    # Vibrato parameters: ±20 cents at 5 Hz
    vibrato_rate_hz = 5.0
    vibrato_depth = 0.006  # fractional frequency deviation

    all_samples = []

    for midi_note in pitch_sequence:
        freq_hz = 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))
        n_samples = int(note_duration_s * sample_rate_hz)
        t = np.linspace(0, note_duration_s, n_samples, endpoint=False)

        # Vibrato modulation applied to instantaneous phase
        vibrato = vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate_hz * t)
        # Instantaneous phase via cumulative sum (accurate FM)
        inst_freq = freq_hz * (1.0 + vibrato)
        phase = 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate_hz

        # Fundamental + 3rd harmonic at -12 dB (~0.25 amplitude)
        signal = np.sin(phase) + 0.25 * np.sin(3.0 * phase)

        # ADSR envelope
        envelope = np.ones(n_samples)
        attack_n = min(int(attack_s * sample_rate_hz), n_samples)
        decay_n = min(int(decay_s * sample_rate_hz), n_samples - attack_n)
        release_n = min(int(release_s * sample_rate_hz), n_samples)

        # Attack
        envelope[:attack_n] = np.linspace(0.0, 1.0, attack_n)
        # Decay
        if decay_n > 0:
            envelope[attack_n:attack_n + decay_n] = np.linspace(
                1.0, sustain_level, decay_n
            )
        # Sustain (everything else already = 1.0, set to sustain_level)
        envelope[attack_n + decay_n:n_samples - release_n] = sustain_level
        # Release
        if release_n > 0:
            envelope[n_samples - release_n:] = np.linspace(
                sustain_level, 0.0, release_n
            )

        note_audio = (signal * envelope).astype(np.float32)
        # Normalize per-note to 0.7 peak amplitude
        peak = np.max(np.abs(note_audio))
        if peak > 0:
            note_audio = note_audio * (0.7 / peak)

        all_samples.append(note_audio)

    audio = np.concatenate(all_samples).astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write as 16-bit PCM WAV
    pcm_16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm_16.tobytes())

    actual_duration = len(audio) / sample_rate_hz
    print(
        f"Generated {output_path} "
        f"({actual_duration:.1f}s, {n_notes} notes, {sample_rate_hz} Hz)"
    )
    return os.path.abspath(output_path)
