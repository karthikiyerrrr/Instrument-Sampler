"""DDSP timbre-cloning inference — runs inside the .venv-ddsp subprocess.

Loads the ONNX decoder exported by train.py, synthesises a new audio file
whose timbre matches the trained instrument.  TensorFlow is NOT imported;
all synthesis runs via onnxruntime + numpy.

Usage:
    python ddsp_worker/infer.py \
        --wav  recordings/session.wav \
        --model models/my_instrument \
        --out   recordings/session_cloned.wav

The ONNX decoder outputs synthesis parameters (amps, harm_dist, noise_mags);
the harmonic + noise synthesis is performed here in pure numpy.
"""

import argparse
import json
import logging
import sys
import wave
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import onnxruntime as ort  # type: ignore[import]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ddsp_worker/infer] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 16_000
OUTPUT_SR: int = 44_100
FRAME_RATE: int = 250
HOP_SAMPLES: int = SAMPLE_RATE // FRAME_RATE
N_HARMONICS: int = 60
N_NOISE_BANDS: int = 65


# ---------------------------------------------------------------------------
# Feature extraction (mirrors normalisation in train.py)
# ---------------------------------------------------------------------------

def _load_audio(wav_path: str) -> np.ndarray:
    """Load WAV at SAMPLE_RATE Hz mono float32."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    logger.info("Loaded %s — %.1f s @ %d Hz", wav_path, len(audio) / SAMPLE_RATE, SAMPLE_RATE)
    return audio.astype(np.float32)


def _compute_features(audio: np.ndarray, frame_rate: int) -> np.ndarray:
    """Compute normalised features [n_frames, 2] matching train.py.

    Args:
        audio: Mono float32 audio at SAMPLE_RATE.
        frame_rate: Feature frame rate (must match training).

    Returns:
        features: float32 [n_frames, 2] — (f0_norm, loudness_norm).
    """
    hop = SAMPLE_RATE // frame_rate
    f0, _voiced, _ = librosa.pyin(
        audio,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=SAMPLE_RATE,
        hop_length=hop,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0].astype(np.float32)
    loudness_db = librosa.amplitude_to_db(rms, ref=1.0).astype(np.float32)
    n = min(len(f0), len(loudness_db))
    f0_norm = np.clip(f0[:n] / 2000.0, 0.0, 1.0)
    loud_norm = np.clip((loudness_db[:n] + 120.0) / 120.0, 0.0, 1.0)
    return np.stack([f0_norm, loud_norm], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Numpy synthesis (mirrors _synthesize() in train.py but runs without TF)
# ---------------------------------------------------------------------------

def _upsample_1d(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linear interpolation from n_frames → n_out samples."""
    n_in = len(x)
    t_in = np.linspace(0.0, 1.0, n_in)
    t_out = np.linspace(0.0, 1.0, n_out)
    return np.interp(t_out, t_in, x).astype(np.float32)


def _harmonic_synth(
    f0_norm: np.ndarray,
    amps: np.ndarray,
    harm_dist: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Additive harmonic synthesis in pure numpy.

    Args:
        f0_norm: [T] normalised F0 (multiply by 2000 for Hz).
        amps:    [T, 1] amplitude envelope.
        harm_dist: [T, N_HARMONICS] harmonic distribution.
        n_samples: Output length in samples.

    Returns:
        audio: [n_samples] float32.
    """
    f0_hz = _upsample_1d(f0_norm * 2000.0, n_samples)       # Hz
    amp_up = _upsample_1d(amps[:, 0], n_samples)

    harm_nums = np.arange(1, N_HARMONICS + 1, dtype=np.float32)
    harm_freqs = f0_hz[:, np.newaxis] * harm_nums             # [n_samples, N_HARM]
    harm_freqs[harm_freqs > SAMPLE_RATE / 2] = 0.0

    phases = 2.0 * np.pi * np.cumsum(harm_freqs / SAMPLE_RATE, axis=0)

    harm_dist_up = np.zeros((n_samples, N_HARMONICS), dtype=np.float32)
    for i in range(N_HARMONICS):
        harm_dist_up[:, i] = _upsample_1d(harm_dist[:, i], n_samples)

    audio = np.sum(np.sin(phases) * harm_dist_up, axis=-1) * amp_up
    return audio.astype(np.float32)


def _noise_synth(noise_mags: np.ndarray, n_samples: int) -> np.ndarray:
    """Spectral-envelope noise synthesis in pure numpy.

    Args:
        noise_mags: [T, N_NOISE_BANDS] noise filter magnitudes.
        n_samples: Output length in samples.

    Returns:
        audio: [n_samples] float32.
    """
    noise = np.random.randn(n_samples).astype(np.float32)
    mean_mags = np.mean(noise_mags, axis=0)                   # [N_NOISE_BANDS]
    noise_fft = np.fft.rfft(noise)
    n_freqs = len(noise_fft)
    mag_up = np.interp(
        np.linspace(0.0, 1.0, n_freqs),
        np.linspace(0.0, 1.0, N_NOISE_BANDS),
        mean_mags,
    ).astype(np.float32)
    noise_fft *= mag_up
    return np.fft.irfft(noise_fft, n=n_samples).astype(np.float32)


def _synthesize(features: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """Run ONNX decoder then synthesise audio in numpy.

    Args:
        features: [n_frames, 2] normalised features.
        session: Loaded ONNX decoder session.

    Returns:
        Mono float32 audio at SAMPLE_RATE.
    """
    feat_batch = features[np.newaxis]                         # [1, T, 2]
    outputs = session.run(None, {"features": feat_batch})
    amps, harm_dist, noise_mags = outputs[0][0], outputs[1][0], outputs[2][0]
    # amps: [T, 1], harm_dist: [T, N_HARM], noise_mags: [T, N_NOISE]

    n_samples = len(features) * HOP_SAMPLES
    f0_norm = features[:, 0]

    harmonic = _harmonic_synth(f0_norm, amps, harm_dist, n_samples)
    noise = _noise_synth(noise_mags, n_samples)

    audio = harmonic + 0.1 * noise
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio / peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _upsample_to_44k(audio_16k: np.ndarray) -> np.ndarray:
    """Upsample 16 kHz → 44.1 kHz using polyphase filter (441/160 ratio)."""
    return scipy.signal.resample_poly(audio_16k, up=441, down=160).astype(np.float32)


def _write_wav(path: str, audio: np.ndarray) -> None:
    """Write float32 audio as 16-bit PCM WAV at OUTPUT_SR."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(OUTPUT_SR)
        wf.writeframes(pcm.tobytes())
    logger.info("Wrote cloned WAV → %s (%.1f s)", path, len(audio) / OUTPUT_SR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DDSP timbre-cloning inference.")
    parser.add_argument("--wav", required=True, help="Input session WAV.")
    parser.add_argument("--model", required=True, help="Model directory (contains decoder.onnx).")
    parser.add_argument("--out", required=True, help="Output cloned WAV path.")
    args = parser.parse_args()

    model_dir = Path(args.model)
    onnx_path = model_dir / "decoder.onnx"
    if not onnx_path.exists():
        logger.error("decoder.onnx not found in %s — run train.py first.", model_dir)
        sys.exit(1)

    frame_rate = FRAME_RATE
    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        frame_rate = meta.get("frame_rate", FRAME_RATE)

    audio = _load_audio(args.wav)
    features = _compute_features(audio, frame_rate)

    logger.info("Loading ONNX decoder from %s", onnx_path)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=opts)

    audio_16k = _synthesize(features, session)
    audio_44k = _upsample_to_44k(audio_16k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(str(out_path), audio_44k)


if __name__ == "__main__":
    main()
