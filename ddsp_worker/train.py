"""DDSP-style timbre cloning trainer — self-contained, no ddsp package required.

Implements a simplified DDSP pipeline using TensorFlow directly:
  - Feature extraction: librosa (F0 via PYIN, loudness via RMS)
  - Decoder: GRU(512) + FC → amps, harm_dist, noise_mags
  - Synthesis: differentiable harmonic + filtered-noise synthesizer (for training)
  - Loss: multi-scale log-magnitude spectral loss
  - Export: ONNX decoder (inputs: features [1,T,2] → outputs: amps, harm_dist, noise_mags)

At inference the synthesis runs in numpy inside infer.py, so TF is not needed
after training.

Usage:
    python ddsp_worker/train.py --wav recordings/calibration.wav --out models/my_instrument

Outputs:
    <out>/decoder.onnx   — ONNX decoder for onnxruntime inference.
    <out>/meta.json      — Training metadata.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore[import]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ddsp_worker/train] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 16_000
FRAME_RATE: int = 250
HOP_SAMPLES: int = SAMPLE_RATE // FRAME_RATE   # 64
N_HARMONICS: int = 60
N_NOISE_BANDS: int = 65
CHUNK_SECS: float = 4.0
CHUNK_FRAMES: int = int(CHUNK_SECS * FRAME_RATE)   # 1000
CHUNK_SAMPLES: int = int(CHUNK_SECS * SAMPLE_RATE)  # 64000
MIN_TRAINING_SECS: float = 30.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _load_audio(wav_path: str) -> np.ndarray:
    """Load and resample WAV to SAMPLE_RATE Hz mono float32."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    duration_s = len(audio) / SAMPLE_RATE
    logger.info("Loaded %s — %.1f s @ %d Hz", wav_path, duration_s, SAMPLE_RATE)
    if duration_s < MIN_TRAINING_SECS:
        logger.warning(
            "Recording is %.1f s — recommend ≥3 min for reliable timbre learning.",
            duration_s,
        )
    return audio.astype(np.float32)


def _compute_features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 (Hz) and loudness (dB) at FRAME_RATE fps using librosa.

    Args:
        audio: Mono float32 audio at SAMPLE_RATE.

    Returns:
        Tuple of (f0_hz, loudness_db), each shape [n_frames].
    """
    f0, _voiced, _ = librosa.pyin(
        audio,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=SAMPLE_RATE,
        hop_length=HOP_SAMPLES,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_SAMPLES)[0].astype(np.float32)
    loudness_db = librosa.amplitude_to_db(rms, ref=1.0).astype(np.float32)
    n = min(len(f0), len(loudness_db))
    return f0[:n], loudness_db[:n]


def _normalize_features(f0_hz: np.ndarray, loudness_db: np.ndarray) -> np.ndarray:
    """Stack F0 + loudness into a normalised [n_frames, 2] feature array."""
    f0_norm = np.clip(f0_hz / 2000.0, 0.0, 1.0)
    loud_norm = np.clip((loudness_db + 120.0) / 120.0, 0.0, 1.0)
    return np.stack([f0_norm, loud_norm], axis=-1).astype(np.float32)


def _make_chunks(
    audio: np.ndarray, features: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split audio + features into overlapping CHUNK_SECS segments."""
    n_frames = len(features)
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    stride = CHUNK_FRAMES // 2
    for start in range(0, n_frames - CHUNK_FRAMES, stride):
        end = start + CHUNK_FRAMES
        a_start = start * HOP_SAMPLES
        a_end = a_start + CHUNK_SAMPLES
        if a_end > len(audio):
            break
        chunks.append((features[start:end], audio[a_start:a_end]))
    return chunks


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_decoder() -> keras.Model:
    """GRU-FC decoder: normalised features → synthesis parameters.

    Input:  features [B, T, 2] (f0_norm, loudness_norm)
    Output: amps [B, T, 1], harm_dist [B, T, N_HARMONICS], noise_mags [B, T, N_NOISE_BANDS]
    """
    inp = keras.Input(shape=(None, 2), name="features")
    x = keras.layers.GRU(512, return_sequences=True, name="gru")(inp)
    x = keras.layers.Dense(512, activation="relu", name="fc1")(x)
    x = keras.layers.Dense(512, activation="relu", name="fc2")(x)
    x = keras.layers.Dense(512, activation="relu", name="fc3")(x)
    amps = keras.layers.Dense(1, activation="sigmoid", name="amps")(x)
    harm_dist = keras.layers.Dense(N_HARMONICS, activation="softmax", name="harm_dist")(x)
    noise_mags = keras.layers.Dense(N_NOISE_BANDS, activation="sigmoid", name="noise_mags")(x)
    return keras.Model(inp, [amps, harm_dist, noise_mags], name="decoder")


# ---------------------------------------------------------------------------
# Differentiable synthesis (used only during training)
# ---------------------------------------------------------------------------

def _upsample_3d(x: tf.Tensor, n_out: int) -> tf.Tensor:
    """Linearly upsample [B, T, C] → [B, n_out, C]."""
    x4d = tf.expand_dims(x, axis=2)                        # [B, T, 1, C]
    up = tf.image.resize(x4d, [n_out, 1], method="bilinear")  # [B, n_out, 1, C]
    return tf.squeeze(up, axis=2)                           # [B, n_out, C]


@tf.function
def _synthesize(
    f0_norm: tf.Tensor,
    amps: tf.Tensor,
    harm_dist: tf.Tensor,
    noise_mags: tf.Tensor,
) -> tf.Tensor:
    """Differentiable harmonic + filtered-noise synthesis.

    Args:
        f0_norm: [B, T, 1] normalised F0 in [0, 1] (divide by 2000 to get Hz).
        amps:    [B, T, 1] global amplitude.
        harm_dist: [B, T, N_HARMONICS] harmonic amplitudes.
        noise_mags: [B, T, N_NOISE_BANDS] noise filter magnitudes.

    Returns:
        audio: [B, CHUNK_SAMPLES] synthesised audio.
    """
    n_samples = CHUNK_SAMPLES

    f0_hz = f0_norm * 2000.0                                  # [B, T, 1]
    f0_up = _upsample_3d(f0_hz, n_samples)[:, :, 0]           # [B, n_samples]
    amp_up = _upsample_3d(amps, n_samples)[:, :, 0]            # [B, n_samples]
    harm_up = _upsample_3d(harm_dist, n_samples)               # [B, n_samples, N_HARM]

    # --- Harmonic synthesis ---
    harm_nums = tf.cast(tf.range(1, N_HARMONICS + 1), tf.float32)  # [N_HARM]
    harm_freqs = f0_up[:, :, tf.newaxis] * harm_nums           # [B, n_samples, N_HARM]
    nyquist = float(SAMPLE_RATE) / 2.0
    harm_freqs = tf.where(harm_freqs > nyquist, 0.0, harm_freqs)
    phases = 2.0 * np.pi * tf.cumsum(harm_freqs / float(SAMPLE_RATE), axis=1)
    harmonic_audio = tf.reduce_sum(tf.sin(phases) * harm_up, axis=-1) * amp_up

    # --- Filtered-noise synthesis ---
    batch_size = tf.shape(f0_norm)[0]
    noise = tf.random.normal([batch_size, n_samples])
    noise_mean = tf.reduce_mean(_upsample_3d(noise_mags, n_samples), axis=1)  # [B, N_NOISE]
    noise_fft = tf.signal.rfft(noise)                          # [B, n_freqs]
    n_freqs = n_samples // 2 + 1
    noise_mean_4d = noise_mean[:, tf.newaxis, tf.newaxis, :]   # [B, 1, 1, N_NOISE]
    mag_up = tf.image.resize(noise_mean_4d, [1, n_freqs], method="bilinear")[:, 0, 0, :]
    noise_shaped = tf.signal.irfft(noise_fft * tf.cast(mag_up, tf.complex64))

    audio = harmonic_audio + 0.1 * noise_shaped
    peak = tf.reduce_max(tf.abs(audio), axis=1, keepdims=True) + 1e-8
    return audio / peak


def _spectral_loss(pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """Multi-scale log-magnitude STFT L1 loss."""
    total = tf.constant(0.0)
    for fft_size in [64, 128, 256, 512, 1024, 2048]:
        hop = fft_size // 4
        pred_mag = tf.abs(tf.signal.stft(pred, fft_size, hop))
        true_mag = tf.abs(tf.signal.stft(target, fft_size, hop))
        total += tf.reduce_mean(tf.abs(tf.math.log(pred_mag + 1e-7) - tf.math.log(true_mag + 1e-7)))
    return total / 6.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(model: keras.Model, chunks: list[tuple[np.ndarray, np.ndarray]], steps: int) -> None:
    """Gradient-descent training on overlapping audio chunks.

    Args:
        model: Uninitialised decoder.
        chunks: List of (features [CHUNK_FRAMES, 2], audio [CHUNK_SAMPLES]) pairs.
        steps: Number of gradient updates.
    """
    if not chunks:
        raise ValueError("No training chunks — recording too short.")
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    logger.info("Training for %d steps on %d chunks …", steps, len(chunks))
    t0 = time.monotonic()

    for step in range(1, steps + 1):
        idx = np.random.randint(len(chunks))
        feat_np, audio_np = chunks[idx]
        feat = tf.constant(feat_np[np.newaxis], dtype=tf.float32)   # [1, T, 2]
        f0_norm = feat[:, :, 0:1]
        audio_true = tf.constant(audio_np[np.newaxis], dtype=tf.float32)

        with tf.GradientTape() as tape:
            amps, harm_dist, noise_mags = model(feat, training=True)
            audio_pred = _synthesize(f0_norm, amps, harm_dist, noise_mags)
            loss = _spectral_loss(audio_pred, audio_true)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 50 == 0 or step == steps:
            logger.info(
                "step %4d/%d  loss=%.4f  elapsed=%.1fs",
                step, steps, float(loss), time.monotonic() - t0,
            )


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _export_onnx(model: keras.Model, out_dir: Path) -> None:
    """Export the decoder to ONNX via tf2onnx.

    The exported model takes:
        features: float32 [1, T, 2]  (normalised f0 + loudness)
    and returns:
        amps:       float32 [1, T, 1]
        harm_dist:  float32 [1, T, N_HARMONICS]
        noise_mags: float32 [1, T, N_NOISE_BANDS]

    Synthesis is performed in numpy inside infer.py so no TF is needed at
    inference time.
    """
    import tf2onnx  # type: ignore[import]
    import onnx    # type: ignore[import]

    logger.info("Exporting decoder to ONNX …")

    @tf.function(input_signature=[
        tf.TensorSpec([1, None, 2], tf.float32, name="features"),
    ])
    def decoder_fn(features: tf.Tensor) -> dict[str, tf.Tensor]:
        amps, harm_dist, noise_mags = model(features, training=False)
        return {"amps": amps, "harm_dist": harm_dist, "noise_mags": noise_mags}

    onnx_model, _ = tf2onnx.convert.from_function(
        decoder_fn,
        input_signature=decoder_fn.input_signature,
        opset=13,
    )
    onnx_path = out_dir / "decoder.onnx"
    onnx.save(onnx_model, str(onnx_path))
    logger.info("Saved ONNX decoder → %s", onnx_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DDSP-style timbre cloning fine-tuner.")
    parser.add_argument("--wav", required=True, help="Path to calibration WAV.")
    parser.add_argument("--out", required=True, help="Output directory for model artefacts.")
    parser.add_argument("--steps", type=int, default=300, help="Gradient steps (default 300).")
    parser.add_argument("--frame-rate", type=int, default=FRAME_RATE, help="Feature frame rate fps.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = _load_audio(args.wav)
    f0_hz, loudness_db = _compute_features(audio)
    features = _normalize_features(f0_hz, loudness_db)
    chunks = _make_chunks(audio, features)

    model = build_decoder()
    _train(model, chunks, args.steps)
    _export_onnx(model, out_dir)

    meta = {
        "wav": str(Path(args.wav).resolve()),
        "sample_rate_hz": SAMPLE_RATE,
        "frame_rate": args.frame_rate,
        "steps": args.steps,
        "n_harmonics": N_HARMONICS,
        "n_noise_bands": N_NOISE_BANDS,
        "duration_s": round(len(audio) / SAMPLE_RATE, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Training complete → %s", out_dir)


if __name__ == "__main__":
    main()
