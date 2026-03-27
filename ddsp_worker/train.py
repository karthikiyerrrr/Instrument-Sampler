"""DDSP fine-tuning script — runs inside the Python 3.10 .venv-ddsp subprocess.

Usage:
    python ddsp_worker/train.py \
        --wav  recordings/calibration.wav \
        --out  models/my_instrument \
        [--steps 300] \
        [--frame-rate 250]

Outputs:
    <out>/decoder.onnx   — ONNX-exported decoder for onnxruntime inference.
    <out>/meta.json      — Training metadata (samplerate, frame_rate, steps).

Requirements:
    ddsp==1.9.0, tensorflow==2.12.0, tf2onnx, librosa (see requirements-ddsp.txt)
    Minimum ~3 minutes of clean, monophonic playing for reliable results.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ddsp_worker/train] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_SAMPLE_RATE_HZ: int = 16_000   # DDSP native rate
_FRAME_RATE: int = 250           # Features computed at 250 fps (4 ms resolution)
_MIN_TRAINING_SECS: float = 30.0 # Warn below this; 3 min is ideal


def _load_audio(wav_path: str) -> np.ndarray:
    """Load and resample WAV to 16 kHz mono float32."""
    audio, _ = librosa.load(wav_path, sr=_SAMPLE_RATE_HZ, mono=True)
    logger.info("Loaded %s — %.1f s @ %d Hz", wav_path, len(audio) / _SAMPLE_RATE_HZ, _SAMPLE_RATE_HZ)
    if len(audio) / _SAMPLE_RATE_HZ < _MIN_TRAINING_SECS:
        logger.warning(
            "Recording is %.1f s — recommend ≥3 min for reliable timbre learning.",
            len(audio) / _SAMPLE_RATE_HZ,
        )
    return audio.astype(np.float32)


def _compute_features(
    audio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 (MIDI semitones) and loudness (dB) at _FRAME_RATE fps.

    Args:
        audio: Mono float32 audio at _SAMPLE_RATE_HZ.

    Returns:
        Tuple of (f0_hz, loudness_db), each shape [1, n_frames, 1].
    """
    import ddsp.spectral_ops as spectral_ops  # type: ignore[import]

    audio_tf = tf.constant(audio[np.newaxis, :])  # [1, T]

    f0_hz, _ = spectral_ops.compute_f0(
        audio_tf, _SAMPLE_RATE_HZ, _FRAME_RATE, viterbi=True
    )
    loudness_db = spectral_ops.compute_loudness(audio_tf, _SAMPLE_RATE_HZ, _FRAME_RATE)

    # Shape: [1, n_frames, 1]
    f0_hz = tf.cast(f0_hz[..., tf.newaxis], tf.float32)
    loudness_db = tf.cast(loudness_db[..., tf.newaxis], tf.float32)
    return f0_hz.numpy(), loudness_db.numpy()


def _build_model() -> "ddsp.training.models.Autoencoder":  # type: ignore[name-defined]
    """Construct a DDSP Autoencoder with RNN-FC decoder."""
    import ddsp  # type: ignore[import]
    import ddsp.training  # type: ignore[import]
    import gin  # type: ignore[import]

    gin_config = """
import ddsp
import ddsp.training

RnnFcDecoder.rnn_channels = 512
RnnFcDecoder.rnn_type = 'gru'
RnnFcDecoder.ch = 512
RnnFcDecoder.layers_per_stack = 3
RnnFcDecoder.output_splits = (('amps', 1), ('harmonic_distribution', 60), ('noise_magnitudes', 65))

Autoencoder.preprocessor = @DefaultPreprocessor()
DefaultPreprocessor.time_steps = 1000

Autoencoder.encoder = None
Autoencoder.decoder = @RnnFcDecoder()

Autoencoder.processor_group = @ProcessorGroup()
ProcessorGroup.dag = [
    (@HarmonicPlusNoise(), ['amps', 'harmonic_distribution', 'noise_magnitudes', 'f0_hz']),
]

HarmonicPlusNoise.name = 'additive'
HarmonicPlusNoise.n_samples = 16000
HarmonicPlusNoise.sample_rate = 16000
HarmonicPlusNoise.normalize_below_nyquist = True
HarmonicPlusNoise.scale_fn = @exp_sigmoid

Autoencoder.losses = [@SpectralLoss()]
SpectralLoss.loss_type = 'L1'
SpectralLoss.mag_weight = 1.0
SpectralLoss.delta_time_weight = 1.0
"""
    gin.parse_config(gin_config)
    return ddsp.training.models.Autoencoder()


def _train(
    model: "ddsp.training.models.Autoencoder",  # type: ignore[name-defined]
    f0_hz: np.ndarray,
    loudness_db: np.ndarray,
    audio: np.ndarray,
    steps: int,
) -> None:
    """Run the fine-tuning loop.

    Args:
        model: Uninitialised DDSP Autoencoder.
        f0_hz: Shape [1, n_frames, 1].
        loudness_db: Shape [1, n_frames, 1].
        audio: Raw mono audio float32 shape [T].
        steps: Number of gradient steps.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    audio_tf = tf.constant(audio[np.newaxis, :], dtype=tf.float32)

    batch: dict[str, tf.Tensor] = {
        "f0_hz": tf.constant(f0_hz, dtype=tf.float32),
        "loudness_db": tf.constant(loudness_db, dtype=tf.float32),
        "audio": audio_tf,
    }

    logger.info("Training for %d steps …", steps)
    t0 = time.monotonic()
    for step in range(1, steps + 1):
        with tf.GradientTape() as tape:
            outputs = model(batch, training=True)
            losses = model.losses_dict
            total_loss = sum(losses.values())
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 50 == 0 or step == steps:
            elapsed = time.monotonic() - t0
            logger.info("step %4d/%d  loss=%.4f  elapsed=%.1fs", step, steps, float(total_loss), elapsed)


def _export_onnx(model: "ddsp.training.models.Autoencoder", out_dir: Path, frame_rate: int) -> None:  # type: ignore[name-defined]
    """Export the DDSP decoder to ONNX via tf2onnx.

    Creates a TF concrete function over (f0_hz, loudness_db) inputs, then
    converts with tf2onnx.  The exported model takes:
        - f0_hz:      float32 [1, n_frames, 1]
        - loudness_db: float32 [1, n_frames, 1]
    and returns:
        - audio_out:  float32 [1, n_samples]

    Args:
        model: Trained DDSP Autoencoder.
        out_dir: Directory to write ``decoder.onnx``.
        frame_rate: Feature frame rate used during training (stored in meta).
    """
    import tf2onnx  # type: ignore[import]

    logger.info("Exporting decoder to ONNX …")

    @tf.function(input_signature=[
        tf.TensorSpec([1, None, 1], tf.float32, name="f0_hz"),
        tf.TensorSpec([1, None, 1], tf.float32, name="loudness_db"),
    ])
    def decoder_fn(f0_hz: tf.Tensor, loudness_db: tf.Tensor) -> tf.Tensor:
        batch = {"f0_hz": f0_hz, "loudness_db": loudness_db}
        outputs = model.decoder(batch, training=False)
        audio_out = model.processor_group(outputs, training=False)
        return tf.cast(audio_out["audio_synth"], tf.float32)

    onnx_model, _ = tf2onnx.convert.from_function(
        decoder_fn,
        input_signature=decoder_fn.input_signature,
        opset=13,
    )

    onnx_path = out_dir / "decoder.onnx"
    import onnx  # type: ignore[import]
    onnx.save(onnx_model, str(onnx_path))
    logger.info("Saved ONNX decoder → %s", onnx_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDSP fine-tuning on a calibration recording.")
    parser.add_argument("--wav", required=True, help="Path to calibration WAV file.")
    parser.add_argument("--out", required=True, help="Output directory for model artefacts.")
    parser.add_argument("--steps", type=int, default=300, help="Gradient steps (default 300).")
    parser.add_argument("--frame-rate", type=int, default=_FRAME_RATE, help="Feature frame rate in fps.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = _load_audio(args.wav)
    f0_hz, loudness_db = _compute_features(audio)

    model = _build_model()
    _train(model, f0_hz, loudness_db, audio, args.steps)
    _export_onnx(model, out_dir, args.frame_rate)

    meta = {
        "wav": str(Path(args.wav).resolve()),
        "sample_rate_hz": _SAMPLE_RATE_HZ,
        "frame_rate": args.frame_rate,
        "steps": args.steps,
        "duration_s": round(len(audio) / _SAMPLE_RATE_HZ, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Training complete — artefacts in %s", out_dir)


if __name__ == "__main__":
    main()
