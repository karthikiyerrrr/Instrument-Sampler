"""Shared DDSP utilities for Group 1 synthesis notebooks.

Provides model download, audio preprocessing, fine-tuning, and inference
functions used across the preprocessing, finetune, and inference notebooks.
"""

import os
import shutil
import time
import wave
from typing import Any, Callable

import gin
import librosa
import numpy as np
import tensorflow as tf
from scipy.signal import resample_poly

# ---- Constants ----

DDSP_SAMPLE_RATE: int = 16_000
FRAME_RATE: int = 250
OUTPUT_SAMPLE_RATE: int = 44_100

_GCS_BASE = "gs://ddsp/models/timbre_transfer_colab/2021-07-08"

AVAILABLE_MODELS: dict[str, str] = {
    "violin": "solo_violin_ckpt",
    "flute": "solo_flute_ckpt",
    "flute2": "solo_flute2_ckpt",
    "trumpet": "solo_trumpet_ckpt",
    "tenor_saxophone": "solo_tenor_saxophone_ckpt",
}


# ---- Model utilities ----

def _checkpoint_exists(ckpt_dir: str) -> bool:
    """Check if a directory contains a TF checkpoint."""
    if not os.path.isdir(ckpt_dir):
        return False
    return any(f.startswith("ckpt") for f in os.listdir(ckpt_dir))


def list_base_models(models_dir: str) -> list[dict[str, Any]]:
    """Return available pretrained models and their download status.

    Args:
        models_dir: Local directory where models are stored.

    Returns:
        List of dicts with ``name``, ``downloaded``, and
        ``checkpoint_dir`` keys.
    """
    result: list[dict[str, Any]] = []
    for name in AVAILABLE_MODELS:
        ckpt_dir = os.path.join(models_dir, name)
        downloaded = _checkpoint_exists(ckpt_dir)
        result.append({
            "name": name,
            "downloaded": downloaded,
            "checkpoint_dir": ckpt_dir if downloaded else None,
        })
    return result


def download_model(model_name: str, output_dir: str) -> str:
    """Download a pretrained DDSP checkpoint from GCS.

    Args:
        model_name: Key in :data:`AVAILABLE_MODELS`.
        output_dir: Local directory to store the checkpoint files.

    Returns:
        Path to the downloaded checkpoint directory.

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(AVAILABLE_MODELS.keys())}"
        )
    os.makedirs(output_dir, exist_ok=True)
    gcs_dir = f"{_GCS_BASE}/{AVAILABLE_MODELS[model_name]}"
    print(f"Downloading model '{model_name}' from {gcs_dir}")
    src_files = tf.io.gfile.listdir(gcs_dir)
    total = len(src_files)
    for i, fname in enumerate(src_files):
        src = f"{gcs_dir}/{fname}"
        dst = os.path.join(output_dir, fname)
        tf.io.gfile.copy(src, dst, overwrite=True)
        print(f"  [{i + 1}/{total}] {fname}")
    print(f"Model '{model_name}' downloaded to {output_dir}")
    return output_dir


def find_gin_config(checkpoint_dir: str) -> str:
    """Locate the operative gin config in a checkpoint directory.

    Args:
        checkpoint_dir: Path to a DDSP checkpoint directory.

    Returns:
        Path to the gin config file.

    Raises:
        FileNotFoundError: If no gin config is found.
    """
    for name in ("operative_config-0.gin", "operative_config.gin"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No operative gin config found in {checkpoint_dir}"
    )


# ---- Preprocessing ----

def preprocess_audio(
    wav_path: str,
    sample_rate_hz: int = DDSP_SAMPLE_RATE,
    frame_rate: int = FRAME_RATE,
) -> dict[str, np.ndarray]:
    """Load and preprocess a WAV file for DDSP inference or training.

    Resamples to 16 kHz, extracts f0 (CREPE via torchcrepe) and
    loudness (A-weighted dB).

    Args:
        wav_path: Path to the input WAV file.
        sample_rate_hz: Target sample rate (default 16000 for DDSP).
        frame_rate: Analysis frame rate in Hz (default 250).

    Returns:
        Dict with keys ``audio``, ``f0_hz``, ``f0_confidence``,
        ``loudness_db`` — all as float32 numpy arrays.
    """
    from ddsp import spectral_ops

    print(f"Loading {wav_path} (target {sample_rate_hz} Hz)")
    audio, _ = librosa.load(wav_path, sr=sample_rate_hz, mono=True)
    audio = audio.astype(np.float32)

    print(f"Extracting f0 with CREPE ({frame_rate} Hz frame rate)")
    f0_hz, f0_confidence = spectral_ops.compute_f0(
        audio, sample_rate_hz, frame_rate
    )

    print("Computing loudness")
    loudness_db = spectral_ops.compute_loudness(
        audio, sample_rate_hz, frame_rate
    )

    # Align lengths
    n_frames = min(len(f0_hz), len(loudness_db))
    f0_hz = f0_hz[:n_frames]
    f0_confidence = f0_confidence[:n_frames]
    loudness_db = loudness_db[:n_frames]
    n_samples = n_frames * (sample_rate_hz // frame_rate)
    audio = audio[:n_samples]

    print(
        f"Preprocessed: {n_samples} samples, {n_frames} frames, "
        f"{n_samples / sample_rate_hz:.1f}s"
    )
    return {
        "audio": audio,
        "f0_hz": f0_hz.astype(np.float32),
        "f0_confidence": f0_confidence.astype(np.float32),
        "loudness_db": loudness_db.astype(np.float32),
    }


def chunk_for_training(
    features: dict[str, np.ndarray],
    example_secs: float = 4.0,
    hop_secs: float = 1.0,
    sample_rate_hz: int = DDSP_SAMPLE_RATE,
    frame_rate: int = FRAME_RATE,
) -> list[dict[str, np.ndarray]]:
    """Chunk preprocessed features into overlapping training examples.

    Args:
        features: Output of :func:`preprocess_audio`.
        example_secs: Duration of each example in seconds.
        hop_secs: Hop between examples in seconds.
        sample_rate_hz: Audio sample rate.
        frame_rate: Feature frame rate.

    Returns:
        List of feature dicts, each with ``audio``, ``f0_hz``,
        ``loudness_db`` arrays.
    """
    example_samples = int(example_secs * sample_rate_hz)
    example_frames = int(example_secs * frame_rate)
    hop_samples = int(hop_secs * sample_rate_hz)
    hop_frames = int(hop_secs * frame_rate)

    audio = features["audio"]
    n_samples = len(audio)
    examples: list[dict[str, np.ndarray]] = []

    offset_samples = 0
    offset_frames = 0
    while offset_samples + example_samples <= n_samples:
        examples.append({
            "audio": audio[offset_samples:offset_samples + example_samples],
            "f0_hz": features["f0_hz"][
                offset_frames:offset_frames + example_frames
            ],
            "loudness_db": features["loudness_db"][
                offset_frames:offset_frames + example_frames
            ],
        })
        offset_samples += hop_samples
        offset_frames += hop_frames

    print(f"Chunked into {len(examples)} examples of {example_secs:.1f}s")
    return examples


# ---- Model building ----

def build_model_from_gin(gin_config_path: str):
    """Build a DDSP Autoencoder from a gin operative config.

    Args:
        gin_config_path: Path to the ``operative_config-0.gin`` file.

    Returns:
        A DDSP Autoencoder model instance (not yet built).
    """
    from ddsp.training import models as ddsp_models

    gin.clear_config()
    gin.parse_config_file(gin_config_path)
    return ddsp_models.Autoencoder()


# ---- Fine-tuning ----

def freeze_layers(model) -> list[str]:
    """Freeze GRU and input FC stack variables for transfer learning.

    Args:
        model: A DDSP model instance with built variables.

    Returns:
        List of variable names that were frozen.
    """
    frozen: list[str] = []
    for var in model.trainable_variables:
        name_lower = var.name.lower()
        if "gru" in name_lower or "rnn" in name_lower:
            var._trainable = False  # noqa: SLF001
            frozen.append(var.name)
        elif "input" in name_lower and "dense" in name_lower:
            var._trainable = False  # noqa: SLF001
            frozen.append(var.name)
    return frozen


def examples_to_dataset(
    examples: list[dict[str, np.ndarray]],
    batch_size: int,
) -> tf.data.Dataset:
    """Convert a list of example dicts to a shuffled, batched TF dataset.

    Args:
        examples: List of dicts with ``audio``, ``f0_hz``,
            ``loudness_db`` arrays.
        batch_size: Batch size.

    Returns:
        A ``tf.data.Dataset`` yielding batched feature dicts.
    """
    audio = np.stack([ex["audio"] for ex in examples])
    f0_hz = np.stack([ex["f0_hz"] for ex in examples])
    loudness_db = np.stack([ex["loudness_db"] for ex in examples])

    # Add feature dimensions expected by DDSP: [batch, frames, 1]
    f0_hz = f0_hz[..., np.newaxis]
    loudness_db = loudness_db[..., np.newaxis]

    ds = tf.data.Dataset.from_tensor_slices({
        "audio": audio.astype(np.float32),
        "f0_hz": f0_hz.astype(np.float32),
        "loudness_db": loudness_db.astype(np.float32),
    })
    return ds.shuffle(len(examples)).repeat().batch(batch_size)


def finetune(
    base_checkpoint_dir: str,
    examples: list[dict[str, np.ndarray]],
    output_dir: str,
    steps: int = 2000,
    learning_rate: float = 0.0001,
    batch_size: int = 4,
    grad_clip_norm: float = 3.0,
) -> str:
    """Fine-tune a DDSP model on calibration examples.

    Args:
        base_checkpoint_dir: Directory with the pretrained checkpoint
            and ``operative_config-0.gin``.
        examples: List of training example dicts from
            :func:`chunk_for_training`.
        output_dir: Directory to save the fine-tuned checkpoint.
        steps: Number of training steps.
        learning_rate: Adam learning rate.
        batch_size: Training batch size.
        grad_clip_norm: Maximum gradient norm.

    Returns:
        Path to the fine-tuned checkpoint directory.

    Raises:
        FileNotFoundError: If the base checkpoint is missing.
    """
    os.makedirs(output_dir, exist_ok=True)
    gin_config = find_gin_config(base_checkpoint_dir)

    print(f"Building model from {gin_config}")
    model = build_model_from_gin(gin_config)

    dataset = examples_to_dataset(examples, batch_size)

    # Forward pass to build variables
    for batch in dataset.take(1):
        _ = model(batch, training=False)

    # Restore pretrained weights
    checkpoint = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(base_checkpoint_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in {base_checkpoint_dir}"
        )
    checkpoint.restore(latest_ckpt).expect_partial()
    print(f"Restored weights from {latest_ckpt}")

    # Freeze encoder layers
    frozen = freeze_layers(model)
    trainable_vars = [v for v in model.trainable_variables]
    total_params = sum(v.numpy().size for v in model.variables)
    trainable_params = sum(v.numpy().size for v in trainable_vars)
    print(
        f"Frozen {len(frozen)} vars. "
        f"Training {trainable_params}/{total_params} params "
        f"({trainable_params / total_params * 100:.0f}%)"
    )

    # Training loop
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    best_loss = float("inf")
    start_time = time.monotonic()

    step = 0
    while step < steps:
        for batch in dataset:
            if step >= steps:
                break
            step += 1

            with tf.GradientTape() as tape:
                outputs = model(batch, training=True)
                loss = model.loss_obj(batch, outputs)

            grads = tape.gradient(loss, trainable_vars)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            train_loss.update_state(loss)

            if step % 50 == 0 or step == steps:
                avg_loss = float(train_loss.result())
                elapsed_s = time.monotonic() - start_time
                print(
                    f"Step {step}/{steps}  "
                    f"loss={avg_loss:.4f}  "
                    f"elapsed={elapsed_s:.1f}s"
                )
                if avg_loss < best_loss:
                    best_loss = avg_loss
                train_loss.reset_states()

            if step % 500 == 0 or step == steps:
                ckpt_prefix = os.path.join(output_dir, "ckpt")
                checkpoint.save(file_prefix=ckpt_prefix)

    # Copy gin config for inference
    shutil.copy2(gin_config, os.path.join(output_dir, "operative_config-0.gin"))
    print(f"\nFine-tuning complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {output_dir}")
    return output_dir


# ---- Inference ----

def run_inference(
    checkpoint_dir: str,
    input_wav: str,
    output_wav: str,
    output_sample_rate_hz: int = OUTPUT_SAMPLE_RATE,
) -> str:
    """Run DDSP timbre transfer on a session WAV file.

    Args:
        checkpoint_dir: Path to the fine-tuned checkpoint directory.
        input_wav: Path to the session WAV to process.
        output_wav: Path for the output WAV file.
        output_sample_rate_hz: Output sample rate (default 44100).

    Returns:
        Path to the written output WAV file.

    Raises:
        FileNotFoundError: If checkpoint or input WAV is missing.
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    if not os.path.isfile(input_wav):
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")

    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

    # Preprocess input
    print(f"Preprocessing {input_wav}")
    features = preprocess_audio(input_wav)
    audio = features["audio"]
    f0_hz = features["f0_hz"]
    loudness_db = features["loudness_db"]

    # Add batch dimension
    batch = {
        "audio": tf.constant(audio[np.newaxis, :]),
        "f0_hz": tf.constant(f0_hz[np.newaxis, :, np.newaxis]),
        "loudness_db": tf.constant(loudness_db[np.newaxis, :, np.newaxis]),
    }

    # Load model
    print(f"Loading model from {checkpoint_dir}")
    gin_config = find_gin_config(checkpoint_dir)
    model = build_model_from_gin(gin_config)

    # Build variables with forward pass, then restore weights
    _ = model(batch, training=False)
    checkpoint = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}")
    checkpoint.restore(latest_ckpt).expect_partial()

    # Run inference
    print("Running synthesis")
    outputs = model(batch, training=False)
    audio_out = outputs["audio_synth"].numpy().squeeze()

    # Resample 16 kHz -> 44.1 kHz
    print(f"Resampling {DDSP_SAMPLE_RATE} -> {output_sample_rate_hz} Hz")
    audio_44k = resample_poly(
        audio_out, up=441, down=160
    ).astype(np.float32)

    # Write 16-bit PCM WAV
    pcm_16 = (np.clip(audio_44k, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(output_sample_rate_hz)
        wf.writeframes(pcm_16.tobytes())

    duration_s = len(audio) / DDSP_SAMPLE_RATE
    print(f"Output written: {output_wav} ({duration_s:.1f}s)")
    return output_wav
