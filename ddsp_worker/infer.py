"""DDSP timbre-cloning inference — runs inside the Python 3.10 .venv-ddsp subprocess.

Loads the ONNX decoder exported by train.py and synthesises a new audio file
whose timbre matches the trained instrument.  TensorFlow is NOT imported here;
all synthesis runs via onnxruntime.

Usage:
    python ddsp_worker/infer.py \
        --wav  recordings/session_20260327_120000.wav \
        --model models/my_instrument \
        --out   recordings/session_20260327_120000_cloned.wav

Outputs:
    <out>   — 44.1 kHz 16-bit PCM WAV (upsampled from DDSP's 16 kHz output).
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

_DDSP_SR: int = 16_000    # DDSP native sample rate
_OUTPUT_SR: int = 44_100  # DAW / FL Studio target sample rate
_FRAME_RATE: int = 250    # Must match what was used during training


def _load_audio(wav_path: str, sr: int) -> np.ndarray:
    """Load WAV and resample to *sr* Hz, returning mono float32."""
    audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    logger.info("Loaded %s — %.1f s @ %d Hz", wav_path, len(audio) / sr, sr)
    return audio.astype(np.float32)


def _compute_features(
    audio: np.ndarray, frame_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 (Hz) and loudness (dB) without importing TensorFlow.

    Uses ``librosa`` for a lightweight pitch/loudness estimate compatible
    with ONNX decoder inputs.

    Args:
        audio: Mono float32 audio at _DDSP_SR.
        frame_rate: Feature frames per second.

    Returns:
        Tuple of (f0_hz, loudness_db), each shape [1, n_frames, 1].
    """
    hop_length = _DDSP_SR // frame_rate  # e.g. 64 samples @ 250 fps

    # Fundamental frequency via PYIN (probabilistic YIN).
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=_DDSP_SR,
        hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)

    # Loudness: A-weighted RMS in dB.
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    loudness_db = librosa.amplitude_to_db(rms, ref=1.0).astype(np.float32)

    n_frames = min(len(f0), len(loudness_db))
    f0_hz = f0[:n_frames][np.newaxis, :, np.newaxis]         # [1, T, 1]
    loudness_db = loudness_db[:n_frames][np.newaxis, :, np.newaxis]  # [1, T, 1]
    return f0_hz, loudness_db


def _run_decoder(
    session: ort.InferenceSession, f0_hz: np.ndarray, loudness_db: np.ndarray
) -> np.ndarray:
    """Run ONNX decoder and return raw 16 kHz audio.

    Args:
        session: Loaded onnxruntime InferenceSession.
        f0_hz: Shape [1, n_frames, 1].
        loudness_db: Shape [1, n_frames, 1].

    Returns:
        Mono float32 audio, shape [n_samples].
    """
    outputs = session.run(
        None,
        {"f0_hz": f0_hz, "loudness_db": loudness_db},
    )
    audio_out: np.ndarray = outputs[0][0]  # [1, T] → [T]
    return audio_out.astype(np.float32)


def _upsample(audio_16k: np.ndarray) -> np.ndarray:
    """Upsample 16 kHz audio to 44.1 kHz using a polyphase filter.

    Uses the exact ratio 44100/16000 = 441/160.

    Args:
        audio_16k: Mono float32 audio at 16 kHz.

    Returns:
        Mono float32 audio at 44.1 kHz.
    """
    return scipy.signal.resample_poly(audio_16k, up=441, down=160).astype(np.float32)


def _write_wav(path: str, audio: np.ndarray) -> None:
    """Write float32 audio as 16-bit PCM WAV at _OUTPUT_SR.

    Args:
        path: Destination file path.
        audio: Mono float32 audio normalised to [-1, 1].
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_OUTPUT_SR)
        wf.writeframes(pcm.tobytes())
    logger.info("Wrote cloned WAV → %s (%.1f s)", path, len(audio) / _OUTPUT_SR)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDSP timbre-cloning inference.")
    parser.add_argument("--wav", required=True, help="Input session WAV file.")
    parser.add_argument("--model", required=True, help="Model directory (contains decoder.onnx).")
    parser.add_argument("--out", required=True, help="Output cloned WAV path.")
    args = parser.parse_args()

    model_dir = Path(args.model)
    onnx_path = model_dir / "decoder.onnx"
    if not onnx_path.exists():
        logger.error("decoder.onnx not found in %s — run ddsp_worker/train.py first.", model_dir)
        sys.exit(1)

    meta_path = model_dir / "meta.json"
    frame_rate = _FRAME_RATE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        frame_rate = meta.get("frame_rate", _FRAME_RATE)

    audio = _load_audio(args.wav, _DDSP_SR)
    f0_hz, loudness_db = _compute_features(audio, frame_rate)

    logger.info("Loading ONNX decoder from %s", onnx_path)
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 2
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)

    audio_16k = _run_decoder(session, f0_hz, loudness_db)
    audio_44k = _upsample(audio_16k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(str(out_path), audio_44k)


if __name__ == "__main__":
    main()
