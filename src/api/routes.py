"""REST API routes for device listing, session management, and post-processing."""

import logging
import threading
from pathlib import Path
from typing import Any

import sounddevice as sd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.api.session import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class StartRequest(BaseModel):
    """Body for the session start endpoint."""

    device_index: int | None = None


class StartResponse(BaseModel):
    """Response after successfully starting a session."""

    status: str = "recording"
    wav_path: str


class StopResponse(BaseModel):
    """Response after stopping the active session."""

    status: str = "stopped"
    wav_path: str
    duration_s: float


class StatusResponse(BaseModel):
    """Current session state."""

    status: str
    wav_path: str | None = None


_session_manager: SessionManager | None = None


def init_routes(session_manager: SessionManager) -> None:
    """Bind the shared SessionManager instance used by all route handlers.

    Args:
        session_manager: The singleton manager created in ``server.py``.
    """
    global _session_manager
    _session_manager = session_manager


def _sm() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialised")
    return _session_manager


@router.get("/devices")
def list_devices() -> list[dict[str, Any]]:
    """Return available audio input devices.

    Returns:
        List of device dicts with index, name, channels, and sample rate.
    """
    devices = sd.query_devices()
    result: list[dict[str, Any]] = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            result.append({
                "index": idx,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
            })
    return result


@router.post("/session/start", response_model=StartResponse)
def start_session(body: StartRequest) -> StartResponse:
    """Start a new recording / analysis session.

    Args:
        body: Optional device index override.

    Returns:
        Recording status and WAV file path.
    """
    sm = _sm()
    try:
        wav_path = sm.start(device_index=body.device_index)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StartResponse(wav_path=wav_path)


@router.post("/session/stop", response_model=StopResponse)
def stop_session() -> StopResponse:
    """Stop the active session.

    Returns:
        Final status with WAV path and duration.
    """
    sm = _sm()
    try:
        info = sm.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StopResponse(wav_path=info["wav_path"], duration_s=info["duration_s"])


@router.get("/session/status", response_model=StatusResponse)
def session_status() -> StatusResponse:
    """Return whether a session is currently active.

    Returns:
        Status string and optional WAV path.
    """
    sm = _sm()
    if sm.active:
        return StatusResponse(status="recording", wav_path=sm.wav_path)
    return StatusResponse(status="idle")


# ---------------------------------------------------------------------------
# Post-processing — transcription
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    """Body for the transcription endpoint."""

    wav_path: str


class TranscribeResponse(BaseModel):
    """Response after a successful transcription."""

    status: str = "done"
    midi_path: str


@router.post("/post-process/transcribe", response_model=TranscribeResponse)
def transcribe(body: TranscribeRequest) -> TranscribeResponse:
    """Run basic-pitch polyphonic transcription on a recorded WAV file.

    Args:
        body: Path to the WAV file to transcribe.

    Returns:
        Path to the generated ``.mid`` file.
    """
    from src.transcription import transcribe_wav  # deferred — not installed by default

    wav = Path(body.wav_path)
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"WAV not found: {body.wav_path}")
    try:
        midi_path = transcribe_wav(str(wav))
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TranscribeResponse(midi_path=midi_path)


# ---------------------------------------------------------------------------
# Post-processing — DDSP calibration (fine-tuning)
# ---------------------------------------------------------------------------

# Job state shared between the background thread and the status endpoint.
_calibration_jobs: dict[str, dict[str, Any]] = {}
_calibration_lock = threading.Lock()


class CalibrateRequest(BaseModel):
    """Body for the calibration endpoint."""

    wav_path: str
    model_name: str
    steps: int = 300


class CalibrateResponse(BaseModel):
    """Immediate response when a calibration job is queued."""

    status: str  # "queued" | "done" | "error"
    job_id: str
    model_dir: str | None = None
    error: str | None = None


def _run_calibration(job_id: str, wav_path: str, model_name: str, steps: int) -> None:
    """Background thread target for DDSP fine-tuning."""
    from src.timbre import calibrate  # deferred — requires .venv-ddsp

    try:
        model_dir = calibrate(wav_path, model_name, steps=steps)
        with _calibration_lock:
            _calibration_jobs[job_id] = {"status": "done", "model_dir": model_dir}
    except Exception as exc:  # noqa: BLE001
        logger.error("Calibration job %s failed: %s", job_id, exc)
        with _calibration_lock:
            _calibration_jobs[job_id] = {"status": "error", "error": str(exc)}


@router.post("/post-process/calibrate", response_model=CalibrateResponse)
def start_calibration(body: CalibrateRequest) -> CalibrateResponse:
    """Begin DDSP fine-tuning on a calibration recording (async, background thread).

    Training can take several minutes.  Poll ``GET /api/post-process/calibrate/{job_id}``
    for completion status.

    Args:
        body: WAV path, model name, and optional step count.

    Returns:
        Job ID and initial status.
    """
    wav = Path(body.wav_path)
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"WAV not found: {body.wav_path}")

    import uuid
    job_id = str(uuid.uuid4())[:8]
    model_dir = str(Path("models") / body.model_name)

    with _calibration_lock:
        _calibration_jobs[job_id] = {"status": "queued"}

    t = threading.Thread(
        target=_run_calibration,
        args=(job_id, str(wav), body.model_name, body.steps),
        daemon=True,
        name=f"ddsp-calibrate-{job_id}",
    )
    t.start()
    logger.info("Calibration job %s started for '%s'", job_id, body.model_name)
    return CalibrateResponse(status="queued", job_id=job_id, model_dir=model_dir)


@router.get("/post-process/calibrate/{job_id}", response_model=CalibrateResponse)
def calibration_status(job_id: str) -> CalibrateResponse:
    """Poll the status of a running calibration job.

    Args:
        job_id: ID returned by ``POST /api/post-process/calibrate``.

    Returns:
        Current job status and model directory if complete.
    """
    with _calibration_lock:
        job = _calibration_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No calibration job '{job_id}'")
    return CalibrateResponse(job_id=job_id, **job)


# ---------------------------------------------------------------------------
# Post-processing — DDSP timbre cloning (inference)
# ---------------------------------------------------------------------------

class CloneRequest(BaseModel):
    """Body for the timbre-cloning endpoint."""

    wav_path: str
    model_dir: str


class CloneResponse(BaseModel):
    """Response after a successful timbre-cloning run."""

    status: str = "done"
    cloned_wav_path: str


@router.post("/post-process/clone", response_model=CloneResponse)
def clone_timbre(body: CloneRequest) -> CloneResponse:
    """Synthesise a timbre-cloned WAV from a session recording.

    Args:
        body: Input WAV path and the model directory to use.

    Returns:
        Path to the cloned ``.wav`` at 44.1 kHz / 16-bit PCM.
    """
    from src.timbre import clone_timbre as _clone  # deferred — requires .venv-ddsp

    wav = Path(body.wav_path)
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"WAV not found: {body.wav_path}")
    model = Path(body.model_dir)
    if not (model / "decoder.onnx").exists():
        raise HTTPException(
            status_code=404,
            detail=f"decoder.onnx not found in {body.model_dir} — run calibration first.",
        )
    try:
        cloned_path = _clone(str(wav), str(model))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return CloneResponse(cloned_wav_path=cloned_path)


# ---------------------------------------------------------------------------
# Models — list available DDSP model directories
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Metadata for one trained DDSP model."""

    name: str
    model_dir: str
    has_onnx: bool


@router.get("/post-process/models", response_model=list[ModelInfo])
def list_models(models_dir: str = "models") -> list[ModelInfo]:
    """Return all trained DDSP models found in *models_dir*.

    Returns:
        List of model descriptors ordered alphabetically by name.
    """
    base = Path(models_dir)
    if not base.is_dir():
        return []
    results: list[ModelInfo] = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir():
            results.append(ModelInfo(
                name=sub.name,
                model_dir=str(sub),
                has_onnx=(sub / "decoder.onnx").exists(),
            ))
    return results
