"""REST API routes for device listing and session management."""

import logging
from typing import Any

import sounddevice as sd
from fastapi import APIRouter, HTTPException
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
