"""WebSocket endpoint for streaming live MIDI events to the frontend."""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.session import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

_session_manager: SessionManager | None = None


def init_ws(session_manager: SessionManager) -> None:
    """Bind the shared SessionManager used by the WebSocket handler.

    Args:
        session_manager: The singleton manager created in ``server.py``.
    """
    global _session_manager
    _session_manager = session_manager


def _sm() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialised")
    return _session_manager


@router.websocket("/api/ws/midi")
async def midi_ws(websocket: WebSocket) -> None:
    """Stream MIDI events as JSON over a WebSocket connection.

    Each message is a JSON object with ``type``, ``note``, ``velocity``,
    and ``time_ms`` fields.  The connection stays open until the session
    ends or the client disconnects.
    """
    await websocket.accept()
    sm = _sm()
    logger.info("WebSocket client connected")
    try:
        async for event in sm.midi_events():
            await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket handler exiting")
