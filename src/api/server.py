"""FastAPI application entry point.

Run with::

    uvicorn src.api.server:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import init_routes, router as rest_router
from src.api.session import SessionManager
from src.api.websocket import init_ws, router as ws_router
from src.config import AppConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="Instrument-Sampler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = AppConfig()
session_manager = SessionManager(config)
init_routes(session_manager)
init_ws(session_manager)

app.include_router(rest_router)
app.include_router(ws_router)
