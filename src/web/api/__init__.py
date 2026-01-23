"""API module for web dashboard."""

from .routes import router
from .websocket import router as ws_router

__all__ = ["router", "ws_router"]
