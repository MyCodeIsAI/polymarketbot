"""FastAPI dependencies for the web API.

Provides dependency injection for:
- Database connections
- Bot state access
- Authentication (future)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Generator

from ...database import DatabaseConfig, DatabaseManager, configure_database
from ...utils.logging import get_logger

logger = get_logger(__name__)

# Global database instance
_db: Optional[DatabaseManager] = None

# Bot state (in production, would be shared with main bot process)
_bot_state: dict = {
    "is_running": False,
    "state": "stopped",
    "health": "unknown",
    "uptime_seconds": None,
    "start_time": None,
}


def get_db() -> DatabaseManager:
    """Get database connection.

    Returns:
        DatabaseManager instance
    """
    global _db

    if _db is None:
        db_path = Path.cwd() / "polymarketbot.db"
        config = DatabaseConfig(url=f"sqlite:///{db_path}")
        _db = configure_database(config)

    return _db


def get_bot_state() -> dict:
    """Get current bot state.

    Returns:
        Bot state dictionary
    """
    # Calculate uptime if running
    if _bot_state.get("start_time"):
        elapsed = (datetime.utcnow() - _bot_state["start_time"]).total_seconds()
        _bot_state["uptime_seconds"] = elapsed

    return _bot_state.copy()


def update_bot_state(**kwargs) -> None:
    """Update bot state.

    Args:
        **kwargs: State values to update
    """
    global _bot_state
    _bot_state.update(kwargs)

    if kwargs.get("is_running") and not _bot_state.get("start_time"):
        _bot_state["start_time"] = datetime.utcnow()
    elif not kwargs.get("is_running", True):
        _bot_state["start_time"] = None


def close_db() -> None:
    """Close database connection."""
    global _db
    if _db:
        _db.close()
        _db = None
