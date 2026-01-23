"""Web dashboard module for PolymarketBot.

This module provides:
- FastAPI REST API for bot management
- WebSocket for real-time updates
- Web dashboard for monitoring
- Authentication and rate limiting
"""

from .app import create_app, get_app
from .auth import (
    AuthConfig,
    AuthManager,
    configure_auth,
    get_auth_manager,
    require_api_key,
    require_auth,
    rate_limit_only,
    generate_api_key,
)

__all__ = [
    "create_app",
    "get_app",
    "AuthConfig",
    "AuthManager",
    "configure_auth",
    "get_auth_manager",
    "require_api_key",
    "require_auth",
    "rate_limit_only",
    "generate_api_key",
]
