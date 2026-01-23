"""Authentication system for web dashboard.

Provides:
- API key authentication
- Session-based authentication for dashboard
- Rate limiting
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, Callable

from fastapi import HTTPException, Request, Depends, status
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from ..utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Models
# =============================================================================

class AuthConfig(BaseModel):
    """Authentication configuration."""

    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    session_secret: Optional[str] = None
    session_expiry_hours: int = 24
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    enable_basic_auth: bool = False
    basic_auth_username: Optional[str] = None
    basic_auth_password: Optional[str] = None


class Session(BaseModel):
    """User session."""

    session_id: str
    created_at: datetime
    expires_at: datetime
    client_ip: str
    user_agent: Optional[str] = None


# =============================================================================
# Session Store
# =============================================================================

class SessionStore:
    """In-memory session store."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}

    def create_session(
        self,
        client_ip: str,
        user_agent: Optional[str] = None,
        expiry_hours: int = 24,
    ) -> Session:
        """Create a new session.

        Args:
            client_ip: Client IP address
            user_agent: Client user agent
            expiry_hours: Session expiry time

        Returns:
            New session
        """
        session_id = secrets.token_urlsafe(32)
        now = datetime.utcnow()

        session = Session(
            session_id=session_id,
            created_at=now,
            expires_at=now + timedelta(hours=expiry_hours),
            client_ip=client_ip,
            user_agent=user_agent,
        )

        self.sessions[session_id] = session
        self._cleanup_expired()

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session if valid, None otherwise
        """
        session = self.sessions.get(session_id)

        if session is None:
            return None

        if session.expires_at < datetime.utcnow():
            self.delete_session(session_id)
            return None

        return session

    def delete_session(self, session_id: str) -> None:
        """Delete a session.

        Args:
            session_id: Session ID to delete
        """
        self.sessions.pop(session_id, None)

    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self.sessions.items()
            if session.expires_at < now
        ]
        for sid in expired:
            del self.sessions[sid]


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed.

        Args:
            client_id: Client identifier (IP or API key)

        Returns:
            True if request is allowed
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Get existing requests for client
        client_requests = self.requests.get(client_id, [])

        # Filter to only requests within window
        client_requests = [ts for ts in client_requests if ts > window_start]

        # Check if under limit
        if len(client_requests) >= self.max_requests:
            return False

        # Add new request
        client_requests.append(now)
        self.requests[client_id] = client_requests

        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client.

        Args:
            client_id: Client identifier

        Returns:
            Number of remaining requests
        """
        now = time.time()
        window_start = now - self.window_seconds

        client_requests = self.requests.get(client_id, [])
        client_requests = [ts for ts in client_requests if ts > window_start]

        return max(0, self.max_requests - len(client_requests))

    def get_reset_time(self, client_id: str) -> int:
        """Get seconds until rate limit resets.

        Args:
            client_id: Client identifier

        Returns:
            Seconds until reset
        """
        client_requests = self.requests.get(client_id, [])

        if not client_requests:
            return 0

        oldest = min(client_requests)
        reset_time = oldest + self.window_seconds - time.time()

        return max(0, int(reset_time))


# =============================================================================
# Authentication Manager
# =============================================================================

class AuthManager:
    """Manages authentication for the web dashboard."""

    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.session_store = SessionStore()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit_requests,
            window_seconds=self.config.rate_limit_window_seconds,
        )

        # API key header dependency
        self.api_key_header = APIKeyHeader(
            name=self.config.api_key_header,
            auto_error=False,
        )

        # Basic auth dependency
        self.http_basic = HTTPBasic(auto_error=False)

    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key.

        Args:
            api_key: API key to verify

        Returns:
            True if valid
        """
        if not self.config.api_key:
            return True  # No API key configured, allow all

        return hmac.compare_digest(api_key, self.config.api_key)

    def verify_basic_auth(self, username: str, password: str) -> bool:
        """Verify basic auth credentials.

        Args:
            username: Username
            password: Password

        Returns:
            True if valid
        """
        if not self.config.enable_basic_auth:
            return False

        if not self.config.basic_auth_username or not self.config.basic_auth_password:
            return False

        username_match = hmac.compare_digest(username, self.config.basic_auth_username)
        password_match = hmac.compare_digest(password, self.config.basic_auth_password)

        return username_match and password_match

    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting.

        Args:
            request: FastAPI request

        Returns:
            Client identifier
        """
        # Try X-Forwarded-For header first
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fall back to client host
        return request.client.host if request.client else "unknown"

    def check_rate_limit(self, request: Request) -> None:
        """Check rate limit for request.

        Args:
            request: FastAPI request

        Raises:
            HTTPException: If rate limit exceeded
        """
        client_id = self.get_client_id(request)

        if not self.rate_limiter.is_allowed(client_id):
            reset_time = self.rate_limiter.get_reset_time(client_id)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {reset_time} seconds.",
                headers={
                    "X-RateLimit-Limit": str(self.config.rate_limit_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                },
            )

    def create_session(self, request: Request) -> Session:
        """Create a new session for the client.

        Args:
            request: FastAPI request

        Returns:
            New session
        """
        return self.session_store.create_session(
            client_ip=self.get_client_id(request),
            user_agent=request.headers.get("User-Agent"),
            expiry_hours=self.config.session_expiry_hours,
        )


# Global auth manager
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance.

    Returns:
        AuthManager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def configure_auth(config: AuthConfig) -> AuthManager:
    """Configure authentication with the given settings.

    Args:
        config: Authentication configuration

    Returns:
        Configured AuthManager
    """
    global _auth_manager
    _auth_manager = AuthManager(config)
    return _auth_manager


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def require_api_key(
    request: Request,
    api_key: Optional[str] = Depends(APIKeyHeader(name="X-API-Key", auto_error=False)),
) -> None:
    """Dependency that requires valid API key.

    Args:
        request: FastAPI request
        api_key: API key from header

    Raises:
        HTTPException: If API key is invalid
    """
    auth = get_auth_manager()

    # Check rate limit first
    auth.check_rate_limit(request)

    # If no API key configured, skip check
    if not auth.config.api_key:
        return

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not auth.verify_api_key(api_key):
        logger.warning(
            "invalid_api_key",
            client_ip=auth.get_client_id(request),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def require_auth(
    request: Request,
    api_key: Optional[str] = Depends(APIKeyHeader(name="X-API-Key", auto_error=False)),
    credentials: Optional[HTTPBasicCredentials] = Depends(HTTPBasic(auto_error=False)),
) -> None:
    """Dependency that requires valid authentication (API key or basic auth).

    Args:
        request: FastAPI request
        api_key: API key from header
        credentials: Basic auth credentials

    Raises:
        HTTPException: If authentication fails
    """
    auth = get_auth_manager()

    # Check rate limit first
    auth.check_rate_limit(request)

    # If no auth configured, skip
    if not auth.config.api_key and not auth.config.enable_basic_auth:
        return

    # Try API key first
    if api_key and auth.verify_api_key(api_key):
        return

    # Try basic auth
    if credentials and auth.verify_basic_auth(credentials.username, credentials.password):
        return

    # Check session cookie
    session_id = request.cookies.get("session_id")
    if session_id:
        session = auth.session_store.get_session(session_id)
        if session:
            return

    # No valid authentication
    logger.warning(
        "authentication_failed",
        client_ip=auth.get_client_id(request),
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Basic, ApiKey"},
    )


async def rate_limit_only(request: Request) -> None:
    """Dependency that only checks rate limit (no auth required).

    Args:
        request: FastAPI request

    Raises:
        HTTPException: If rate limit exceeded
    """
    auth = get_auth_manager()
    auth.check_rate_limit(request)


# =============================================================================
# Utility Functions
# =============================================================================

def generate_api_key() -> str:
    """Generate a secure API key.

    Returns:
        Random API key
    """
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage.

    Args:
        api_key: API key to hash

    Returns:
        Hashed API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()
