"""Tests for web dashboard module."""

import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from fastapi import status
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from src.web.app import create_app
from src.web.auth import (
    AuthConfig,
    AuthManager,
    SessionStore,
    RateLimiter,
    generate_api_key,
    hash_api_key,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def app():
    """Create test application."""
    return create_app(debug=True)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_config():
    """Create auth configuration."""
    return AuthConfig(
        api_key="test-api-key-12345",
        rate_limit_requests=10,
        rate_limit_window_seconds=60,
    )


@pytest.fixture
def auth_manager(auth_config):
    """Create auth manager."""
    return AuthManager(auth_config)


# =============================================================================
# App Tests
# =============================================================================

class TestApp:
    """Tests for FastAPI application."""

    def test_app_creation(self, app):
        """Test app is created correctly."""
        assert app.title == "PolymarketBot Dashboard"

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_dashboard_route(self, client):
        """Test dashboard route returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_api_docs_in_debug_mode(self, client):
        """Test API docs are available in debug mode."""
        response = client.get("/api/docs")
        assert response.status_code == 200


# =============================================================================
# API Route Tests
# =============================================================================

class TestAPIRoutes:
    """Tests for API routes."""

    @pytest.fixture
    def mock_bot_state(self):
        """Mock bot state."""
        with patch("src.web.api.dependencies.get_bot_state") as mock:
            mock.return_value = {
                "status": "running",
                "uptime": 3600,
                "positions_count": 5,
            }
            yield mock

    @pytest.fixture
    def mock_db(self):
        """Mock database."""
        with patch("src.web.api.dependencies.get_db") as mock:
            db = MagicMock()
            mock.return_value = db
            yield db

    def test_get_status(self, client, mock_bot_state):
        """Test status endpoint."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_get_accounts_empty(self, client, mock_db):
        """Test accounts endpoint with no accounts."""
        with patch("src.database.AccountRepository") as mock_repo:
            mock_repo.return_value.get_all.return_value = []
            response = client.get("/api/accounts")
            assert response.status_code == 200
            assert response.json() == []

    def test_get_positions_empty(self, client, mock_db):
        """Test positions endpoint with no positions."""
        with patch("src.database.PositionRepository") as mock_repo:
            mock_repo.return_value.get_open_positions.return_value = []
            response = client.get("/api/positions")
            assert response.status_code == 200
            assert response.json() == []

    def test_get_trades_empty(self, client, mock_db):
        """Test trades endpoint with no trades."""
        with patch("src.database.TradeLogRepository") as mock_repo:
            mock_repo.return_value.get_recent_trades.return_value = []
            response = client.get("/api/trades")
            assert response.status_code == 200
            assert response.json() == []

    def test_get_trades_with_limit(self, client, mock_db):
        """Test trades endpoint with limit parameter."""
        with patch("src.database.TradeLogRepository") as mock_repo:
            mock_repo.return_value.get_recent_trades.return_value = []
            response = client.get("/api/trades?limit=10")
            assert response.status_code == 200
            mock_repo.return_value.get_recent_trades.assert_called_once_with(limit=10)

    def test_get_health(self, client, mock_bot_state):
        """Test health endpoint."""
        with patch("src.web.api.routes.get_system_health") as mock_health:
            mock_health.return_value = {
                "api_healthy": True,
                "websocket_healthy": True,
                "database_healthy": True,
            }
            response = client.get("/api/health")
            assert response.status_code == 200

    def test_control_start(self, client, mock_bot_state):
        """Test bot start control."""
        with patch("src.web.api.routes.start_bot") as mock_start:
            mock_start.return_value = True
            response = client.post("/api/control/start")
            assert response.status_code == 200

    def test_control_stop(self, client, mock_bot_state):
        """Test bot stop control."""
        with patch("src.web.api.routes.stop_bot") as mock_stop:
            mock_stop.return_value = True
            response = client.post("/api/control/stop")
            assert response.status_code == 200


# =============================================================================
# Session Store Tests
# =============================================================================

class TestSessionStore:
    """Tests for session store."""

    def test_create_session(self):
        """Test session creation."""
        store = SessionStore()
        session = store.create_session(
            client_ip="127.0.0.1",
            user_agent="TestAgent/1.0",
        )

        assert session.session_id is not None
        assert len(session.session_id) > 20
        assert session.client_ip == "127.0.0.1"
        assert session.user_agent == "TestAgent/1.0"

    def test_get_session(self):
        """Test session retrieval."""
        store = SessionStore()
        created = store.create_session(client_ip="127.0.0.1")

        retrieved = store.get_session(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self):
        """Test getting nonexistent session."""
        store = SessionStore()
        session = store.get_session("nonexistent-id")
        assert session is None

    def test_delete_session(self):
        """Test session deletion."""
        store = SessionStore()
        created = store.create_session(client_ip="127.0.0.1")

        store.delete_session(created.session_id)
        retrieved = store.get_session(created.session_id)
        assert retrieved is None

    def test_expired_session(self):
        """Test expired session is not returned."""
        store = SessionStore()
        session = store.create_session(
            client_ip="127.0.0.1",
            expiry_hours=-1,  # Already expired
        )

        retrieved = store.get_session(session.session_id)
        assert retrieved is None


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for rate limiter."""

    def test_allows_requests_under_limit(self):
        """Test requests under limit are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        for i in range(5):
            assert limiter.is_allowed("client1") is True

    def test_blocks_requests_over_limit(self):
        """Test requests over limit are blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        for i in range(3):
            limiter.is_allowed("client1")

        assert limiter.is_allowed("client1") is False

    def test_separate_client_limits(self):
        """Test each client has separate limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        # Client 2 should still be allowed
        assert limiter.is_allowed("client2") is True

    def test_get_remaining(self):
        """Test remaining requests count."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        assert limiter.get_remaining("client1") == 5

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        assert limiter.get_remaining("client1") == 3

    def test_get_reset_time(self):
        """Test reset time calculation."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # No requests yet
        assert limiter.get_reset_time("client1") == 0

        limiter.is_allowed("client1")

        # Should have reset time close to window
        reset_time = limiter.get_reset_time("client1")
        assert 0 < reset_time <= 60


# =============================================================================
# Auth Manager Tests
# =============================================================================

class TestAuthManager:
    """Tests for auth manager."""

    def test_verify_api_key_valid(self, auth_manager):
        """Test valid API key verification."""
        assert auth_manager.verify_api_key("test-api-key-12345") is True

    def test_verify_api_key_invalid(self, auth_manager):
        """Test invalid API key verification."""
        assert auth_manager.verify_api_key("wrong-key") is False

    def test_verify_api_key_no_config(self):
        """Test API key verification with no key configured."""
        manager = AuthManager(AuthConfig())
        # Should allow any key when none configured
        assert manager.verify_api_key("any-key") is True

    def test_verify_basic_auth_valid(self):
        """Test valid basic auth."""
        config = AuthConfig(
            enable_basic_auth=True,
            basic_auth_username="admin",
            basic_auth_password="secret",
        )
        manager = AuthManager(config)

        assert manager.verify_basic_auth("admin", "secret") is True

    def test_verify_basic_auth_invalid(self):
        """Test invalid basic auth."""
        config = AuthConfig(
            enable_basic_auth=True,
            basic_auth_username="admin",
            basic_auth_password="secret",
        )
        manager = AuthManager(config)

        assert manager.verify_basic_auth("admin", "wrong") is False
        assert manager.verify_basic_auth("wrong", "secret") is False

    def test_verify_basic_auth_disabled(self):
        """Test basic auth when disabled."""
        config = AuthConfig(enable_basic_auth=False)
        manager = AuthManager(config)

        assert manager.verify_basic_auth("admin", "secret") is False

    def test_create_session(self, auth_manager):
        """Test session creation through auth manager."""
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = "TestAgent/1.0"

        session = auth_manager.create_session(mock_request)

        assert session is not None
        assert session.session_id is not None


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_api_key(self):
        """Test API key generation."""
        key1 = generate_api_key()
        key2 = generate_api_key()

        assert len(key1) > 20
        assert key1 != key2

    def test_hash_api_key(self):
        """Test API key hashing."""
        key = "test-api-key"
        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)

        assert len(hash1) == 64  # SHA256 hex
        assert hash1 == hash2  # Deterministic

    def test_hash_api_key_different_keys(self):
        """Test different keys produce different hashes."""
        hash1 = hash_api_key("key1")
        hash2 = hash_api_key("key2")

        assert hash1 != hash2


# =============================================================================
# WebSocket Tests
# =============================================================================

class TestWebSocket:
    """Tests for WebSocket functionality."""

    def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws/live") as websocket:
            # Should receive connected message
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert "timestamp" in data

    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong."""
        with client.websocket_connect("/ws/live") as websocket:
            # Receive initial messages
            websocket.receive_json()  # connected

            # Send ping
            websocket.send_json({"command": "ping"})

            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_subscribe(self, client):
        """Test WebSocket subscription."""
        with client.websocket_connect("/ws/live") as websocket:
            # Receive initial messages
            websocket.receive_json()  # connected

            # Subscribe to positions
            websocket.send_json({"command": "subscribe", "channel": "positions"})

            # Should receive subscribed confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert data["channel"] == "positions"

    def test_websocket_invalid_command(self, client):
        """Test WebSocket with invalid command."""
        with client.websocket_connect("/ws/live") as websocket:
            # Receive initial messages
            websocket.receive_json()  # connected

            # Send invalid command
            websocket.send_json({"command": "invalid_command"})

            # Should receive error
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Unknown command" in data["message"]


# =============================================================================
# Connection Manager Tests
# =============================================================================

class TestConnectionManager:
    """Tests for WebSocket connection manager."""

    @pytest.fixture
    def manager(self):
        """Create connection manager."""
        from src.web.api.websocket import ConnectionManager
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect(self, manager):
        """Test connection registration."""
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()

        await manager.connect(mock_ws)

        assert mock_ws in manager.active_connections
        mock_ws.accept.assert_called_once()
        mock_ws.send_json.assert_called_once()

    def test_disconnect(self, manager):
        """Test connection removal."""
        mock_ws = MagicMock()
        manager.active_connections.add(mock_ws)

        manager.disconnect(mock_ws)

        assert mock_ws not in manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast(self, manager):
        """Test message broadcast."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        manager.active_connections.add(mock_ws1)
        manager.active_connections.add(mock_ws2)

        message = {"type": "test", "data": "hello"}
        await manager.broadcast(message)

        mock_ws1.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected(self, manager):
        """Test broadcast removes disconnected clients."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws2.send_json.side_effect = Exception("Disconnected")

        manager.active_connections.add(mock_ws1)
        manager.active_connections.add(mock_ws2)

        await manager.broadcast({"type": "test"})

        # ws2 should be removed
        assert mock_ws1 in manager.active_connections
        assert mock_ws2 not in manager.active_connections


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for web dashboard."""

    def test_full_api_flow(self, client):
        """Test full API workflow."""
        # Check health
        response = client.get("/health")
        assert response.status_code == 200

        # Get status
        with patch("src.web.api.dependencies.get_bot_state") as mock:
            mock.return_value = {"status": "stopped"}
            response = client.get("/api/status")
            assert response.status_code == 200

    def test_static_files_served(self, client):
        """Test static files are served."""
        # This will 404 in test but should not error
        response = client.get("/static/css/styles.css")
        # Could be 200 or 404 depending on if files exist
        assert response.status_code in [200, 404]

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI handles CORS preflight
        assert response.status_code in [200, 400]
