"""WebSocket client with auto-reconnection.

This module provides a robust WebSocket client that:
- Maintains persistent connection to Polymarket CLOB WebSocket
- Auto-reconnects with exponential backoff on disconnection
- Handles heartbeats and connection health monitoring
- Supports both authenticated and unauthenticated connections
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Awaitable, Any
from collections.abc import Coroutine

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    from websockets.exceptions import (
        ConnectionClosed,
        ConnectionClosedError,
        ConnectionClosedOK,
        InvalidStatusCode,
    )
except ImportError:
    websockets = None
    WebSocketClientProtocol = Any

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Polymarket WebSocket endpoints
WS_ENDPOINT = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_ENDPOINT_USER = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


class ConnectionState(str, Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class ReconnectConfig:
    """Configuration for reconnection behavior."""

    initial_delay_ms: int = 100
    max_delay_ms: int = 30000
    multiplier: float = 2.0
    max_attempts: int = 0  # 0 = unlimited
    jitter: float = 0.1  # Random jitter factor


@dataclass
class ConnectionStats:
    """Statistics for WebSocket connection."""

    connects: int = 0
    disconnects: int = 0
    reconnects: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    last_message_at: Optional[datetime] = None
    last_ping_at: Optional[datetime] = None
    last_pong_at: Optional[datetime] = None
    connected_at: Optional[datetime] = None
    total_connected_time_s: float = 0

    def to_dict(self) -> dict:
        return {
            "connects": self.connects,
            "disconnects": self.disconnects,
            "reconnects": self.reconnects,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
            "uptime_s": round(self.total_connected_time_s, 2),
        }


# Type aliases for callbacks
MessageHandler = Callable[[dict], Awaitable[None]]
StateChangeHandler = Callable[[ConnectionState, ConnectionState], Awaitable[None]]


class WebSocketClient:
    """Robust WebSocket client with auto-reconnection.

    Features:
    - Automatic reconnection with exponential backoff
    - Heartbeat (ping/pong) support
    - Message queuing during reconnection
    - State change notifications
    - Connection statistics

    Example:
        client = WebSocketClient(
            url=WS_ENDPOINT,
            on_message=handle_message,
        )
        await client.connect()

        # Send messages
        await client.send({"type": "subscribe", "channel": "book"})

        # Later...
        await client.close()
    """

    def __init__(
        self,
        url: str = WS_ENDPOINT,
        on_message: Optional[MessageHandler] = None,
        on_state_change: Optional[StateChangeHandler] = None,
        reconnect_config: Optional[ReconnectConfig] = None,
        heartbeat_interval_s: float = 10.0,
        heartbeat_timeout_s: float = 30.0,
        connection_timeout_s: float = 10.0,
        auth_headers: Optional[dict] = None,
    ):
        """Initialize WebSocket client.

        Args:
            url: WebSocket endpoint URL
            on_message: Callback for received messages
            on_state_change: Callback for connection state changes
            reconnect_config: Reconnection behavior configuration
            heartbeat_interval_s: Interval between heartbeats
            heartbeat_timeout_s: Timeout before considering connection dead
            connection_timeout_s: Timeout for initial connection
            auth_headers: Optional authentication headers
        """
        self.url = url
        self._on_message = on_message
        self._on_state_change = on_state_change
        self.reconnect_config = reconnect_config or ReconnectConfig()
        self.heartbeat_interval_s = heartbeat_interval_s
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.connection_timeout_s = connection_timeout_s
        self.auth_headers = auth_headers or {}

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._ws: Optional[WebSocketClientProtocol] = None
        self._should_reconnect = True
        self._reconnect_attempts = 0

        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # Message queue for during reconnection
        self._pending_messages: list[dict] = []
        self._max_pending = 100

        # Statistics
        self.stats = ConnectionStats()

        # Subscriptions to restore on reconnect
        self._subscriptions: list[dict] = []

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._state == ConnectionState.CONNECTED and self._ws is not None

    async def connect(self) -> bool:
        """Establish WebSocket connection.

        Returns:
            True if connected successfully
        """
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            return self.is_connected

        self._should_reconnect = True
        return await self._connect()

    async def _connect(self) -> bool:
        """Internal connection logic.

        Returns:
            True if connected successfully
        """
        await self._set_state(ConnectionState.CONNECTING)

        try:
            if websockets is None:
                raise ImportError("websockets library not installed")

            # Build connection kwargs - handle different websockets library versions
            # v11.0+ renamed extra_headers to additional_headers
            # v13.0+ may require different handling
            connect_kwargs = {
                "ping_interval": None,  # We handle heartbeats ourselves
                "ping_timeout": None,
                "close_timeout": 5,
            }

            # Add headers if provided (compatible with websockets v11.0+)
            if self.auth_headers:
                connect_kwargs["additional_headers"] = self.auth_headers

            self._ws = await asyncio.wait_for(
                websockets.connect(self.url, **connect_kwargs),
                timeout=self.connection_timeout_s,
            )

            await self._set_state(ConnectionState.CONNECTED)
            self.stats.connects += 1
            self.stats.connected_at = datetime.utcnow()

            # Start receive and heartbeat tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Restore subscriptions
            if self._subscriptions:
                await self._restore_subscriptions()

            # Send pending messages
            await self._flush_pending_messages()

            logger.info(
                "websocket_connected",
                url=self.url,
                attempt=self._reconnect_attempts,
            )

            self._reconnect_attempts = 0
            return True

        except asyncio.TimeoutError:
            logger.warning(
                "websocket_connect_timeout",
                url=self.url,
                timeout_s=self.connection_timeout_s,
            )
            await self._handle_disconnect()
            return False

        except Exception as e:
            logger.error(
                "websocket_connect_error",
                url=self.url,
                error=str(e),
            )
            await self._handle_disconnect()
            return False

    async def close(self) -> None:
        """Close the WebSocket connection gracefully."""
        self._should_reconnect = False
        await self._set_state(ConnectionState.CLOSING)

        # Cancel tasks
        for task in [self._receive_task, self._heartbeat_task, self._reconnect_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Update connected time
        if self.stats.connected_at:
            delta = (datetime.utcnow() - self.stats.connected_at).total_seconds()
            self.stats.total_connected_time_s += delta

        await self._set_state(ConnectionState.CLOSED)
        logger.info("websocket_closed", stats=self.stats.to_dict())

    async def send(self, message: dict) -> bool:
        """Send a message over the WebSocket.

        Args:
            message: Message to send (will be JSON-encoded)

        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            # Queue message if reconnecting
            if self._state == ConnectionState.RECONNECTING:
                if len(self._pending_messages) < self._max_pending:
                    self._pending_messages.append(message)
                    return True
            return False

        try:
            data = json.dumps(message)
            await self._ws.send(data)
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(data)
            return True

        except Exception as e:
            logger.warning("websocket_send_error", error=str(e))
            return False

    async def subscribe(self, subscription: dict) -> bool:
        """Subscribe to a channel/topic.

        Args:
            subscription: Subscription message

        Returns:
            True if subscription sent
        """
        # Track subscription for reconnect
        if subscription not in self._subscriptions:
            self._subscriptions.append(subscription)

        return await self.send(subscription)

    async def unsubscribe(self, subscription: dict) -> bool:
        """Unsubscribe from a channel/topic.

        Args:
            subscription: Unsubscription message

        Returns:
            True if unsubscription sent
        """
        # Remove from tracked subscriptions
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

        # Modify to unsubscribe type
        unsub = {**subscription, "type": "unsubscribe"}
        return await self.send(unsub)

    async def _receive_loop(self) -> None:
        """Main receive loop for incoming messages."""
        while self.is_connected:
            try:
                data = await self._ws.recv()
                self.stats.messages_received += 1
                self.stats.bytes_received += len(data)
                self.stats.last_message_at = datetime.utcnow()

                # Parse JSON
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("websocket_invalid_json", data=data[:100])
                    continue

                # Handle both dict and list messages
                # Polymarket can send arrays of trades directly
                if isinstance(message, list):
                    # Wrap list in a dict for consistent handling
                    message = {"type": "trades", "trades": message}

                # Handle different message types
                msg_type = message.get("type", "")

                if msg_type == "pong":
                    self.stats.last_pong_at = datetime.utcnow()
                    continue

                # Dispatch to handler
                if self._on_message:
                    try:
                        await self._on_message(message)
                    except Exception as e:
                        logger.error(
                            "websocket_message_handler_error",
                            error=str(e),
                            message_type=msg_type,
                        )

            except ConnectionClosed as e:
                logger.warning(
                    "websocket_connection_closed",
                    code=e.code,
                    reason=e.reason,
                )
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error("websocket_receive_error", error=str(e))
                break

        # Connection lost, handle disconnect
        await self._handle_disconnect()

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and check for responses."""
        while self.is_connected:
            try:
                await asyncio.sleep(self.heartbeat_interval_s)

                if not self.is_connected:
                    break

                # Send ping
                await self.send({"type": "ping"})
                self.stats.last_ping_at = datetime.utcnow()

                # Check if we received a pong recently
                if self.stats.last_pong_at:
                    since_pong = (datetime.utcnow() - self.stats.last_pong_at).total_seconds()
                    if since_pong > self.heartbeat_timeout_s:
                        logger.warning(
                            "websocket_heartbeat_timeout",
                            since_pong_s=since_pong,
                        )
                        # Force reconnect
                        break

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.warning("websocket_heartbeat_error", error=str(e))

    async def _handle_disconnect(self) -> None:
        """Handle disconnection and trigger reconnect if needed."""
        if self._state in (ConnectionState.CLOSING, ConnectionState.CLOSED):
            return

        # Update stats
        self.stats.disconnects += 1
        if self.stats.connected_at:
            delta = (datetime.utcnow() - self.stats.connected_at).total_seconds()
            self.stats.total_connected_time_s += delta
            self.stats.connected_at = None

        # Cancel tasks
        for task in [self._receive_task, self._heartbeat_task]:
            if task and not task.done():
                task.cancel()

        self._ws = None

        if self._should_reconnect:
            await self._set_state(ConnectionState.RECONNECTING)
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        else:
            await self._set_state(ConnectionState.DISCONNECTED)

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        config = self.reconnect_config

        while self._should_reconnect:
            self._reconnect_attempts += 1

            # Check max attempts
            if config.max_attempts > 0 and self._reconnect_attempts > config.max_attempts:
                logger.error(
                    "websocket_max_reconnects_exceeded",
                    attempts=self._reconnect_attempts,
                )
                await self._set_state(ConnectionState.DISCONNECTED)
                return

            # Calculate delay with exponential backoff
            delay_ms = min(
                config.initial_delay_ms * (config.multiplier ** (self._reconnect_attempts - 1)),
                config.max_delay_ms,
            )

            # Add jitter
            import random
            jitter = delay_ms * config.jitter * (random.random() * 2 - 1)
            delay_ms = max(0, delay_ms + jitter)

            logger.info(
                "websocket_reconnecting",
                attempt=self._reconnect_attempts,
                delay_ms=round(delay_ms),
            )

            await asyncio.sleep(delay_ms / 1000)

            if not self._should_reconnect:
                return

            # Attempt reconnection
            success = await self._connect()

            if success:
                self.stats.reconnects += 1
                return

    async def _restore_subscriptions(self) -> None:
        """Restore subscriptions after reconnect."""
        for subscription in self._subscriptions:
            try:
                await self.send(subscription)
                logger.debug("subscription_restored", subscription=subscription)
            except Exception as e:
                logger.warning(
                    "subscription_restore_failed",
                    subscription=subscription,
                    error=str(e),
                )

    async def _flush_pending_messages(self) -> None:
        """Send queued messages after reconnect."""
        while self._pending_messages and self.is_connected:
            message = self._pending_messages.pop(0)
            await self.send(message)

    async def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state and notify handler.

        Args:
            new_state: New connection state
        """
        old_state = self._state
        self._state = new_state

        if self._on_state_change and old_state != new_state:
            try:
                await self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(
                    "websocket_state_change_handler_error",
                    error=str(e),
                )


class AuthenticatedWebSocketClient(WebSocketClient):
    """WebSocket client with Polymarket authentication.

    Used for user-specific feeds like order updates and fill confirmations.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        **kwargs,
    ):
        """Initialize authenticated WebSocket client.

        Args:
            api_key: API key from CLOB
            api_secret: API secret
            passphrase: API passphrase
            **kwargs: Passed to WebSocketClient
        """
        # Set user endpoint
        kwargs.setdefault("url", WS_ENDPOINT_USER)

        super().__init__(**kwargs)

        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

    async def connect(self) -> bool:
        """Connect with authentication."""
        # First establish connection
        connected = await super().connect()

        if connected:
            # Send authentication message
            auth_success = await self._authenticate()
            if not auth_success:
                await self.close()
                return False

        return connected

    async def _authenticate(self) -> bool:
        """Send authentication message.

        Returns:
            True if authentication succeeded
        """
        import hmac
        import hashlib
        import base64

        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}websocket"

        # Create signature
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode(),
            hashlib.sha256,
        )
        signature_b64 = base64.b64encode(signature.digest()).decode()

        auth_message = {
            "type": "auth",
            "apiKey": self.api_key,
            "passphrase": self.passphrase,
            "timestamp": timestamp,
            "signature": signature_b64,
        }

        success = await self.send(auth_message)

        if success:
            logger.info("websocket_authenticated")
        else:
            logger.error("websocket_authentication_failed")

        return success
