"""WebSocket handler for real-time updates.

Provides:
- Real-time status updates
- Position changes
- Trade notifications
- Alert broadcasts
"""

import asyncio
import json
from datetime import datetime
from typing import Set, Optional, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ...utils.logging import get_logger
from .dependencies import get_bot_state, get_db

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        logger.info(
            "websocket_connected",
            total_connections=len(self.active_connections),
        )

        # Send initial state
        await self.send_personal_message(
            websocket,
            {
                "type": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Connected to PolymarketBot",
            },
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        self.active_connections.discard(websocket)

        logger.info(
            "websocket_disconnected",
            total_connections=len(self.active_connections),
        )

    async def send_personal_message(self, websocket: WebSocket, message: dict) -> None:
        """Send message to specific connection.

        Args:
            websocket: Target WebSocket
            message: Message to send
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error("websocket_send_error", error=str(e))

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connections.

        Args:
            message: Message to broadcast
        """
        if not self.active_connections:
            return

        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.active_connections.discard(conn)

    async def broadcast_status(self) -> None:
        """Broadcast current status to all connections."""
        state = get_bot_state()

        await self.broadcast({
            "type": "status",
            "timestamp": datetime.utcnow().isoformat(),
            "data": state,
        })

    async def broadcast_position_update(self, position: dict) -> None:
        """Broadcast position update.

        Args:
            position: Position data
        """
        await self.broadcast({
            "type": "position_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": position,
        })

    async def broadcast_trade(self, trade: dict) -> None:
        """Broadcast new trade.

        Args:
            trade: Trade data
        """
        await self.broadcast({
            "type": "trade",
            "timestamp": datetime.utcnow().isoformat(),
            "data": trade,
        })

    async def broadcast_alert(
        self,
        severity: str,
        title: str,
        message: str,
        details: Optional[dict] = None,
    ) -> None:
        """Broadcast alert.

        Args:
            severity: Alert severity
            title: Alert title
            message: Alert message
            details: Additional details
        """
        await self.broadcast({
            "type": "alert",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "severity": severity,
                "title": title,
                "message": message,
                "details": details,
            },
        })

    async def start_periodic_updates(self, interval_s: float = 5.0) -> None:
        """Start periodic status updates.

        Args:
            interval_s: Update interval in seconds
        """
        async def update_loop():
            while True:
                try:
                    await asyncio.sleep(interval_s)
                    await self.broadcast_status()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("periodic_update_error", error=str(e))

        self._broadcast_task = asyncio.create_task(update_loop())

    def stop_periodic_updates(self) -> None:
        """Stop periodic status updates."""
        if self._broadcast_task:
            self._broadcast_task.cancel()


# Global connection manager
manager = ConnectionManager()


@router.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates.

    Provides:
    - Automatic status updates every 5 seconds
    - Position change notifications
    - Trade notifications
    - Alert broadcasts

    Client can send commands:
    - {"command": "subscribe", "channel": "positions"}
    - {"command": "subscribe", "channel": "trades"}
    - {"command": "ping"}
    """
    await manager.connect(websocket)

    subscriptions = {"status"}  # Always subscribe to status

    try:
        # Send initial data
        await send_initial_data(websocket)

        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_json()
                await handle_client_message(websocket, data, subscriptions)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    websocket,
                    {"type": "error", "message": "Invalid JSON"},
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        manager.disconnect(websocket)


async def send_initial_data(websocket: WebSocket) -> None:
    """Send initial state data to new connection.

    Args:
        websocket: WebSocket connection
    """
    # Send current status
    state = get_bot_state()
    await manager.send_personal_message(
        websocket,
        {
            "type": "initial_status",
            "timestamp": datetime.utcnow().isoformat(),
            "data": state,
        },
    )

    # Send positions summary
    try:
        db = get_db()
        from ...database import PositionRepository

        repo = PositionRepository(db)
        positions = repo.get_open_positions()

        await manager.send_personal_message(
            websocket,
            {
                "type": "initial_positions",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "count": len(positions),
                    "positions": [p.to_dict() for p in positions[:10]],  # Limit initial load
                },
            },
        )
    except Exception as e:
        logger.error("initial_positions_error", error=str(e))


async def handle_client_message(
    websocket: WebSocket,
    data: dict,
    subscriptions: set,
) -> None:
    """Handle incoming client message.

    Args:
        websocket: WebSocket connection
        data: Message data
        subscriptions: Client's subscriptions
    """
    command = data.get("command")

    if command == "ping":
        await manager.send_personal_message(
            websocket,
            {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    elif command == "subscribe":
        channel = data.get("channel")
        if channel in ("positions", "trades", "alerts", "status"):
            subscriptions.add(channel)
            await manager.send_personal_message(
                websocket,
                {
                    "type": "subscribed",
                    "channel": channel,
                },
            )

    elif command == "unsubscribe":
        channel = data.get("channel")
        subscriptions.discard(channel)
        await manager.send_personal_message(
            websocket,
            {
                "type": "unsubscribed",
                "channel": channel,
            },
        )

    elif command == "get_positions":
        try:
            db = get_db()
            from ...database import PositionRepository

            repo = PositionRepository(db)
            positions = repo.get_open_positions()

            await manager.send_personal_message(
                websocket,
                {
                    "type": "positions",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": [p.to_dict() for p in positions],
                },
            )
        except Exception as e:
            await manager.send_personal_message(
                websocket,
                {"type": "error", "message": f"Failed to get positions: {e}"},
            )

    elif command == "get_trades":
        limit = data.get("limit", 20)
        try:
            db = get_db()
            from ...database import TradeLogRepository

            repo = TradeLogRepository(db)
            trades = repo.get_recent_trades(limit=limit)

            await manager.send_personal_message(
                websocket,
                {
                    "type": "trades",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": [t.to_dict() for t in trades],
                },
            )
        except Exception as e:
            await manager.send_personal_message(
                websocket,
                {"type": "error", "message": f"Failed to get trades: {e}"},
            )

    else:
        await manager.send_personal_message(
            websocket,
            {"type": "error", "message": f"Unknown command: {command}"},
        )


# Helper functions for broadcasting from other parts of the app

def get_manager() -> ConnectionManager:
    """Get the global connection manager.

    Returns:
        ConnectionManager instance
    """
    return manager


async def notify_position_update(position: dict) -> None:
    """Notify all clients of position update.

    Args:
        position: Position data
    """
    await manager.broadcast_position_update(position)


async def notify_trade(trade: dict) -> None:
    """Notify all clients of new trade.

    Args:
        trade: Trade data
    """
    await manager.broadcast_trade(trade)


async def notify_alert(
    severity: str,
    title: str,
    message: str,
    details: Optional[dict] = None,
) -> None:
    """Notify all clients of alert.

    Args:
        severity: Alert severity
        title: Alert title
        message: Alert message
        details: Additional details
    """
    await manager.broadcast_alert(severity, title, message, details)
