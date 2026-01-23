"""Authenticated user feed for order and fill updates.

This module provides real-time updates for:
- Order status changes (placed, matched, filled, cancelled)
- Trade/fill confirmations
- Position updates
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Callable, Awaitable

from ..utils.logging import get_logger
from .client import AuthenticatedWebSocketClient, ConnectionState, WS_ENDPOINT_USER

logger = get_logger(__name__)


class OrderStatus(str, Enum):
    """Order status from user feed."""

    PENDING = "pending"
    OPEN = "open"
    MATCHED = "matched"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class FillType(str, Enum):
    """Type of fill."""

    MAKER = "maker"
    TAKER = "taker"


@dataclass
class OrderUpdate:
    """Order status update from WebSocket."""

    order_id: str
    status: OrderStatus
    token_id: str
    side: str
    price: Decimal
    original_size: Decimal
    filled_size: Decimal
    remaining_size: Decimal
    timestamp: datetime

    # Optional fill info
    last_fill_price: Optional[Decimal] = None
    last_fill_size: Optional[Decimal] = None
    fill_type: Optional[FillType] = None

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        )

    @property
    def fill_percent(self) -> Decimal:
        """Get fill percentage."""
        if self.original_size == 0:
            return Decimal("0")
        return self.filled_size / self.original_size * 100


@dataclass
class TradeUpdate:
    """Trade/fill update from WebSocket."""

    trade_id: str
    order_id: str
    token_id: str
    side: str
    price: Decimal
    size: Decimal
    fee: Decimal
    timestamp: datetime
    fill_type: FillType

    @property
    def value(self) -> Decimal:
        """Trade value in USD."""
        return self.price * self.size


@dataclass
class PositionUpdate:
    """Position change notification."""

    token_id: str
    condition_id: str
    outcome: str
    size: Decimal
    average_price: Decimal
    current_value: Decimal
    timestamp: datetime


# Callback types
OrderUpdateHandler = Callable[[OrderUpdate], Awaitable[None]]
TradeUpdateHandler = Callable[[TradeUpdate], Awaitable[None]]
PositionUpdateHandler = Callable[[PositionUpdate], Awaitable[None]]


@dataclass
class UserFeedStats:
    """Statistics for user feed."""

    order_updates: int = 0
    trade_updates: int = 0
    position_updates: int = 0
    errors: int = 0
    connected_since: Optional[datetime] = None


class UserFeed:
    """Authenticated feed for order and trade updates.

    Provides real-time notifications for:
    - Order status changes
    - Fill/trade confirmations
    - Position updates

    Example:
        feed = UserFeed(
            api_key="...",
            api_secret="...",
            passphrase="...",
            on_order_update=handle_order,
            on_trade=handle_trade,
        )
        await feed.start()

        # Receive updates via callbacks...

        await feed.stop()
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        on_order_update: Optional[OrderUpdateHandler] = None,
        on_trade: Optional[TradeUpdateHandler] = None,
        on_position: Optional[PositionUpdateHandler] = None,
    ):
        """Initialize user feed.

        Args:
            api_key: CLOB API key
            api_secret: CLOB API secret
            passphrase: CLOB API passphrase
            on_order_update: Callback for order updates
            on_trade: Callback for trade/fill updates
            on_position: Callback for position updates
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

        self._on_order_update = on_order_update
        self._on_trade = on_trade
        self._on_position = on_position

        # WebSocket client
        self._client: Optional[AuthenticatedWebSocketClient] = None

        # Track active orders
        self._orders: dict[str, OrderUpdate] = {}

        # Statistics
        self.stats = UserFeedStats()

        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    async def start(self) -> None:
        """Start the user feed."""
        if self._running:
            return

        self._running = True

        # Create authenticated client
        self._client = AuthenticatedWebSocketClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
            passphrase=self.passphrase,
            on_message=self._handle_message,
            on_state_change=self._handle_state_change,
        )

        # Connect (authentication happens automatically)
        connected = await self._client.connect()

        if connected:
            self.stats.connected_since = datetime.utcnow()
            # Subscribe to user channels
            await self._subscribe_channels()

        logger.info("user_feed_started", connected=connected)

    async def stop(self) -> None:
        """Stop the user feed."""
        if not self._running:
            return

        self._running = False

        if self._client:
            await self._client.close()
            self._client = None

        logger.info(
            "user_feed_stopped",
            order_updates=self.stats.order_updates,
            trade_updates=self.stats.trade_updates,
        )

    def get_order(self, order_id: str) -> Optional[OrderUpdate]:
        """Get tracked order by ID.

        Args:
            order_id: Order ID

        Returns:
            OrderUpdate or None
        """
        return self._orders.get(order_id)

    def get_active_orders(self) -> list[OrderUpdate]:
        """Get all active (non-terminal) orders.

        Returns:
            List of active orders
        """
        return [o for o in self._orders.values() if not o.is_terminal]

    async def _subscribe_channels(self) -> None:
        """Subscribe to user channels."""
        channels = ["orders", "trades", "positions"]

        for channel in channels:
            subscription = {
                "type": "subscribe",
                "channel": channel,
            }
            await self._client.subscribe(subscription)

        logger.debug("user_channels_subscribed", channels=channels)

    async def _handle_message(self, message: dict) -> None:
        """Handle incoming WebSocket message.

        Args:
            message: Parsed JSON message
        """
        msg_type = message.get("type", "")
        channel = message.get("channel", "")

        try:
            if channel == "orders" or msg_type == "order":
                await self._handle_order_update(message)
            elif channel == "trades" or msg_type == "trade":
                await self._handle_trade_update(message)
            elif channel == "positions" or msg_type == "position":
                await self._handle_position_update(message)
            elif msg_type == "subscribed":
                logger.debug("channel_subscribed", channel=message.get("channel"))
            elif msg_type == "error":
                self._handle_error(message)

        except Exception as e:
            logger.error(
                "user_feed_message_error",
                error=str(e),
                message_type=msg_type,
            )
            self.stats.errors += 1

    async def _handle_order_update(self, message: dict) -> None:
        """Handle order status update.

        Args:
            message: Order update message
        """
        data = message.get("data", message)

        order_id = data.get("orderId") or data.get("order_id", "")
        status_str = data.get("status", "").lower()

        # Map status
        status_map = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "live": OrderStatus.OPEN,
            "matched": OrderStatus.MATCHED,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "partial": OrderStatus.PARTIALLY_FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        status = status_map.get(status_str, OrderStatus.PENDING)

        update = OrderUpdate(
            order_id=order_id,
            status=status,
            token_id=data.get("tokenId") or data.get("token_id", ""),
            side=data.get("side", "").upper(),
            price=Decimal(str(data.get("price", "0"))),
            original_size=Decimal(str(data.get("originalSize") or data.get("size", "0"))),
            filled_size=Decimal(str(data.get("filledSize") or data.get("filled_size", "0"))),
            remaining_size=Decimal(str(data.get("remainingSize") or data.get("remaining_size", "0"))),
            timestamp=datetime.utcnow(),
            last_fill_price=Decimal(str(data["lastFillPrice"])) if data.get("lastFillPrice") else None,
            last_fill_size=Decimal(str(data["lastFillSize"])) if data.get("lastFillSize") else None,
        )

        # Track order
        self._orders[order_id] = update
        self.stats.order_updates += 1

        logger.debug(
            "order_update_received",
            order_id=order_id,
            status=status.value,
            filled=str(update.filled_size),
        )

        # Notify handler
        if self._on_order_update:
            await self._on_order_update(update)

        # Clean up terminal orders after some time
        if update.is_terminal:
            asyncio.create_task(self._cleanup_order(order_id, delay=60))

    async def _handle_trade_update(self, message: dict) -> None:
        """Handle trade/fill update.

        Args:
            message: Trade update message
        """
        data = message.get("data", message)

        fill_type_str = data.get("fillType") or data.get("fill_type", "taker")
        fill_type = FillType.MAKER if fill_type_str.lower() == "maker" else FillType.TAKER

        trade = TradeUpdate(
            trade_id=data.get("tradeId") or data.get("trade_id", ""),
            order_id=data.get("orderId") or data.get("order_id", ""),
            token_id=data.get("tokenId") or data.get("token_id", ""),
            side=data.get("side", "").upper(),
            price=Decimal(str(data.get("price", "0"))),
            size=Decimal(str(data.get("size", "0"))),
            fee=Decimal(str(data.get("fee", "0"))),
            timestamp=datetime.utcnow(),
            fill_type=fill_type,
        )

        self.stats.trade_updates += 1

        logger.info(
            "trade_update_received",
            trade_id=trade.trade_id,
            order_id=trade.order_id,
            side=trade.side,
            price=str(trade.price),
            size=str(trade.size),
        )

        # Notify handler
        if self._on_trade:
            await self._on_trade(trade)

    async def _handle_position_update(self, message: dict) -> None:
        """Handle position update.

        Args:
            message: Position update message
        """
        data = message.get("data", message)

        position = PositionUpdate(
            token_id=data.get("tokenId") or data.get("token_id", ""),
            condition_id=data.get("conditionId") or data.get("condition_id", ""),
            outcome=data.get("outcome", ""),
            size=Decimal(str(data.get("size", "0"))),
            average_price=Decimal(str(data.get("avgPrice") or data.get("average_price", "0"))),
            current_value=Decimal(str(data.get("currentValue") or data.get("current_value", "0"))),
            timestamp=datetime.utcnow(),
        )

        self.stats.position_updates += 1

        logger.debug(
            "position_update_received",
            token_id=position.token_id[:16] + "..." if position.token_id else "",
            size=str(position.size),
        )

        # Notify handler
        if self._on_position:
            await self._on_position(position)

    def _handle_error(self, message: dict) -> None:
        """Handle error message.

        Args:
            message: Error message
        """
        error = message.get("error", "Unknown error")
        logger.warning("user_feed_error", error=error)
        self.stats.errors += 1

    async def _handle_state_change(
        self,
        old_state: ConnectionState,
        new_state: ConnectionState,
    ) -> None:
        """Handle WebSocket state changes.

        Args:
            old_state: Previous state
            new_state: New state
        """
        logger.info(
            "user_feed_state_change",
            old_state=old_state.value,
            new_state=new_state.value,
        )

        if new_state == ConnectionState.CONNECTED:
            self.stats.connected_since = datetime.utcnow()
            # Resubscribe to channels
            await self._subscribe_channels()

    async def _cleanup_order(self, order_id: str, delay: float = 60) -> None:
        """Clean up terminal order after delay.

        Args:
            order_id: Order to clean up
            delay: Delay in seconds
        """
        await asyncio.sleep(delay)
        self._orders.pop(order_id, None)


class FillTracker:
    """Tracks order fills and calculates execution quality.

    Integrates with UserFeed to monitor fill rates and
    slippage on executed orders.
    """

    def __init__(self, user_feed: UserFeed):
        """Initialize fill tracker.

        Args:
            user_feed: User feed to track
        """
        self.user_feed = user_feed

        # Track fills per order
        self._fills: dict[str, list[TradeUpdate]] = {}
        self._expected_prices: dict[str, Decimal] = {}

    def set_expected_price(self, order_id: str, price: Decimal) -> None:
        """Set expected execution price for an order.

        Args:
            order_id: Order ID
            price: Expected price
        """
        self._expected_prices[order_id] = price

    async def on_trade(self, trade: TradeUpdate) -> None:
        """Handle trade update.

        Args:
            trade: Trade update
        """
        if trade.order_id not in self._fills:
            self._fills[trade.order_id] = []

        self._fills[trade.order_id].append(trade)

    def get_execution_quality(self, order_id: str) -> Optional[dict]:
        """Get execution quality metrics for an order.

        Args:
            order_id: Order ID

        Returns:
            Dict with execution metrics or None
        """
        fills = self._fills.get(order_id)
        if not fills:
            return None

        expected_price = self._expected_prices.get(order_id)

        # Calculate weighted average fill price
        total_size = sum(f.size for f in fills)
        if total_size == 0:
            return None

        weighted_price = sum(f.price * f.size for f in fills) / total_size
        total_fees = sum(f.fee for f in fills)

        result = {
            "order_id": order_id,
            "fill_count": len(fills),
            "total_size": str(total_size),
            "avg_fill_price": str(weighted_price),
            "total_fees": str(total_fees),
        }

        if expected_price:
            slippage = (weighted_price - expected_price) / expected_price
            result["expected_price"] = str(expected_price)
            result["slippage_percent"] = str(slippage * 100)

        return result

    def clear(self, order_id: Optional[str] = None) -> None:
        """Clear tracked fills.

        Args:
            order_id: Order to clear, or None for all
        """
        if order_id:
            self._fills.pop(order_id, None)
            self._expected_prices.pop(order_id, None)
        else:
            self._fills.clear()
            self._expected_prices.clear()
