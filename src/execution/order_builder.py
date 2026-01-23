"""Order building for copy trades.

This module constructs order objects ready for submission to the CLOB,
handling price formatting, size validation, and order metadata.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import Enum
from typing import Optional
from uuid import uuid4

from ..api.clob import OrderSide, OrderType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OrderSource(str, Enum):
    """Source/reason for the order."""

    COPY_TRADE = "copy_trade"
    POSITION_SYNC = "position_sync"
    MANUAL = "manual"
    REBALANCE = "rebalance"


@dataclass
class CopyOrder:
    """A copy trade order ready for execution.

    This is our internal order representation before sending to the CLOB.
    """

    # Identification
    order_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Market info
    token_id: str = ""
    condition_id: str = ""
    outcome: str = ""

    # Order parameters
    side: OrderSide = OrderSide.BUY
    price: Decimal = Decimal("0")
    size: Decimal = Decimal("0")
    order_type: OrderType = OrderType.GTC

    # Expiration (for GTD orders)
    expiration_timestamp: Optional[int] = None

    # Metadata
    source: OrderSource = OrderSource.COPY_TRADE
    target_name: str = ""
    target_order_price: Optional[Decimal] = None  # Original target's price

    # Execution tracking
    submitted: bool = False
    clob_order_id: Optional[str] = None
    fill_size: Decimal = Decimal("0")
    status: str = "pending"

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    @property
    def estimated_cost(self) -> Decimal:
        """Estimated cost for buy orders."""
        if self.is_buy:
            return self.size * self.price
        return Decimal("0")

    @property
    def estimated_proceeds(self) -> Decimal:
        """Estimated proceeds for sell orders."""
        if self.is_sell:
            return self.size * self.price
        return Decimal("0")

    @property
    def remaining_size(self) -> Decimal:
        """Unfilled size."""
        return self.size - self.fill_size

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.fill_size >= self.size

    def to_clob_params(self) -> dict:
        """Convert to CLOB API parameters."""
        params = {
            "token_id": self.token_id,
            "side": self.side.value,
            "price": str(self.price),
            "size": str(self.size),
            "type": self.order_type.value,
        }

        if self.expiration_timestamp and self.order_type == OrderType.GTD:
            params["expiration"] = self.expiration_timestamp

        return params

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "order_id": self.order_id,
            "token_id": self.token_id[:16] + "..." if self.token_id else "",
            "side": self.side.value,
            "price": str(self.price),
            "size": str(self.size),
            "order_type": self.order_type.value,
            "source": self.source.value,
            "target_name": self.target_name,
            "status": self.status,
        }


class OrderBuilder:
    """Builds orders with proper formatting and validation.

    Handles:
    - Price tick size compliance
    - Size rounding
    - Order type selection
    - Expiration calculation

    Example:
        builder = OrderBuilder(tick_size=Decimal("0.01"))
        order = builder.build_market_buy(
            token_id="0x...",
            size=Decimal("100"),
            price=Decimal("0.55"),
            target_name="whale_1",
        )
    """

    def __init__(
        self,
        tick_size: Decimal = Decimal("0.01"),
        min_tick_size: Decimal = Decimal("0.001"),
        default_expiry_minutes: int = 5,
    ):
        """Initialize the order builder.

        Args:
            tick_size: Price tick size for rounding
            min_tick_size: Minimum tick size
            default_expiry_minutes: Default GTD expiration in minutes
        """
        self.tick_size = tick_size
        self.min_tick_size = min_tick_size
        self.default_expiry_minutes = default_expiry_minutes

    def round_price(
        self,
        price: Decimal,
        side: OrderSide,
    ) -> Decimal:
        """Round price to tick size.

        For buys, round up (pay more to ensure fill).
        For sells, round down (receive less to ensure fill).

        Args:
            price: Raw price
            side: Order side

        Returns:
            Tick-aligned price
        """
        # Ensure price is in valid range
        price = max(self.min_tick_size, min(price, Decimal("1") - self.min_tick_size))

        if side == OrderSide.BUY:
            # Round up for buys
            return (price / self.tick_size).quantize(Decimal("1"), rounding=ROUND_UP) * self.tick_size
        else:
            # Round down for sells
            return (price / self.tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.tick_size

    def round_size(self, size: Decimal) -> Decimal:
        """Round size to valid precision.

        Args:
            size: Raw size

        Returns:
            Rounded size
        """
        return size.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def calculate_expiration(self, minutes: Optional[int] = None) -> int:
        """Calculate expiration timestamp.

        Args:
            minutes: Minutes from now, or None for default

        Returns:
            Unix timestamp in seconds
        """
        if minutes is None:
            minutes = self.default_expiry_minutes

        return int(time.time()) + (minutes * 60)

    def build_order(
        self,
        token_id: str,
        condition_id: str,
        outcome: str,
        side: OrderSide,
        size: Decimal,
        price: Decimal,
        order_type: OrderType = OrderType.GTC,
        target_name: str = "",
        target_price: Optional[Decimal] = None,
        source: OrderSource = OrderSource.COPY_TRADE,
        expiry_minutes: Optional[int] = None,
    ) -> CopyOrder:
        """Build a complete order.

        Args:
            token_id: Token ID to trade
            condition_id: Market condition ID
            outcome: Outcome name (Yes/No)
            side: Order side
            size: Order size in shares
            price: Order price
            order_type: Order time-in-force
            target_name: Name of target being copied
            target_price: Original price target executed at
            source: Order source/reason
            expiry_minutes: Expiration for GTD orders

        Returns:
            CopyOrder ready for execution
        """
        # Round price and size
        rounded_price = self.round_price(price, side)
        rounded_size = self.round_size(size)

        # Calculate expiration if needed
        expiration = None
        if order_type == OrderType.GTD:
            expiration = self.calculate_expiration(expiry_minutes)

        order = CopyOrder(
            token_id=token_id,
            condition_id=condition_id,
            outcome=outcome,
            side=side,
            price=rounded_price,
            size=rounded_size,
            order_type=order_type,
            expiration_timestamp=expiration,
            source=source,
            target_name=target_name,
            target_order_price=target_price,
        )

        logger.debug(
            "order_built",
            order_id=order.order_id,
            side=side.value,
            price=str(rounded_price),
            size=str(rounded_size),
            target=target_name,
        )

        return order

    def build_market_buy(
        self,
        token_id: str,
        condition_id: str,
        outcome: str,
        size: Decimal,
        price: Decimal,
        target_name: str = "",
        target_price: Optional[Decimal] = None,
    ) -> CopyOrder:
        """Build a marketable buy order.

        Creates a limit order priced to cross the spread immediately.

        Args:
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name
            size: Order size
            price: Best ask price (order will be at this price)
            target_name: Target being copied
            target_price: Target's execution price

        Returns:
            CopyOrder for immediate execution
        """
        return self.build_order(
            token_id=token_id,
            condition_id=condition_id,
            outcome=outcome,
            side=OrderSide.BUY,
            size=size,
            price=price,
            order_type=OrderType.GTC,
            target_name=target_name,
            target_price=target_price,
            source=OrderSource.COPY_TRADE,
        )

    def build_market_sell(
        self,
        token_id: str,
        condition_id: str,
        outcome: str,
        size: Decimal,
        price: Decimal,
        target_name: str = "",
        target_price: Optional[Decimal] = None,
    ) -> CopyOrder:
        """Build a marketable sell order.

        Creates a limit order priced to cross the spread immediately.

        Args:
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name
            size: Order size
            price: Best bid price (order will be at this price)
            target_name: Target being copied
            target_price: Target's execution price

        Returns:
            CopyOrder for immediate execution
        """
        return self.build_order(
            token_id=token_id,
            condition_id=condition_id,
            outcome=outcome,
            side=OrderSide.SELL,
            size=size,
            price=price,
            order_type=OrderType.GTC,
            target_name=target_name,
            target_price=target_price,
            source=OrderSource.COPY_TRADE,
        )

    def build_passive_order(
        self,
        token_id: str,
        condition_id: str,
        outcome: str,
        side: OrderSide,
        size: Decimal,
        price: Decimal,
        target_name: str = "",
        expiry_minutes: int = 5,
    ) -> CopyOrder:
        """Build a passive limit order (may not fill immediately).

        Used when slippage is too high - places order at target's price
        and waits for fill.

        Args:
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name
            side: Order side
            size: Order size
            price: Limit price (target's price)
            target_name: Target being copied
            expiry_minutes: Order expiration in minutes

        Returns:
            CopyOrder with GTD expiration
        """
        return self.build_order(
            token_id=token_id,
            condition_id=condition_id,
            outcome=outcome,
            side=side,
            size=size,
            price=price,
            order_type=OrderType.GTD,
            target_name=target_name,
            target_price=price,
            source=OrderSource.COPY_TRADE,
            expiry_minutes=expiry_minutes,
        )
