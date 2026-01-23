"""Position management for copy trading accounts.

This module provides:
- Position tracking per account
- Cost basis calculation (FIFO)
- Position aggregation and summary
- Position change detection
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PositionStatus(str, Enum):
    """Status of a position."""

    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class PositionLot:
    """A single lot (buy) in a position for FIFO tracking."""

    size: Decimal
    price: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    order_id: Optional[str] = None

    @property
    def cost_basis(self) -> Decimal:
        """Get cost basis of this lot."""
        return self.size * self.price

    def sell(self, amount: Decimal) -> tuple[Decimal, Decimal]:
        """Sell from this lot.

        Args:
            amount: Amount to sell

        Returns:
            Tuple of (amount_sold, cost_basis_sold)
        """
        sell_amount = min(amount, self.size)
        cost_basis = sell_amount * self.price
        self.size -= sell_amount
        return sell_amount, cost_basis


@dataclass
class CopyPosition:
    """A position in a market being copied.

    Tracks our position alongside the target's position
    for drift detection.
    """

    # Identification
    token_id: str
    condition_id: str
    outcome: str
    market_slug: Optional[str] = None

    # Our position
    size: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")

    # Target's position (for comparison)
    target_size: Decimal = Decimal("0")
    target_avg_price: Decimal = Decimal("0")

    # Current market data
    current_price: Optional[Decimal] = None
    current_value: Optional[Decimal] = None

    # Lots for FIFO accounting
    lots: list[PositionLot] = field(default_factory=list)

    # Timestamps
    opened_at: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None

    # Statistics
    total_bought: Decimal = Decimal("0")
    total_sold: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def status(self) -> PositionStatus:
        """Get position status."""
        if self.size <= 0:
            return PositionStatus.CLOSED
        return PositionStatus.OPEN

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.size > 0

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.current_price is None or self.size <= 0:
            return Decimal("0")
        current_value = self.size * self.current_price
        return current_value - self.total_cost

    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """Calculate unrealized P&L percentage."""
        if self.total_cost <= 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.total_cost) * 100

    @property
    def total_pnl(self) -> Decimal:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def add_shares(
        self,
        size: Decimal,
        price: Decimal,
        order_id: Optional[str] = None,
    ) -> None:
        """Add shares to position (buy).

        Args:
            size: Number of shares
            price: Price per share
            order_id: Order ID for tracking
        """
        cost = size * price

        # Update totals
        total_value = (self.size * self.average_price) + cost
        self.size += size
        self.total_cost += cost
        self.total_bought += size

        # Recalculate average price
        if self.size > 0:
            self.average_price = total_value / self.size

        # Add lot for FIFO
        lot = PositionLot(
            size=size,
            price=price,
            order_id=order_id,
        )
        self.lots.append(lot)

        # Update timestamps
        now = datetime.utcnow()
        if self.opened_at is None:
            self.opened_at = now
        self.last_trade_at = now

        logger.debug(
            "position_shares_added",
            token_id=self.token_id[:16] + "...",
            size=str(size),
            price=str(price),
            total_size=str(self.size),
        )

    def remove_shares(
        self,
        size: Decimal,
        price: Decimal,
        order_id: Optional[str] = None,
    ) -> Decimal:
        """Remove shares from position (sell).

        Uses FIFO to determine cost basis.

        Args:
            size: Number of shares to remove
            price: Sale price per share
            order_id: Order ID for tracking

        Returns:
            Realized P&L from this sale
        """
        if size > self.size:
            size = self.size

        if size <= 0:
            return Decimal("0")

        # Calculate proceeds
        proceeds = size * price

        # Determine cost basis using FIFO
        remaining_to_sell = size
        cost_basis = Decimal("0")

        while remaining_to_sell > 0 and self.lots:
            lot = self.lots[0]
            sold_from_lot, lot_cost = lot.sell(remaining_to_sell)
            cost_basis += lot_cost
            remaining_to_sell -= sold_from_lot

            # Remove empty lots
            if lot.size <= 0:
                self.lots.pop(0)

        # Calculate realized P&L
        realized = proceeds - cost_basis
        self.realized_pnl += realized

        # Update totals
        self.size -= size
        self.total_cost -= cost_basis
        self.total_sold += size

        # Recalculate average price
        if self.size > 0:
            self.average_price = self.total_cost / self.size
        else:
            self.average_price = Decimal("0")
            self.total_cost = Decimal("0")

        self.last_trade_at = datetime.utcnow()

        logger.debug(
            "position_shares_removed",
            token_id=self.token_id[:16] + "...",
            size=str(size),
            price=str(price),
            realized_pnl=str(realized),
            remaining=str(self.size),
        )

        return realized

    def update_market_price(self, price: Decimal) -> None:
        """Update current market price.

        Args:
            price: Current market price
        """
        self.current_price = price
        if self.size > 0:
            self.current_value = self.size * price

    def update_target(self, target_size: Decimal, target_price: Decimal) -> None:
        """Update target's position data.

        Args:
            target_size: Target's current position size
            target_price: Target's average price
        """
        self.target_size = target_size
        self.target_avg_price = target_price

    def close(self, settlement_price: Optional[Decimal] = None) -> Decimal:
        """Close the position (market settled).

        Args:
            settlement_price: Settlement price (1.0 for win, 0.0 for loss)

        Returns:
            Final realized P&L
        """
        if self.size <= 0:
            return Decimal("0")

        if settlement_price is not None:
            # Calculate settlement value
            settlement_value = self.size * settlement_price
            realized = settlement_value - self.total_cost
            self.realized_pnl += realized
        else:
            realized = Decimal("0")

        # Clear position
        self.size = Decimal("0")
        self.total_cost = Decimal("0")
        self.average_price = Decimal("0")
        self.lots.clear()
        self.current_value = Decimal("0")

        logger.info(
            "position_closed",
            token_id=self.token_id[:16] + "...",
            settlement_price=str(settlement_price) if settlement_price else None,
            realized_pnl=str(realized),
        )

        return realized

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "outcome": self.outcome,
            "size": str(self.size),
            "average_price": str(self.average_price),
            "total_cost": str(self.total_cost),
            "current_price": str(self.current_price) if self.current_price else None,
            "current_value": str(self.current_value) if self.current_value else None,
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "status": self.status.value,
        }


class PositionManager:
    """Manages positions for a copy trading account.

    Handles:
    - Position tracking by token ID
    - Aggregated position statistics
    - Position updates from trades

    Example:
        manager = PositionManager(account_name="whale_copy")

        # Record a buy
        manager.record_buy(
            token_id="0x...",
            condition_id="0x...",
            outcome="Yes",
            size=Decimal("100"),
            price=Decimal("0.50"),
        )

        # Get position
        position = manager.get_position("0x...")
        print(f"Unrealized P&L: {position.unrealized_pnl}")
    """

    def __init__(self, account_name: str):
        """Initialize position manager.

        Args:
            account_name: Account identifier
        """
        self.account_name = account_name
        self._positions: dict[str, CopyPosition] = {}

    def get_position(self, token_id: str) -> Optional[CopyPosition]:
        """Get position by token ID.

        Args:
            token_id: Token ID

        Returns:
            CopyPosition or None
        """
        return self._positions.get(token_id)

    def get_or_create_position(
        self,
        token_id: str,
        condition_id: str,
        outcome: str,
    ) -> CopyPosition:
        """Get or create a position.

        Args:
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name

        Returns:
            CopyPosition
        """
        if token_id not in self._positions:
            self._positions[token_id] = CopyPosition(
                token_id=token_id,
                condition_id=condition_id,
                outcome=outcome,
            )
        return self._positions[token_id]

    def record_buy(
        self,
        token_id: str,
        condition_id: str,
        outcome: str,
        size: Decimal,
        price: Decimal,
        order_id: Optional[str] = None,
    ) -> CopyPosition:
        """Record a buy trade.

        Args:
            token_id: Token ID
            condition_id: Condition ID
            outcome: Outcome name
            size: Shares bought
            price: Price per share
            order_id: Order ID

        Returns:
            Updated position
        """
        position = self.get_or_create_position(token_id, condition_id, outcome)
        position.add_shares(size, price, order_id)
        return position

    def record_sell(
        self,
        token_id: str,
        size: Decimal,
        price: Decimal,
        order_id: Optional[str] = None,
    ) -> tuple[Optional[CopyPosition], Decimal]:
        """Record a sell trade.

        Args:
            token_id: Token ID
            size: Shares sold
            price: Price per share
            order_id: Order ID

        Returns:
            Tuple of (position, realized_pnl)
        """
        position = self.get_position(token_id)
        if position is None:
            logger.warning("sell_without_position", token_id=token_id[:16] + "...")
            return None, Decimal("0")

        realized = position.remove_shares(size, price, order_id)

        # Remove closed positions
        if position.size <= 0:
            self._remove_if_closed(token_id)

        return position, realized

    def record_settlement(
        self,
        token_id: str,
        settlement_price: Decimal,
    ) -> Decimal:
        """Record market settlement.

        Args:
            token_id: Token ID
            settlement_price: Settlement price (0 or 1)

        Returns:
            Realized P&L from settlement
        """
        position = self.get_position(token_id)
        if position is None:
            return Decimal("0")

        realized = position.close(settlement_price)

        # Remove closed position
        self._remove_if_closed(token_id)

        return realized

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """Update current market prices for all positions.

        Args:
            prices: Dict mapping token_id to current price
        """
        for token_id, price in prices.items():
            position = self.get_position(token_id)
            if position:
                position.update_market_price(price)

    def get_open_positions(self) -> list[CopyPosition]:
        """Get all open positions.

        Returns:
            List of open positions
        """
        return [p for p in self._positions.values() if p.is_open]

    def get_all_positions(self) -> list[CopyPosition]:
        """Get all positions (including closed).

        Returns:
            List of all positions
        """
        return list(self._positions.values())

    def get_position_count(self) -> int:
        """Get count of open positions."""
        return len([p for p in self._positions.values() if p.is_open])

    def get_total_value(self) -> Decimal:
        """Get total current value of all positions.

        Returns:
            Total value in USDC
        """
        total = Decimal("0")
        for position in self._positions.values():
            if position.current_value:
                total += position.current_value
        return total

    def get_total_cost(self) -> Decimal:
        """Get total cost basis of all positions.

        Returns:
            Total cost in USDC
        """
        return sum(p.total_cost for p in self._positions.values())

    def get_total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L.

        Returns:
            Total unrealized P&L
        """
        return sum(p.unrealized_pnl for p in self._positions.values())

    def get_total_realized_pnl(self) -> Decimal:
        """Get total realized P&L.

        Returns:
            Total realized P&L
        """
        return sum(p.realized_pnl for p in self._positions.values())

    def _remove_if_closed(self, token_id: str) -> None:
        """Remove position if closed.

        Args:
            token_id: Token ID to check
        """
        position = self._positions.get(token_id)
        if position and position.size <= 0:
            # Keep for history but could implement cleanup
            pass

    def get_summary(self) -> dict:
        """Get position summary.

        Returns:
            Summary dictionary
        """
        open_positions = self.get_open_positions()

        return {
            "account_name": self.account_name,
            "open_positions": len(open_positions),
            "total_cost": str(self.get_total_cost()),
            "total_value": str(self.get_total_value()),
            "unrealized_pnl": str(self.get_total_unrealized_pnl()),
            "realized_pnl": str(self.get_total_realized_pnl()),
            "total_pnl": str(self.get_total_unrealized_pnl() + self.get_total_realized_pnl()),
        }

    def sync_from_api(
        self,
        positions: list[dict],
    ) -> tuple[int, int, int]:
        """Sync positions from API data.

        Args:
            positions: List of position data from API

        Returns:
            Tuple of (added, updated, removed) counts
        """
        added = 0
        updated = 0
        removed = 0

        api_token_ids = set()

        for pos_data in positions:
            token_id = pos_data.get("assetId") or pos_data.get("token_id", "")
            api_token_ids.add(token_id)

            size = Decimal(str(pos_data.get("size", "0")))
            avg_price = Decimal(str(pos_data.get("avgPrice") or pos_data.get("average_price", "0")))

            existing = self.get_position(token_id)

            if existing:
                # Update existing position
                existing.size = size
                existing.average_price = avg_price
                existing.total_cost = size * avg_price
                updated += 1
            else:
                # Add new position
                position = CopyPosition(
                    token_id=token_id,
                    condition_id=pos_data.get("conditionId") or pos_data.get("condition_id", ""),
                    outcome=pos_data.get("outcome", ""),
                    size=size,
                    average_price=avg_price,
                    total_cost=size * avg_price,
                )
                self._positions[token_id] = position
                added += 1

        # Mark removed positions
        for token_id in list(self._positions.keys()):
            if token_id not in api_token_ids:
                position = self._positions[token_id]
                if position.size > 0:
                    position.size = Decimal("0")
                    removed += 1

        logger.info(
            "positions_synced",
            account=self.account_name,
            added=added,
            updated=updated,
            removed=removed,
        )

        return added, updated, removed
