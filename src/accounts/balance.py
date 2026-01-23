"""Balance tracking for copy trading accounts.

This module provides:
- USDC balance tracking
- Reserved amounts for pending orders
- Available balance calculation
- Balance change history
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Callable, Awaitable

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BalanceChangeType(str, Enum):
    """Types of balance changes."""

    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRADE_BUY = "trade_buy"
    TRADE_SELL = "trade_sell"
    FEE = "fee"
    RESERVE = "reserve"
    RELEASE = "release"
    SYNC = "sync"
    SETTLEMENT = "settlement"


@dataclass
class BalanceChange:
    """Record of a balance change."""

    change_type: BalanceChangeType
    amount: Decimal
    balance_before: Decimal
    balance_after: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reference_id: Optional[str] = None  # Order ID, trade ID, etc.
    notes: Optional[str] = None

    @property
    def is_debit(self) -> bool:
        """Check if this change reduced balance."""
        return self.balance_after < self.balance_before

    @property
    def is_credit(self) -> bool:
        """Check if this change increased balance."""
        return self.balance_after > self.balance_before


@dataclass
class BalanceSnapshot:
    """Snapshot of balance state at a point in time."""

    total_balance: Decimal
    reserved_balance: Decimal
    available_balance: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_balance": str(self.total_balance),
            "reserved_balance": str(self.reserved_balance),
            "available_balance": str(self.available_balance),
            "timestamp": self.timestamp.isoformat(),
        }


# Callback type for balance change notifications
BalanceChangeHandler = Callable[[BalanceChange], Awaitable[None]]


class BalanceTracker:
    """Tracks USDC balance with reservations.

    Manages:
    - Total balance
    - Reserved balance (for pending orders)
    - Available balance (total - reserved)
    - Change history

    Example:
        tracker = BalanceTracker(initial_balance=Decimal("1000"))

        # Reserve for pending order
        tracker.reserve(Decimal("100"), order_id="order123")

        # Check available
        available = tracker.available_balance  # 900

        # Order filled, release reservation
        tracker.release(Decimal("100"), order_id="order123")
        tracker.record_buy(Decimal("100"), order_id="order123")
    """

    def __init__(
        self,
        initial_balance: Decimal = Decimal("0"),
        on_change: Optional[BalanceChangeHandler] = None,
        max_history: int = 1000,
    ):
        """Initialize balance tracker.

        Args:
            initial_balance: Starting USDC balance
            on_change: Callback for balance changes
            max_history: Maximum history entries to keep
        """
        self._total_balance = initial_balance
        self._reserved_balance = Decimal("0")
        self._on_change = on_change
        self._max_history = max_history

        # Track reservations by order ID
        self._reservations: dict[str, Decimal] = {}

        # Change history
        self._history: list[BalanceChange] = []

        # Statistics
        self._total_deposited = Decimal("0")
        self._total_withdrawn = Decimal("0")
        self._total_fees_paid = Decimal("0")

        self._last_sync: Optional[datetime] = None

    @property
    def total_balance(self) -> Decimal:
        """Get total balance (including reserved)."""
        return self._total_balance

    @property
    def reserved_balance(self) -> Decimal:
        """Get reserved balance."""
        return self._reserved_balance

    @property
    def available_balance(self) -> Decimal:
        """Get available balance (total - reserved)."""
        return self._total_balance - self._reserved_balance

    @property
    def reservation_count(self) -> int:
        """Get number of active reservations."""
        return len(self._reservations)

    def get_snapshot(self) -> BalanceSnapshot:
        """Get current balance snapshot.

        Returns:
            BalanceSnapshot with current state
        """
        return BalanceSnapshot(
            total_balance=self._total_balance,
            reserved_balance=self._reserved_balance,
            available_balance=self.available_balance,
        )

    def reserve(
        self,
        amount: Decimal,
        order_id: str,
    ) -> bool:
        """Reserve balance for a pending order.

        Args:
            amount: Amount to reserve
            order_id: Order ID for tracking

        Returns:
            True if reservation successful
        """
        if amount <= 0:
            return False

        if amount > self.available_balance:
            logger.warning(
                "reserve_failed_insufficient",
                amount=str(amount),
                available=str(self.available_balance),
            )
            return False

        # Track reservation
        self._reservations[order_id] = amount
        balance_before = self._reserved_balance
        self._reserved_balance += amount

        # Record change
        change = BalanceChange(
            change_type=BalanceChangeType.RESERVE,
            amount=amount,
            balance_before=self._total_balance,
            balance_after=self._total_balance,  # Total doesn't change
            reference_id=order_id,
            notes=f"Reserved for order, available: {self.available_balance}",
        )
        self._record_change(change)

        logger.debug(
            "balance_reserved",
            amount=str(amount),
            order_id=order_id,
            available=str(self.available_balance),
        )

        return True

    def release(
        self,
        order_id: str,
        amount: Optional[Decimal] = None,
    ) -> Decimal:
        """Release a reservation.

        Args:
            order_id: Order ID to release
            amount: Specific amount to release, or None for full reservation

        Returns:
            Amount released
        """
        if order_id not in self._reservations:
            return Decimal("0")

        reserved = self._reservations[order_id]

        if amount is None:
            amount = reserved
        else:
            amount = min(amount, reserved)

        # Update reservation
        remaining = reserved - amount
        if remaining <= 0:
            del self._reservations[order_id]
        else:
            self._reservations[order_id] = remaining

        self._reserved_balance -= amount

        # Record change
        change = BalanceChange(
            change_type=BalanceChangeType.RELEASE,
            amount=amount,
            balance_before=self._total_balance,
            balance_after=self._total_balance,
            reference_id=order_id,
            notes=f"Released reservation, available: {self.available_balance}",
        )
        self._record_change(change)

        logger.debug(
            "balance_released",
            amount=str(amount),
            order_id=order_id,
            available=str(self.available_balance),
        )

        return amount

    def record_buy(
        self,
        cost: Decimal,
        order_id: Optional[str] = None,
        fee: Decimal = Decimal("0"),
    ) -> None:
        """Record a buy trade (reduces balance).

        Args:
            cost: Trade cost in USDC
            order_id: Order ID
            fee: Trading fee
        """
        balance_before = self._total_balance
        self._total_balance -= cost

        # Record trade
        change = BalanceChange(
            change_type=BalanceChangeType.TRADE_BUY,
            amount=cost,
            balance_before=balance_before,
            balance_after=self._total_balance,
            reference_id=order_id,
        )
        self._record_change(change)

        # Record fee if any
        if fee > 0:
            self._record_fee(fee, order_id)

        logger.debug(
            "balance_buy_recorded",
            cost=str(cost),
            fee=str(fee),
            balance=str(self._total_balance),
        )

    def record_sell(
        self,
        proceeds: Decimal,
        order_id: Optional[str] = None,
        fee: Decimal = Decimal("0"),
    ) -> None:
        """Record a sell trade (increases balance).

        Args:
            proceeds: Trade proceeds in USDC
            order_id: Order ID
            fee: Trading fee
        """
        balance_before = self._total_balance
        self._total_balance += proceeds

        # Record trade
        change = BalanceChange(
            change_type=BalanceChangeType.TRADE_SELL,
            amount=proceeds,
            balance_before=balance_before,
            balance_after=self._total_balance,
            reference_id=order_id,
        )
        self._record_change(change)

        # Record fee if any
        if fee > 0:
            self._record_fee(fee, order_id)

        logger.debug(
            "balance_sell_recorded",
            proceeds=str(proceeds),
            fee=str(fee),
            balance=str(self._total_balance),
        )

    def record_settlement(
        self,
        amount: Decimal,
        market_id: str,
    ) -> None:
        """Record market settlement (winning position).

        Args:
            amount: Settlement amount received
            market_id: Market that settled
        """
        balance_before = self._total_balance
        self._total_balance += amount

        change = BalanceChange(
            change_type=BalanceChangeType.SETTLEMENT,
            amount=amount,
            balance_before=balance_before,
            balance_after=self._total_balance,
            reference_id=market_id,
            notes="Market settlement",
        )
        self._record_change(change)

        logger.info(
            "settlement_recorded",
            amount=str(amount),
            market=market_id,
            balance=str(self._total_balance),
        )

    def sync_balance(
        self,
        actual_balance: Decimal,
        source: str = "api",
    ) -> Decimal:
        """Sync with actual balance from API/chain.

        Args:
            actual_balance: Actual balance from external source
            source: Source of the sync (api, chain, etc.)

        Returns:
            Difference between tracked and actual
        """
        difference = actual_balance - self._total_balance

        if difference != 0:
            balance_before = self._total_balance
            self._total_balance = actual_balance

            change = BalanceChange(
                change_type=BalanceChangeType.SYNC,
                amount=abs(difference),
                balance_before=balance_before,
                balance_after=self._total_balance,
                notes=f"Sync from {source}, diff: {difference}",
            )
            self._record_change(change)

            if abs(difference) > Decimal("1"):
                logger.warning(
                    "balance_sync_difference",
                    difference=str(difference),
                    source=source,
                )
            else:
                logger.debug(
                    "balance_synced",
                    difference=str(difference),
                    source=source,
                )

        self._last_sync = datetime.utcnow()
        return difference

    def _record_fee(self, fee: Decimal, reference_id: Optional[str]) -> None:
        """Record a trading fee.

        Args:
            fee: Fee amount
            reference_id: Related order/trade ID
        """
        balance_before = self._total_balance
        self._total_balance -= fee
        self._total_fees_paid += fee

        change = BalanceChange(
            change_type=BalanceChangeType.FEE,
            amount=fee,
            balance_before=balance_before,
            balance_after=self._total_balance,
            reference_id=reference_id,
        )
        self._record_change(change)

    def _record_change(self, change: BalanceChange) -> None:
        """Record a balance change.

        Args:
            change: Change to record
        """
        self._history.append(change)

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Notify handler
        if self._on_change:
            asyncio.create_task(self._on_change(change))

    def get_history(
        self,
        limit: int = 100,
        change_type: Optional[BalanceChangeType] = None,
    ) -> list[BalanceChange]:
        """Get balance change history.

        Args:
            limit: Maximum entries to return
            change_type: Filter by change type

        Returns:
            List of balance changes
        """
        history = self._history

        if change_type:
            history = [c for c in history if c.change_type == change_type]

        return history[-limit:]

    def get_stats(self) -> dict:
        """Get balance statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_balance": str(self._total_balance),
            "reserved_balance": str(self._reserved_balance),
            "available_balance": str(self.available_balance),
            "reservation_count": self.reservation_count,
            "total_fees_paid": str(self._total_fees_paid),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "history_entries": len(self._history),
        }


class MultiAccountBalanceTracker:
    """Tracks balances across multiple accounts.

    Used when copying multiple targets with separate balance tracking.
    """

    def __init__(self):
        """Initialize multi-account tracker."""
        self._trackers: dict[str, BalanceTracker] = {}
        self._total_allocated: Decimal = Decimal("0")

    def add_account(
        self,
        account_name: str,
        initial_balance: Decimal,
        on_change: Optional[BalanceChangeHandler] = None,
    ) -> BalanceTracker:
        """Add an account to track.

        Args:
            account_name: Account identifier
            initial_balance: Starting balance
            on_change: Change callback

        Returns:
            BalanceTracker for the account
        """
        tracker = BalanceTracker(
            initial_balance=initial_balance,
            on_change=on_change,
        )
        self._trackers[account_name] = tracker
        self._total_allocated += initial_balance

        logger.info(
            "account_added",
            account=account_name,
            balance=str(initial_balance),
        )

        return tracker

    def get_tracker(self, account_name: str) -> Optional[BalanceTracker]:
        """Get tracker for an account.

        Args:
            account_name: Account identifier

        Returns:
            BalanceTracker or None
        """
        return self._trackers.get(account_name)

    def get_total_balance(self) -> Decimal:
        """Get total balance across all accounts."""
        return sum(t.total_balance for t in self._trackers.values())

    def get_total_available(self) -> Decimal:
        """Get total available balance across all accounts."""
        return sum(t.available_balance for t in self._trackers.values())

    def get_all_snapshots(self) -> dict[str, BalanceSnapshot]:
        """Get balance snapshots for all accounts.

        Returns:
            Dict mapping account name to snapshot
        """
        return {
            name: tracker.get_snapshot()
            for name, tracker in self._trackers.items()
        }

    def get_summary(self) -> dict:
        """Get summary across all accounts.

        Returns:
            Summary dictionary
        """
        return {
            "account_count": len(self._trackers),
            "total_balance": str(self.get_total_balance()),
            "total_available": str(self.get_total_available()),
            "total_allocated": str(self._total_allocated),
            "accounts": {
                name: tracker.get_stats()
                for name, tracker in self._trackers.items()
            },
        }
