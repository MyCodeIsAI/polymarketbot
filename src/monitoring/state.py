"""Position state tracking for target accounts.

This module maintains local state of all tracked positions, enabling
fast change detection without repeated API calls for full position lists.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PositionStatus(str, Enum):
    """Status of a tracked position."""

    OPEN = "open"
    CLOSED = "closed"
    PENDING_SYNC = "pending_sync"


@dataclass
class TrackedPosition:
    """A position being tracked for a target account.

    Represents the target's position in a specific market outcome.
    """

    # Identification
    target_name: str
    target_wallet: str
    condition_id: str
    token_id: str
    outcome: str

    # Position data
    size: Decimal
    average_price: Decimal
    current_value: Decimal

    # Tracking metadata
    status: PositionStatus = PositionStatus.OPEN
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    last_trade_timestamp: Optional[datetime] = None

    # Copy tracking (filled by execution engine)
    our_size: Decimal = Decimal("0")
    our_average_price: Decimal = Decimal("0")

    @property
    def usd_value(self) -> Decimal:
        """Estimated USD value based on size and average price."""
        return self.size * self.average_price

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.status == PositionStatus.OPEN and self.size > 0

    @property
    def position_key(self) -> str:
        """Unique key for this position."""
        return f"{self.target_wallet}:{self.token_id}"

    def update(
        self,
        size: Decimal,
        average_price: Optional[Decimal] = None,
        current_value: Optional[Decimal] = None,
    ) -> "PositionChange":
        """Update position and return the change.

        Args:
            size: New position size
            average_price: New average price (if available)
            current_value: New current value (if available)

        Returns:
            PositionChange describing what changed
        """
        old_size = self.size
        size_delta = size - old_size

        change_type = ChangeType.NO_CHANGE
        if old_size == 0 and size > 0:
            change_type = ChangeType.OPENED
        elif old_size > 0 and size == 0:
            change_type = ChangeType.CLOSED
        elif size > old_size:
            change_type = ChangeType.INCREASED
        elif size < old_size:
            change_type = ChangeType.DECREASED

        self.size = size
        if average_price is not None:
            self.average_price = average_price
        if current_value is not None:
            self.current_value = current_value
        self.last_updated = datetime.utcnow()

        if size == 0:
            self.status = PositionStatus.CLOSED

        return PositionChange(
            change_type=change_type,
            position=self,
            old_size=old_size,
            new_size=size,
            size_delta=size_delta,
        )


class ChangeType(str, Enum):
    """Type of position change."""

    NO_CHANGE = "no_change"
    OPENED = "opened"
    INCREASED = "increased"
    DECREASED = "decreased"
    CLOSED = "closed"


@dataclass
class PositionChange:
    """Describes a change to a position."""

    change_type: ChangeType
    position: TrackedPosition
    old_size: Decimal
    new_size: Decimal
    size_delta: Decimal

    @property
    def is_significant(self) -> bool:
        """Check if this is a significant change (not NO_CHANGE)."""
        return self.change_type != ChangeType.NO_CHANGE


class PositionStateManager:
    """Manages position state for all tracked targets.

    Thread-safe state management with fast lookups.

    Example:
        manager = PositionStateManager()
        manager.update_position("whale_1", "0x...", position_data)
        position = manager.get_position("whale_1", token_id)
    """

    def __init__(self):
        # Positions indexed by target_name -> token_id -> TrackedPosition
        self._positions: dict[str, dict[str, TrackedPosition]] = {}

        # Last activity timestamp per target (for polling)
        self._last_activity_ts: dict[str, int] = {}

        # Lock for thread-safe updates
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_positions": 0,
            "open_positions": 0,
            "updates": 0,
            "changes_detected": 0,
        }

    async def get_position(
        self,
        target_name: str,
        token_id: str,
    ) -> Optional[TrackedPosition]:
        """Get a specific position.

        Args:
            target_name: Name of the target account
            token_id: Token ID

        Returns:
            TrackedPosition if found, None otherwise
        """
        async with self._lock:
            if target_name not in self._positions:
                return None
            return self._positions[target_name].get(token_id)

    async def get_all_positions(
        self,
        target_name: str,
        open_only: bool = True,
    ) -> list[TrackedPosition]:
        """Get all positions for a target.

        Args:
            target_name: Name of the target account
            open_only: If True, only return open positions

        Returns:
            List of TrackedPosition objects
        """
        async with self._lock:
            if target_name not in self._positions:
                return []

            positions = list(self._positions[target_name].values())

            if open_only:
                positions = [p for p in positions if p.is_open]

            return positions

    async def update_position(
        self,
        target_name: str,
        target_wallet: str,
        token_id: str,
        condition_id: str,
        outcome: str,
        size: Decimal,
        average_price: Decimal,
        current_value: Decimal = Decimal("0"),
    ) -> PositionChange:
        """Update or create a position.

        Args:
            target_name: Name of the target account
            target_wallet: Wallet address
            token_id: Token ID
            condition_id: Market condition ID
            outcome: Outcome name (Yes/No)
            size: Position size
            average_price: Average entry price
            current_value: Current position value

        Returns:
            PositionChange describing what changed
        """
        async with self._lock:
            # Ensure target dict exists
            if target_name not in self._positions:
                self._positions[target_name] = {}

            self._stats["updates"] += 1

            # Check if position exists
            if token_id in self._positions[target_name]:
                position = self._positions[target_name][token_id]
                change = position.update(size, average_price, current_value)
            else:
                # New position
                position = TrackedPosition(
                    target_name=target_name,
                    target_wallet=target_wallet,
                    condition_id=condition_id,
                    token_id=token_id,
                    outcome=outcome,
                    size=size,
                    average_price=average_price,
                    current_value=current_value,
                )
                self._positions[target_name][token_id] = position

                # Determine change type for new position
                if size > 0:
                    change = PositionChange(
                        change_type=ChangeType.OPENED,
                        position=position,
                        old_size=Decimal("0"),
                        new_size=size,
                        size_delta=size,
                    )
                else:
                    change = PositionChange(
                        change_type=ChangeType.NO_CHANGE,
                        position=position,
                        old_size=Decimal("0"),
                        new_size=Decimal("0"),
                        size_delta=Decimal("0"),
                    )

            if change.is_significant:
                self._stats["changes_detected"] += 1
                logger.info(
                    "position_change_detected",
                    target=target_name,
                    token_id=token_id[:16] + "...",
                    change_type=change.change_type.value,
                    old_size=str(change.old_size),
                    new_size=str(change.new_size),
                )

            # Update stats
            self._update_stats()

            return change

    async def remove_position(
        self,
        target_name: str,
        token_id: str,
    ) -> Optional[TrackedPosition]:
        """Remove a position from tracking.

        Args:
            target_name: Name of the target account
            token_id: Token ID

        Returns:
            The removed position if found
        """
        async with self._lock:
            if target_name not in self._positions:
                return None

            position = self._positions[target_name].pop(token_id, None)
            if position:
                self._update_stats()

            return position

    async def set_last_activity_timestamp(
        self,
        target_name: str,
        timestamp: int,
    ) -> None:
        """Set the last seen activity timestamp for a target.

        Used for incremental activity polling.

        Args:
            target_name: Name of the target account
            timestamp: Unix timestamp in seconds
        """
        async with self._lock:
            self._last_activity_ts[target_name] = timestamp

    async def get_last_activity_timestamp(
        self,
        target_name: str,
    ) -> Optional[int]:
        """Get the last seen activity timestamp for a target.

        Args:
            target_name: Name of the target account

        Returns:
            Unix timestamp in seconds, or None if not set
        """
        async with self._lock:
            return self._last_activity_ts.get(target_name)

    async def sync_positions(
        self,
        target_name: str,
        target_wallet: str,
        positions: list[dict],
    ) -> list[PositionChange]:
        """Sync positions from API response.

        Compares current state with API positions and detects all changes.

        Args:
            target_name: Name of the target account
            target_wallet: Wallet address
            positions: List of position dicts from API

        Returns:
            List of PositionChange for all detected changes
        """
        changes = []

        # Track which tokens we've seen in the API response
        seen_tokens: set[str] = set()

        # Process each position from API
        for pos in positions:
            token_id = pos.get("assetId", pos.get("asset_id", pos.get("tokenId", "")))
            if not token_id:
                continue

            seen_tokens.add(token_id)

            change = await self.update_position(
                target_name=target_name,
                target_wallet=target_wallet,
                token_id=token_id,
                condition_id=pos.get("conditionId", pos.get("condition_id", "")),
                outcome=pos.get("outcome", ""),
                size=Decimal(str(pos.get("size", 0))),
                average_price=Decimal(str(pos.get("avgPrice", pos.get("average_price", 0)))),
                current_value=Decimal(str(pos.get("currentValue", pos.get("current_value", 0)))),
            )

            if change.is_significant:
                changes.append(change)

        # Check for closed positions (in our state but not in API)
        async with self._lock:
            if target_name in self._positions:
                current_tokens = set(self._positions[target_name].keys())
                closed_tokens = current_tokens - seen_tokens

                for token_id in closed_tokens:
                    position = self._positions[target_name][token_id]
                    if position.is_open:
                        # Position was closed
                        old_size = position.size
                        change = position.update(Decimal("0"))
                        if change.is_significant:
                            changes.append(change)

        return changes

    async def clear_target(self, target_name: str) -> None:
        """Clear all positions for a target.

        Args:
            target_name: Name of the target account
        """
        async with self._lock:
            if target_name in self._positions:
                del self._positions[target_name]
            if target_name in self._last_activity_ts:
                del self._last_activity_ts[target_name]
            self._update_stats()

    async def clear_all(self) -> None:
        """Clear all tracked positions."""
        async with self._lock:
            self._positions.clear()
            self._last_activity_ts.clear()
            self._update_stats()

    def _update_stats(self) -> None:
        """Update internal statistics (call within lock)."""
        total = 0
        open_count = 0

        for target_positions in self._positions.values():
            for position in target_positions.values():
                total += 1
                if position.is_open:
                    open_count += 1

        self._stats["total_positions"] = total
        self._stats["open_positions"] = open_count

    @property
    def stats(self) -> dict[str, int]:
        """Get state manager statistics."""
        return self._stats.copy()

    async def get_summary(self) -> dict:
        """Get a summary of all tracked state."""
        async with self._lock:
            summary = {
                "targets": {},
                "stats": self._stats.copy(),
            }

            for target_name, positions in self._positions.items():
                open_positions = [p for p in positions.values() if p.is_open]
                summary["targets"][target_name] = {
                    "total_positions": len(positions),
                    "open_positions": len(open_positions),
                    "last_activity_ts": self._last_activity_ts.get(target_name),
                }

            return summary


# Global state manager instance
_state_manager: Optional[PositionStateManager] = None


def get_state_manager() -> PositionStateManager:
    """Get the global position state manager.

    Returns:
        The global PositionStateManager instance
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = PositionStateManager()
    return _state_manager


def reset_state_manager() -> None:
    """Reset the global state manager (mainly for testing)."""
    global _state_manager
    _state_manager = None
