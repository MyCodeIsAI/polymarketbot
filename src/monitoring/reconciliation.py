"""Position reconciliation for catching missed changes.

This module periodically fetches full position state from the API
and reconciles with local state, detecting any trades that may have
been missed by the activity poller.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from ..api.data import DataAPIClient, Position
from ..config.models import TargetAccount
from ..utils.logging import get_logger
from .events import (
    EventBus,
    SyncCompletedEvent,
    PositionOpenedEvent,
    PositionClosedEvent,
    ErrorEvent,
)
from .state import (
    PositionStateManager,
    PositionChange,
    ChangeType,
)

logger = get_logger(__name__)


@dataclass
class ReconciliationResult:
    """Result of a reconciliation run."""

    target_name: str
    positions_synced: int
    new_positions: int
    closed_positions: int
    size_changes: int
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    @property
    def total_changes(self) -> int:
        return self.new_positions + self.closed_positions + self.size_changes


@dataclass
class ReconcilerConfig:
    """Configuration for the reconciler."""

    sync_interval_s: int = 60
    error_backoff_s: int = 30
    max_positions_per_sync: int = 500


class PositionReconciler:
    """Reconciles local position state with API state.

    Periodically fetches full positions from the API and compares
    with local state to detect any missed changes.

    Example:
        reconciler = PositionReconciler(
            target=target_account,
            data_client=data_client,
            event_bus=event_bus,
            state_manager=state_manager,
        )
        await reconciler.start()
    """

    def __init__(
        self,
        target: TargetAccount,
        data_client: DataAPIClient,
        event_bus: EventBus,
        state_manager: PositionStateManager,
        config: Optional[ReconcilerConfig] = None,
    ):
        """Initialize the reconciler.

        Args:
            target: Target account configuration
            data_client: Data API client instance
            event_bus: Event bus for publishing events
            state_manager: Position state manager
            config: Reconciler configuration
        """
        self.target = target
        self.data_client = data_client
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.config = config or ReconcilerConfig()

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0
        self._error_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the reconciliation loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._reconciliation_loop())

        logger.info(
            "reconciler_started",
            target=self.target.name,
            interval_s=self.config.sync_interval_s,
        )

    async def stop(self) -> None:
        """Stop the reconciliation loop."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(
            "reconciler_stopped",
            target=self.target.name,
            sync_count=self._sync_count,
        )

    async def _reconciliation_loop(self) -> None:
        """Main reconciliation loop."""
        # Initial sync on startup
        await self._sync_once()

        while self._running:
            try:
                await asyncio.sleep(self.config.sync_interval_s)

                if not self._running:
                    break

                await self._sync_once()
                self._error_count = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(
                    "reconciliation_error",
                    target=self.target.name,
                    error=str(e),
                )

                await self.event_bus.publish(ErrorEvent(
                    target_name=self.target.name,
                    error_type="reconciliation_error",
                    error_message=str(e),
                    recoverable=True,
                ))

                # Backoff on error
                await asyncio.sleep(self.config.error_backoff_s)

    async def _sync_once(self) -> ReconciliationResult:
        """Execute a single reconciliation sync.

        Returns:
            ReconciliationResult with sync statistics
        """
        logger.debug("reconciliation_starting", target=self.target.name)

        # Fetch positions from API
        api_positions = await self.data_client.get_positions(
            user=self.target.wallet,
            limit=self.config.max_positions_per_sync,
        )

        # Convert to dict format for state manager
        positions_data = []
        for pos in api_positions:
            positions_data.append({
                "assetId": pos.token_id,
                "conditionId": pos.condition_id,
                "outcome": pos.outcome,
                "size": str(pos.size),
                "avgPrice": str(pos.average_price),
                "currentValue": str(pos.current_value),
            })

        # Sync with state manager
        changes = await self.state_manager.sync_positions(
            target_name=self.target.name,
            target_wallet=self.target.wallet,
            positions=positions_data,
        )

        # Count change types
        new_positions = 0
        closed_positions = 0
        size_changes = 0

        for change in changes:
            if change.change_type == ChangeType.OPENED:
                new_positions += 1
                # Emit event for missed new position
                await self._emit_missed_position_opened(change)
            elif change.change_type == ChangeType.CLOSED:
                closed_positions += 1
                # Emit event for missed position close
                await self._emit_missed_position_closed(change)
            elif change.change_type in (ChangeType.INCREASED, ChangeType.DECREASED):
                size_changes += 1

        self._sync_count += 1
        self._last_sync = datetime.utcnow()

        result = ReconciliationResult(
            target_name=self.target.name,
            positions_synced=len(api_positions),
            new_positions=new_positions,
            closed_positions=closed_positions,
            size_changes=size_changes,
        )

        # Publish sync completed event
        await self.event_bus.publish(SyncCompletedEvent(
            target_name=self.target.name,
            target_wallet=self.target.wallet,
            positions_synced=result.positions_synced,
            changes_detected=result.total_changes,
        ))

        if result.total_changes > 0:
            logger.warning(
                "reconciliation_found_changes",
                target=self.target.name,
                new_positions=new_positions,
                closed_positions=closed_positions,
                size_changes=size_changes,
            )
        else:
            logger.debug(
                "reconciliation_complete",
                target=self.target.name,
                positions_synced=result.positions_synced,
            )

        return result

    async def _emit_missed_position_opened(self, change: PositionChange) -> None:
        """Emit event for a position that was missed by polling.

        Args:
            change: The position change detected
        """
        position = change.position
        logger.warning(
            "missed_position_opened",
            target=position.target_name,
            token_id=position.token_id[:16] + "...",
            size=str(position.size),
        )

        await self.event_bus.publish(PositionOpenedEvent(
            target_name=position.target_name,
            target_wallet=position.target_wallet,
            condition_id=position.condition_id,
            token_id=position.token_id,
            outcome=position.outcome,
            size=position.size,
            entry_price=position.average_price,
            usd_value=position.size * position.average_price,
        ))

    async def _emit_missed_position_closed(self, change: PositionChange) -> None:
        """Emit event for a position close that was missed.

        Args:
            change: The position change detected
        """
        position = change.position
        logger.warning(
            "missed_position_closed",
            target=position.target_name,
            token_id=position.token_id[:16] + "...",
            closed_size=str(change.old_size),
        )

        await self.event_bus.publish(PositionClosedEvent(
            target_name=position.target_name,
            target_wallet=position.target_wallet,
            condition_id=position.condition_id,
            token_id=position.token_id,
            outcome=position.outcome,
            closed_size=change.old_size,
            exit_price=position.average_price,  # Approximation
            usd_value=change.old_size * position.average_price,
        ))

    async def force_sync(self) -> ReconciliationResult:
        """Force an immediate synchronization.

        Returns:
            ReconciliationResult with sync statistics
        """
        return await self._sync_once()

    @property
    def last_sync(self) -> Optional[datetime]:
        """Get timestamp of last successful sync."""
        return self._last_sync

    @property
    def stats(self) -> dict:
        """Get reconciler statistics."""
        return {
            "sync_count": self._sync_count,
            "error_count": self._error_count,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
        }


class MultiTargetReconciler:
    """Manages reconciliation for multiple targets.

    Coordinates reconciliation across all targets with staggered
    timing to avoid rate limit issues.
    """

    def __init__(
        self,
        targets: list[TargetAccount],
        data_client: DataAPIClient,
        event_bus: EventBus,
        state_manager: PositionStateManager,
        config: Optional[ReconcilerConfig] = None,
    ):
        """Initialize multi-target reconciler.

        Args:
            targets: List of target accounts
            data_client: Shared Data API client
            event_bus: Shared event bus
            state_manager: Shared state manager
            config: Reconciler configuration
        """
        self.targets = targets
        self.config = config or ReconcilerConfig()

        self._reconcilers: dict[str, PositionReconciler] = {}

        for target in targets:
            self._reconcilers[target.name] = PositionReconciler(
                target=target,
                data_client=data_client,
                event_bus=event_bus,
                state_manager=state_manager,
                config=config,
            )

    async def start_all(self) -> None:
        """Start reconciliation for all enabled targets."""
        # Stagger start times to avoid API bursts
        delay_between = self.config.sync_interval_s / max(len(self.targets), 1)

        for i, (name, reconciler) in enumerate(self._reconcilers.items()):
            if reconciler.target.enabled:
                # Stagger the starts
                if i > 0:
                    await asyncio.sleep(delay_between)
                await reconciler.start()

        logger.info(
            "multi_reconciler_started",
            targets=[t.name for t in self.targets if t.enabled],
        )

    async def stop_all(self) -> None:
        """Stop all reconcilers."""
        for reconciler in self._reconcilers.values():
            await reconciler.stop()

        logger.info("multi_reconciler_stopped")

    async def force_sync_all(self) -> dict[str, ReconciliationResult]:
        """Force sync all targets.

        Returns:
            Dictionary mapping target name to ReconciliationResult
        """
        results = {}

        for name, reconciler in self._reconcilers.items():
            if reconciler.target.enabled:
                results[name] = await reconciler.force_sync()

        return results

    def get_reconciler(self, target_name: str) -> Optional[PositionReconciler]:
        """Get reconciler for a specific target."""
        return self._reconcilers.get(target_name)

    def get_stats(self) -> dict[str, dict]:
        """Get statistics for all reconcilers."""
        return {name: rec.stats for name, rec in self._reconcilers.items()}
