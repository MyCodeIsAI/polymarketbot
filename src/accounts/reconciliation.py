"""Position drift detection and account reconciliation.

This module provides:
- Position drift detection (vs target)
- Account state reconciliation with API
- Balance verification with chain state
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Callable, Awaitable

from ..api.data import DataAPIClient
from ..config.models import TargetAccount
from ..utils.logging import get_logger
from .balance import BalanceTracker
from .positions import PositionManager, CopyPosition

logger = get_logger(__name__)


class DriftStatus(str, Enum):
    """Status of position drift."""

    SYNCED = "synced"  # Within tolerance
    MINOR = "minor"  # 5-10% drift
    SIGNIFICANT = "significant"  # 10-20% drift
    CRITICAL = "critical"  # >20% drift
    NO_POSITION = "no_position"  # We have no position but target does


@dataclass
class DriftResult:
    """Result of drift check for a single position."""

    token_id: str
    status: DriftStatus

    # Our position
    our_size: Decimal
    our_value: Decimal

    # Target position
    target_size: Decimal
    target_value: Decimal

    # Expected vs actual
    expected_size: Decimal  # Based on ratio
    drift_percent: Decimal
    drift_shares: Decimal

    # Recommendation
    recommended_action: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "status": self.status.value,
            "our_size": str(self.our_size),
            "target_size": str(self.target_size),
            "expected_size": str(self.expected_size),
            "drift_percent": str(round(self.drift_percent * 100, 2)) + "%",
            "drift_shares": str(self.drift_shares),
            "recommended_action": self.recommended_action,
        }


@dataclass
class ReconciliationResult:
    """Result of account reconciliation."""

    account_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Balance reconciliation
    tracked_balance: Decimal = Decimal("0")
    actual_balance: Decimal = Decimal("0")
    balance_difference: Decimal = Decimal("0")
    balance_synced: bool = False

    # Position reconciliation
    positions_checked: int = 0
    positions_synced: int = 0
    positions_added: int = 0
    positions_removed: int = 0

    # Drift results
    drift_results: list[DriftResult] = field(default_factory=list)

    @property
    def has_balance_discrepancy(self) -> bool:
        """Check if balance has discrepancy."""
        return abs(self.balance_difference) > Decimal("0.01")

    @property
    def has_drift(self) -> bool:
        """Check if any position has significant drift."""
        return any(
            d.status in (DriftStatus.SIGNIFICANT, DriftStatus.CRITICAL)
            for d in self.drift_results
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "account_name": self.account_name,
            "timestamp": self.timestamp.isoformat(),
            "balance": {
                "tracked": str(self.tracked_balance),
                "actual": str(self.actual_balance),
                "difference": str(self.balance_difference),
                "synced": self.balance_synced,
            },
            "positions": {
                "checked": self.positions_checked,
                "synced": self.positions_synced,
                "added": self.positions_added,
                "removed": self.positions_removed,
            },
            "drift_count": len([d for d in self.drift_results if d.status != DriftStatus.SYNCED]),
        }


# Callback type for drift alerts
DriftAlertHandler = Callable[[DriftResult], Awaitable[None]]


class DriftDetector:
    """Detects position drift between our position and target.

    Compares our position size against expected size based on
    the target's position and our configured ratio.

    Example:
        detector = DriftDetector(
            position_ratio=Decimal("0.01"),
            thresholds=DriftThresholds(),
        )

        drift = detector.check_drift(
            our_position=our_pos,
            target_position=target_pos,
            current_price=Decimal("0.50"),
        )

        if drift.status == DriftStatus.CRITICAL:
            # Take action
            pass
    """

    def __init__(
        self,
        position_ratio: Decimal,
        minor_threshold: Decimal = Decimal("0.05"),
        significant_threshold: Decimal = Decimal("0.10"),
        critical_threshold: Decimal = Decimal("0.20"),
        on_drift_alert: Optional[DriftAlertHandler] = None,
    ):
        """Initialize drift detector.

        Args:
            position_ratio: Ratio of our position to target's
            minor_threshold: Threshold for minor drift (5%)
            significant_threshold: Threshold for significant drift (10%)
            critical_threshold: Threshold for critical drift (20%)
            on_drift_alert: Callback for drift alerts
        """
        self.position_ratio = position_ratio
        self.minor_threshold = minor_threshold
        self.significant_threshold = significant_threshold
        self.critical_threshold = critical_threshold
        self._on_drift_alert = on_drift_alert

    def check_drift(
        self,
        our_position: Optional[CopyPosition],
        target_size: Decimal,
        current_price: Decimal,
    ) -> DriftResult:
        """Check drift for a single position.

        Args:
            our_position: Our position (or None)
            target_size: Target's position size
            current_price: Current market price

        Returns:
            DriftResult with drift analysis
        """
        our_size = our_position.size if our_position else Decimal("0")
        our_value = our_size * current_price
        target_value = target_size * current_price

        # Calculate expected size
        expected_size = target_size * self.position_ratio

        # Calculate drift
        if expected_size > 0:
            drift_shares = our_size - expected_size
            drift_percent = abs(drift_shares) / expected_size
        elif our_size > 0:
            # We have position but shouldn't
            drift_shares = our_size
            drift_percent = Decimal("1")
        else:
            # Both zero
            drift_shares = Decimal("0")
            drift_percent = Decimal("0")

        # Determine status
        token_id = our_position.token_id if our_position else "unknown"

        if target_size > 0 and our_size == 0:
            status = DriftStatus.NO_POSITION
            action = f"Buy {expected_size} shares to match target"
        elif drift_percent > self.critical_threshold:
            status = DriftStatus.CRITICAL
            action = f"Rebalance: {'buy' if drift_shares < 0 else 'sell'} {abs(drift_shares):.2f} shares"
        elif drift_percent > self.significant_threshold:
            status = DriftStatus.SIGNIFICANT
            action = f"Consider rebalancing: {drift_percent*100:.1f}% drift"
        elif drift_percent > self.minor_threshold:
            status = DriftStatus.MINOR
            action = None
        else:
            status = DriftStatus.SYNCED
            action = None

        result = DriftResult(
            token_id=token_id,
            status=status,
            our_size=our_size,
            our_value=our_value,
            target_size=target_size,
            target_value=target_value,
            expected_size=expected_size,
            drift_percent=drift_percent,
            drift_shares=drift_shares,
            recommended_action=action,
        )

        # Trigger alert for significant drift
        if status in (DriftStatus.SIGNIFICANT, DriftStatus.CRITICAL, DriftStatus.NO_POSITION):
            if self._on_drift_alert:
                asyncio.create_task(self._on_drift_alert(result))

            logger.warning(
                "position_drift_detected",
                token_id=token_id[:16] + "..." if len(token_id) > 16 else token_id,
                status=status.value,
                drift_percent=f"{drift_percent*100:.1f}%",
            )

        return result


class AccountReconciler:
    """Reconciles account state with API data.

    Periodically syncs:
    - USDC balance
    - Position sizes
    - Target positions for drift detection

    Example:
        reconciler = AccountReconciler(
            data_client=data_client,
            target=target_config,
            balance_tracker=balance_tracker,
            position_manager=position_manager,
        )

        result = await reconciler.reconcile()
        if result.has_drift:
            # Handle drift
            pass
    """

    def __init__(
        self,
        data_client: DataAPIClient,
        target: TargetAccount,
        balance_tracker: BalanceTracker,
        position_manager: PositionManager,
        our_wallet: str,
    ):
        """Initialize reconciler.

        Args:
            data_client: Data API client
            target: Target account configuration
            balance_tracker: Balance tracker to reconcile
            position_manager: Position manager to reconcile
            our_wallet: Our wallet address
        """
        self.data_client = data_client
        self.target = target
        self.balance_tracker = balance_tracker
        self.position_manager = position_manager
        self.our_wallet = our_wallet

        self.drift_detector = DriftDetector(
            position_ratio=target.position_ratio,
        )

        self._last_reconciliation: Optional[ReconciliationResult] = None

    async def reconcile(self) -> ReconciliationResult:
        """Perform full account reconciliation.

        Returns:
            ReconciliationResult with all findings
        """
        result = ReconciliationResult(account_name=self.target.name)

        # Reconcile balance
        await self._reconcile_balance(result)

        # Reconcile our positions
        await self._reconcile_our_positions(result)

        # Check drift against target
        await self._check_all_drift(result)

        self._last_reconciliation = result

        logger.info(
            "reconciliation_complete",
            account=self.target.name,
            balance_diff=str(result.balance_difference),
            positions_synced=result.positions_synced,
            drift_detected=result.has_drift,
        )

        return result

    async def _reconcile_balance(self, result: ReconciliationResult) -> None:
        """Reconcile USDC balance.

        Args:
            result: Result to update
        """
        try:
            # Get actual balance from API
            actual_balance = await self.data_client.get_balance(self.our_wallet)

            result.tracked_balance = self.balance_tracker.total_balance
            result.actual_balance = actual_balance
            result.balance_difference = actual_balance - self.balance_tracker.total_balance

            # Sync if difference is significant
            if abs(result.balance_difference) > Decimal("0.01"):
                self.balance_tracker.sync_balance(actual_balance, source="api")
                result.balance_synced = True

        except Exception as e:
            logger.error("balance_reconciliation_failed", error=str(e))

    async def _reconcile_our_positions(self, result: ReconciliationResult) -> None:
        """Reconcile our positions with API.

        Args:
            result: Result to update
        """
        try:
            # Get positions from API
            api_positions = await self.data_client.get_positions(
                user=self.our_wallet,
            )

            # Convert to dict format
            positions_data = []
            for pos in api_positions:
                positions_data.append({
                    "assetId": pos.token_id,
                    "conditionId": pos.condition_id,
                    "outcome": pos.outcome,
                    "size": str(pos.size),
                    "avgPrice": str(pos.average_price),
                })

            # Sync positions
            added, updated, removed = self.position_manager.sync_from_api(positions_data)

            result.positions_checked = len(api_positions)
            result.positions_synced = updated
            result.positions_added = added
            result.positions_removed = removed

        except Exception as e:
            logger.error("position_reconciliation_failed", error=str(e))

    async def _check_all_drift(self, result: ReconciliationResult) -> None:
        """Check drift for all positions.

        Args:
            result: Result to update
        """
        try:
            # Get target's positions
            target_positions = await self.data_client.get_positions(
                user=self.target.wallet,
            )

            # Build map of target positions
            target_map = {}
            for pos in target_positions:
                target_map[pos.token_id] = pos

            # Get current prices for drift calculation
            # (In practice, would get from order book)

            # Check each position
            checked_tokens = set()

            # Check our positions vs target
            for our_pos in self.position_manager.get_open_positions():
                target_pos = target_map.get(our_pos.token_id)
                target_size = target_pos.size if target_pos else Decimal("0")

                # Use position's current price or average
                price = our_pos.current_price or our_pos.average_price

                drift = self.drift_detector.check_drift(
                    our_position=our_pos,
                    target_size=target_size,
                    current_price=price,
                )
                result.drift_results.append(drift)
                checked_tokens.add(our_pos.token_id)

            # Check target positions we don't have
            for token_id, target_pos in target_map.items():
                if token_id not in checked_tokens:
                    # Target has position we don't
                    drift = self.drift_detector.check_drift(
                        our_position=None,
                        target_size=target_pos.size,
                        current_price=target_pos.average_price,
                    )
                    # Update token_id since we didn't have position
                    drift = DriftResult(
                        token_id=token_id,
                        status=drift.status,
                        our_size=drift.our_size,
                        our_value=drift.our_value,
                        target_size=drift.target_size,
                        target_value=drift.target_value,
                        expected_size=drift.expected_size,
                        drift_percent=drift.drift_percent,
                        drift_shares=drift.drift_shares,
                        recommended_action=drift.recommended_action,
                    )
                    result.drift_results.append(drift)

        except Exception as e:
            logger.error("drift_check_failed", error=str(e))

    @property
    def last_reconciliation(self) -> Optional[ReconciliationResult]:
        """Get last reconciliation result."""
        return self._last_reconciliation


class PeriodicReconciler:
    """Runs reconciliation on a schedule.

    Automatically reconciles account state at configured intervals.
    """

    def __init__(
        self,
        reconciler: AccountReconciler,
        interval_s: int = 60,
        on_drift: Optional[Callable[[ReconciliationResult], Awaitable[None]]] = None,
    ):
        """Initialize periodic reconciler.

        Args:
            reconciler: Account reconciler to use
            interval_s: Reconciliation interval in seconds
            on_drift: Callback when drift is detected
        """
        self.reconciler = reconciler
        self.interval_s = interval_s
        self._on_drift = on_drift

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reconciliation_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start periodic reconciliation."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._reconciliation_loop())

        logger.info(
            "periodic_reconciler_started",
            interval_s=self.interval_s,
        )

    async def stop(self) -> None:
        """Stop periodic reconciliation."""
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
            "periodic_reconciler_stopped",
            reconciliation_count=self._reconciliation_count,
        )

    async def _reconciliation_loop(self) -> None:
        """Main reconciliation loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)

                if not self._running:
                    break

                result = await self.reconciler.reconcile()
                self._reconciliation_count += 1

                # Notify if drift detected
                if result.has_drift and self._on_drift:
                    await self._on_drift(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("periodic_reconciliation_error", error=str(e))
                await asyncio.sleep(10)  # Brief backoff on error
