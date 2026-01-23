"""Database service layer.

Provides high-level business logic on top of repositories:
- Trade logging with full audit trail
- Position persistence and synchronization
- Recovery from restart
- Statistics aggregation
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any

from ..utils.logging import get_logger
from .connection import DatabaseManager
from .models import (
    TrackedAccount,
    Position,
    TradeLog,
    PositionStatus,
    TradeStatus,
    EventSeverity,
)
from .repositories import (
    AccountRepository,
    PositionRepository,
    TradeLogRepository,
    SystemEventRepository,
    BalanceSnapshotRepository,
    CircuitBreakerEventRepository,
    DailyStatsRepository,
)

logger = get_logger(__name__)


@dataclass
class TradeContext:
    """Context for a trade execution."""

    account_id: int
    account_name: str
    market_id: str
    condition_id: str
    token_id: str
    outcome: str
    side: str
    target_price: Decimal
    target_size: Decimal
    detected_at: datetime

    # Optional fields
    target_order_id: Optional[str] = None
    target_tx_hash: Optional[str] = None
    slippage_allowed: Optional[Decimal] = None


@dataclass
class TradeResult:
    """Result of a trade execution."""

    success: bool
    trade_id: int
    status: TradeStatus

    # Execution details
    execution_price: Optional[Decimal] = None
    execution_size: Optional[Decimal] = None
    execution_cost: Optional[Decimal] = None
    slippage_percent: Optional[Decimal] = None

    # Order info
    order_id: Optional[str] = None
    tx_hash: Optional[str] = None

    # Error info
    error_message: Optional[str] = None

    # Latency
    detection_latency_ms: Optional[int] = None
    execution_latency_ms: Optional[int] = None


class TradeLogger:
    """High-level trade logging service.

    Provides full audit trail for all trade attempts.

    Example:
        logger = TradeLogger(db)

        # Log when trade detected
        trade_id = logger.log_detected(context)

        # Update when queued
        logger.log_queued(trade_id)

        # Update on execution
        logger.log_result(trade_id, result)
    """

    def __init__(self, db: DatabaseManager):
        """Initialize trade logger.

        Args:
            db: Database manager
        """
        self.db = db
        self._trade_repo = TradeLogRepository(db)
        self._stats_repo = DailyStatsRepository(db)
        self._event_repo = SystemEventRepository(db)

    def log_detected(self, context: TradeContext) -> int:
        """Log a detected trade.

        Args:
            context: Trade context

        Returns:
            Trade log ID
        """
        # Ensure daily stats exist
        self._stats_repo.get_or_create_today()
        self._stats_repo.increment_trades_detected()

        trade = self._trade_repo.log_trade(
            account_id=context.account_id,
            market_id=context.market_id,
            condition_id=context.condition_id,
            token_id=context.token_id,
            outcome=context.outcome,
            side=context.side,
            target_price=context.target_price,
            target_size=context.target_size,
            detected_at=context.detected_at,
            slippage_allowed=context.slippage_allowed,
        )

        logger.debug(
            "trade_logged_detected",
            trade_id=trade.id,
            account_id=context.account_id,
            side=context.side,
        )

        return trade.id

    def log_queued(self, trade_id: int) -> None:
        """Log trade queued for execution.

        Args:
            trade_id: Trade log ID
        """
        self._trade_repo.update_status(
            trade_id,
            TradeStatus.QUEUED,
            queued_at=datetime.utcnow(),
        )

    def log_executing(self, trade_id: int) -> None:
        """Log trade execution started.

        Args:
            trade_id: Trade log ID
        """
        self._trade_repo.update_status(
            trade_id,
            TradeStatus.EXECUTING,
        )

    def log_result(self, trade_id: int, result: TradeResult) -> None:
        """Log trade execution result.

        Args:
            trade_id: Trade log ID
            result: Execution result
        """
        updates = {
            "execution_price": result.execution_price,
            "execution_size": result.execution_size,
            "execution_cost": result.execution_cost,
            "slippage_percent": result.slippage_percent,
            "order_id": result.order_id,
            "tx_hash": result.tx_hash,
            "error_message": result.error_message,
            "detection_latency_ms": result.detection_latency_ms,
            "execution_latency_ms": result.execution_latency_ms,
        }

        # Calculate total latency
        if result.detection_latency_ms and result.execution_latency_ms:
            updates["total_latency_ms"] = (
                result.detection_latency_ms + result.execution_latency_ms
            )

        self._trade_repo.update_status(trade_id, result.status, **updates)

        # Update daily stats
        if result.status == TradeStatus.FILLED:
            self._stats_repo.increment_trades_executed()

        logger.info(
            "trade_logged_result",
            trade_id=trade_id,
            status=result.status.value,
            success=result.success,
        )

    def log_skipped(
        self,
        trade_id: int,
        reason: TradeStatus,
        message: str,
    ) -> None:
        """Log trade skipped.

        Args:
            trade_id: Trade log ID
            reason: Skip reason (SKIPPED_*)
            message: Skip message
        """
        self._trade_repo.update_status(
            trade_id,
            reason,
            error_message=message,
        )

        logger.info(
            "trade_logged_skipped",
            trade_id=trade_id,
            reason=reason.value,
            message=message,
        )

    def log_failed(
        self,
        trade_id: int,
        error: str,
        retry_count: int = 0,
    ) -> None:
        """Log trade failure.

        Args:
            trade_id: Trade log ID
            error: Error message
            retry_count: Number of retries
        """
        self._trade_repo.update_status(
            trade_id,
            TradeStatus.FAILED,
            error_message=error,
            retry_count=retry_count,
        )

        # Log system event for failures
        self._event_repo.log_event(
            event_type="trade_failed",
            severity=EventSeverity.ERROR,
            message=f"Trade {trade_id} failed: {error}",
            metadata={"trade_id": trade_id, "retry_count": retry_count},
            source="trade_logger",
        )

    def get_trade_summary(self, days: int = 1) -> dict:
        """Get trade summary for a period.

        Args:
            days: Number of days

        Returns:
            Summary dictionary
        """
        start = datetime.utcnow() - timedelta(days=days)
        return self._trade_repo.get_trade_stats(start)


class PositionPersistence:
    """Position persistence service.

    Manages position state synchronization with database.

    Example:
        persistence = PositionPersistence(db)

        # Sync positions from in-memory state
        persistence.sync_positions(account_id, positions)

        # Load on restart
        positions = persistence.load_positions(account_id)
    """

    def __init__(self, db: DatabaseManager):
        """Initialize position persistence.

        Args:
            db: Database manager
        """
        self.db = db
        self._position_repo = PositionRepository(db)

    def sync_position(
        self,
        account_id: int,
        token_id: str,
        market_id: str,
        condition_id: str,
        outcome: str,
        our_size: Decimal,
        target_size: Decimal,
        average_price: Decimal,
        current_price: Optional[Decimal] = None,
        total_cost: Optional[Decimal] = None,
    ) -> Position:
        """Sync a single position to database.

        Args:
            account_id: Account ID
            token_id: Token ID
            market_id: Market ID
            condition_id: Condition ID
            outcome: Outcome name
            our_size: Our position size
            target_size: Target's position size
            average_price: Average entry price
            current_price: Current market price
            total_cost: Total cost basis

        Returns:
            Updated position
        """
        # Calculate unrealized P&L if we have current price
        unrealized_pnl = None
        if current_price and our_size > 0:
            unrealized_pnl = (current_price - average_price) * our_size

        # Calculate drift
        drift_percent = None
        status = PositionStatus.SYNCED

        if target_size > 0:
            expected_size = target_size  # Ratio applied elsewhere
            if our_size > 0:
                drift_percent = abs(our_size - expected_size) / expected_size
                if drift_percent > Decimal("0.1"):
                    status = PositionStatus.DRIFT
                elif drift_percent > Decimal("0.05"):
                    status = PositionStatus.PENDING

        position = self._position_repo.upsert_position(
            account_id=account_id,
            token_id=token_id,
            market_id=market_id,
            condition_id=condition_id,
            outcome=outcome,
            our_size=our_size,
            target_size=target_size,
            average_price=average_price,
            current_price=current_price,
            total_cost=total_cost or (our_size * average_price),
            unrealized_pnl=unrealized_pnl,
            drift_percent=drift_percent,
            status=status,
        )

        return position

    def close_position(self, account_id: int, token_id: str) -> bool:
        """Mark a position as closed.

        Args:
            account_id: Account ID
            token_id: Token ID

        Returns:
            True if closed
        """
        return self._position_repo.close_position(account_id, token_id)

    def load_positions(self, account_id: int) -> List[Position]:
        """Load all open positions for an account.

        Args:
            account_id: Account ID

        Returns:
            List of positions
        """
        return self._position_repo.get_open_positions(account_id)

    def load_all_positions(self) -> List[Position]:
        """Load all open positions.

        Returns:
            List of all open positions
        """
        return self._position_repo.get_open_positions()

    def get_positions_needing_attention(self) -> List[Position]:
        """Get positions with drift above threshold.

        Returns:
            List of drifted positions
        """
        return self._position_repo.get_positions_with_drift()

    def get_total_value(self, account_id: Optional[int] = None) -> Decimal:
        """Get total position value.

        Args:
            account_id: Optional account filter

        Returns:
            Total value
        """
        return self._position_repo.get_total_value(account_id)


@dataclass
class RecoveryState:
    """State recovered from database."""

    accounts: List[TrackedAccount]
    positions: List[Position]
    last_trade_timestamp: Optional[datetime]
    balance_snapshot: Optional[dict]
    pending_trades: List[TradeLog]


class RecoveryService:
    """Service for recovering state on restart.

    Loads all necessary state from database to resume operation.

    Example:
        recovery = RecoveryService(db)
        state = recovery.recover()

        # Use recovered state to initialize components
        for account in state.accounts:
            monitor.add_target(account)
    """

    def __init__(self, db: DatabaseManager):
        """Initialize recovery service.

        Args:
            db: Database manager
        """
        self.db = db
        self._account_repo = AccountRepository(db)
        self._position_repo = PositionRepository(db)
        self._trade_repo = TradeLogRepository(db)
        self._balance_repo = BalanceSnapshotRepository(db)
        self._event_repo = SystemEventRepository(db)

    def recover(self) -> RecoveryState:
        """Recover full state from database.

        Returns:
            RecoveryState with all loaded data
        """
        logger.info("starting_recovery")

        # Load enabled accounts
        accounts = self._account_repo.get_enabled()
        logger.info(
            "recovered_accounts",
            count=len(accounts),
        )

        # Load open positions
        positions = self._position_repo.get_open_positions()
        logger.info(
            "recovered_positions",
            count=len(positions),
        )

        # Find last trade timestamp for each account
        last_timestamp = None
        recent_trades = self._trade_repo.get_recent_trades(limit=1)
        if recent_trades:
            last_timestamp = recent_trades[0].detected_at

        # Load latest balance snapshot
        balance_snapshot = None
        latest_balance = self._balance_repo.get_latest()
        if latest_balance:
            balance_snapshot = latest_balance.to_dict()

        # Find any pending/executing trades to handle
        pending_trades = []
        for status in [TradeStatus.QUEUED, TradeStatus.EXECUTING]:
            trades = self._trade_repo.get_recent_trades(status=status, limit=100)
            pending_trades.extend(trades)

        if pending_trades:
            logger.warning(
                "found_incomplete_trades",
                count=len(pending_trades),
            )

        # Log recovery event
        self._event_repo.log_event(
            event_type="system_recovery",
            severity=EventSeverity.INFO,
            message="System recovered from restart",
            metadata={
                "accounts": len(accounts),
                "positions": len(positions),
                "pending_trades": len(pending_trades),
            },
            source="recovery_service",
        )

        return RecoveryState(
            accounts=accounts,
            positions=positions,
            last_trade_timestamp=last_timestamp,
            balance_snapshot=balance_snapshot,
            pending_trades=pending_trades,
        )

    def mark_incomplete_trades_failed(self, reason: str = "System restart") -> int:
        """Mark incomplete trades as failed.

        Args:
            reason: Failure reason

        Returns:
            Number of trades marked
        """
        count = 0

        for status in [TradeStatus.QUEUED, TradeStatus.EXECUTING]:
            trades = self._trade_repo.get_recent_trades(status=status, limit=1000)

            for trade in trades:
                self._trade_repo.update_status(
                    trade.id,
                    TradeStatus.FAILED,
                    error_message=reason,
                )
                count += 1

        if count > 0:
            logger.warning(
                "marked_incomplete_trades_failed",
                count=count,
                reason=reason,
            )

        return count

    def get_last_activity_timestamp(self, account_id: int) -> Optional[datetime]:
        """Get last activity timestamp for an account.

        Args:
            account_id: Account ID

        Returns:
            Last activity timestamp or None
        """
        trades = self._trade_repo.get_recent_trades(
            account_id=account_id,
            limit=1,
        )

        if trades:
            return trades[0].detected_at

        return None


class AuditService:
    """Service for audit trail queries.

    Provides methods to query and export audit data.

    Example:
        audit = AuditService(db)

        # Get trade history
        trades = audit.get_trade_history(days=7)

        # Get system events
        events = audit.get_events(severity="error")

        # Export audit data
        data = audit.export_audit_data(start, end)
    """

    def __init__(self, db: DatabaseManager):
        """Initialize audit service.

        Args:
            db: Database manager
        """
        self.db = db
        self._trade_repo = TradeLogRepository(db)
        self._event_repo = SystemEventRepository(db)
        self._cb_repo = CircuitBreakerEventRepository(db)
        self._stats_repo = DailyStatsRepository(db)

    def get_trade_history(
        self,
        days: int = 7,
        account_id: Optional[int] = None,
        status: Optional[TradeStatus] = None,
    ) -> List[dict]:
        """Get trade history.

        Args:
            days: Number of days
            account_id: Optional account filter
            status: Optional status filter

        Returns:
            List of trade records
        """
        start = datetime.utcnow() - timedelta(days=days)
        trades = self._trade_repo.get_trades_in_period(
            start=start,
            account_id=account_id,
        )

        if status:
            trades = [t for t in trades if t.status == status]

        return [t.to_dict() for t in trades]

    def get_events(
        self,
        days: int = 7,
        severity: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[dict]:
        """Get system events.

        Args:
            days: Number of days
            severity: Optional severity filter
            event_type: Optional type filter

        Returns:
            List of event records
        """
        sev = EventSeverity(severity) if severity else None
        events = self._event_repo.get_recent_events(
            limit=1000,
            severity=sev,
            event_type=event_type,
        )

        cutoff = datetime.utcnow() - timedelta(days=days)
        events = [e for e in events if e.created_at >= cutoff]

        return [e.to_dict() for e in events]

    def get_circuit_breaker_history(self, limit: int = 50) -> List[dict]:
        """Get circuit breaker event history.

        Args:
            limit: Maximum results

        Returns:
            List of circuit breaker events
        """
        events = self._cb_repo.get_recent_trips(limit=limit)
        return [e.to_dict() for e in events]

    def get_daily_stats(self, days: int = 30) -> List[dict]:
        """Get daily statistics.

        Args:
            days: Number of days

        Returns:
            List of daily stats
        """
        stats = self._stats_repo.get_history(days)
        return [s.to_dict() for s in stats]

    def export_audit_data(
        self,
        start: datetime,
        end: datetime,
    ) -> dict:
        """Export full audit data for a period.

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            Complete audit data
        """
        trades = self._trade_repo.get_trades_in_period(start, end)

        events = self._event_repo.get_recent_events(limit=10000)
        events = [e for e in events if start <= e.created_at <= end]

        return {
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "trades": [t.to_dict() for t in trades],
            "events": [e.to_dict() for e in events],
            "statistics": self._trade_repo.get_trade_stats(start, end),
        }

    def get_performance_summary(self, days: int = 30) -> dict:
        """Get performance summary.

        Args:
            days: Number of days

        Returns:
            Performance summary
        """
        stats = self._stats_repo.get_history(days)

        total_detected = sum(s.trades_detected for s in stats)
        total_executed = sum(s.trades_executed for s in stats)
        total_failed = sum(s.trades_failed for s in stats)
        total_volume = sum(s.total_volume_usd for s in stats)
        total_pnl = sum(s.realized_pnl for s in stats)

        return {
            "period_days": days,
            "total_trades_detected": total_detected,
            "total_trades_executed": total_executed,
            "total_trades_failed": total_failed,
            "execution_rate": total_executed / total_detected if total_detected > 0 else 0,
            "total_volume_usd": str(total_volume),
            "total_realized_pnl": str(total_pnl),
            "daily_breakdown": [s.to_dict() for s in stats[-7:]],  # Last 7 days
        }
