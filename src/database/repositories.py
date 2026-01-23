"""Repository pattern for database operations.

Provides clean interfaces for:
- TrackedAccount CRUD
- Position management
- Trade logging
- System events
- Statistics and queries
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Generic, TypeVar, Type, Any

from sqlalchemy import and_, or_, func, desc
from sqlalchemy.orm import Session

from ..utils.logging import get_logger
from .models import (
    Base,
    TrackedAccount,
    Position,
    TradeLog,
    SystemEvent,
    BalanceSnapshot,
    CircuitBreakerEvent,
    DailyStats,
    PositionStatus,
    TradeStatus,
    EventSeverity,
)
from .connection import DatabaseManager

logger = get_logger(__name__)

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T], ABC):
    """Base repository with common CRUD operations."""

    def __init__(self, db: DatabaseManager, model: Type[T]):
        """Initialize repository.

        Args:
            db: Database manager
            model: SQLAlchemy model class
        """
        self.db = db
        self.model = model

    def get_by_id(self, id: int) -> Optional[T]:
        """Get entity by ID.

        Args:
            id: Entity ID

        Returns:
            Entity or None
        """
        with self.db.session() as session:
            return session.query(self.model).filter(self.model.id == id).first()

    def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination.

        Args:
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of entities
        """
        with self.db.session() as session:
            return (
                session.query(self.model)
                .limit(limit)
                .offset(offset)
                .all()
            )

    def create(self, **kwargs) -> T:
        """Create a new entity.

        Args:
            **kwargs: Entity attributes

        Returns:
            Created entity
        """
        with self.db.session() as session:
            entity = self.model(**kwargs)
            session.add(entity)
            session.flush()
            session.refresh(entity)
            return entity

    def update(self, id: int, **kwargs) -> Optional[T]:
        """Update an entity.

        Args:
            id: Entity ID
            **kwargs: Attributes to update

        Returns:
            Updated entity or None
        """
        with self.db.session() as session:
            entity = session.query(self.model).filter(self.model.id == id).first()
            if entity:
                for key, value in kwargs.items():
                    setattr(entity, key, value)
                session.flush()
                session.refresh(entity)
            return entity

    def delete(self, id: int) -> bool:
        """Delete an entity.

        Args:
            id: Entity ID

        Returns:
            True if deleted
        """
        with self.db.session() as session:
            result = session.query(self.model).filter(self.model.id == id).delete()
            return result > 0

    def count(self) -> int:
        """Count all entities.

        Returns:
            Total count
        """
        with self.db.session() as session:
            return session.query(self.model).count()


class AccountRepository(BaseRepository[TrackedAccount]):
    """Repository for tracked accounts."""

    def __init__(self, db: DatabaseManager):
        super().__init__(db, TrackedAccount)

    def get_by_name(self, name: str) -> Optional[TrackedAccount]:
        """Get account by name.

        Args:
            name: Account name

        Returns:
            Account or None
        """
        with self.db.session() as session:
            return (
                session.query(TrackedAccount)
                .filter(TrackedAccount.name == name)
                .first()
            )

    def get_by_wallet(self, wallet: str) -> Optional[TrackedAccount]:
        """Get account by wallet address.

        Args:
            wallet: Wallet address

        Returns:
            Account or None
        """
        with self.db.session() as session:
            return (
                session.query(TrackedAccount)
                .filter(TrackedAccount.target_wallet == wallet.lower())
                .first()
            )

    def get_enabled(self) -> List[TrackedAccount]:
        """Get all enabled accounts.

        Returns:
            List of enabled accounts
        """
        with self.db.session() as session:
            return (
                session.query(TrackedAccount)
                .filter(TrackedAccount.enabled == True)
                .all()
            )

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable an account.

        Args:
            name: Account name
            enabled: Enable status

        Returns:
            True if updated
        """
        with self.db.session() as session:
            result = (
                session.query(TrackedAccount)
                .filter(TrackedAccount.name == name)
                .update({TrackedAccount.enabled: enabled})
            )
            return result > 0


class PositionRepository(BaseRepository[Position]):
    """Repository for positions."""

    def __init__(self, db: DatabaseManager):
        super().__init__(db, Position)

    def get_by_token(self, account_id: int, token_id: str) -> Optional[Position]:
        """Get position by token ID.

        Args:
            account_id: Account ID
            token_id: Token ID

        Returns:
            Position or None
        """
        with self.db.session() as session:
            return (
                session.query(Position)
                .filter(
                    and_(
                        Position.account_id == account_id,
                        Position.token_id == token_id,
                    )
                )
                .first()
            )

    def get_open_positions(self, account_id: Optional[int] = None) -> List[Position]:
        """Get all open positions.

        Args:
            account_id: Optional account filter

        Returns:
            List of open positions
        """
        with self.db.session() as session:
            query = session.query(Position).filter(
                Position.status != PositionStatus.CLOSED
            )

            if account_id is not None:
                query = query.filter(Position.account_id == account_id)

            return query.all()

    def get_positions_with_drift(
        self,
        min_drift: Decimal = Decimal("0.05"),
    ) -> List[Position]:
        """Get positions with drift above threshold.

        Args:
            min_drift: Minimum drift percentage

        Returns:
            List of drifted positions
        """
        with self.db.session() as session:
            return (
                session.query(Position)
                .filter(
                    and_(
                        Position.status == PositionStatus.DRIFT,
                        Position.drift_percent >= min_drift,
                    )
                )
                .all()
            )

    def upsert_position(
        self,
        account_id: int,
        token_id: str,
        **kwargs,
    ) -> Position:
        """Create or update a position.

        Args:
            account_id: Account ID
            token_id: Token ID
            **kwargs: Position attributes

        Returns:
            Position
        """
        with self.db.session() as session:
            position = (
                session.query(Position)
                .filter(
                    and_(
                        Position.account_id == account_id,
                        Position.token_id == token_id,
                    )
                )
                .first()
            )

            if position:
                for key, value in kwargs.items():
                    setattr(position, key, value)
            else:
                position = Position(
                    account_id=account_id,
                    token_id=token_id,
                    **kwargs,
                )
                session.add(position)

            session.flush()
            session.refresh(position)
            return position

    def close_position(self, account_id: int, token_id: str) -> bool:
        """Mark a position as closed.

        Args:
            account_id: Account ID
            token_id: Token ID

        Returns:
            True if closed
        """
        with self.db.session() as session:
            result = (
                session.query(Position)
                .filter(
                    and_(
                        Position.account_id == account_id,
                        Position.token_id == token_id,
                    )
                )
                .update({
                    Position.status: PositionStatus.CLOSED,
                    Position.closed_at: datetime.utcnow(),
                    Position.our_size: Decimal("0"),
                })
            )
            return result > 0

    def get_total_value(self, account_id: Optional[int] = None) -> Decimal:
        """Get total position value.

        Args:
            account_id: Optional account filter

        Returns:
            Total value
        """
        with self.db.session() as session:
            query = session.query(
                func.sum(Position.our_size * Position.current_price)
            ).filter(Position.status != PositionStatus.CLOSED)

            if account_id is not None:
                query = query.filter(Position.account_id == account_id)

            result = query.scalar()
            return Decimal(str(result)) if result else Decimal("0")


class TradeLogRepository(BaseRepository[TradeLog]):
    """Repository for trade logs."""

    def __init__(self, db: DatabaseManager):
        super().__init__(db, TradeLog)

    def log_trade(
        self,
        account_id: int,
        market_id: str,
        condition_id: str,
        token_id: str,
        outcome: str,
        side: str,
        target_price: Decimal,
        target_size: Decimal,
        detected_at: datetime,
        **kwargs,
    ) -> TradeLog:
        """Log a new trade.

        Args:
            account_id: Account ID
            market_id: Market ID
            condition_id: Condition ID
            token_id: Token ID
            outcome: Outcome name
            side: BUY or SELL
            target_price: Target's execution price
            target_size: Target's trade size
            detected_at: When trade was detected
            **kwargs: Additional attributes

        Returns:
            Created trade log
        """
        return self.create(
            account_id=account_id,
            market_id=market_id,
            condition_id=condition_id,
            token_id=token_id,
            outcome=outcome,
            side=side,
            target_price=target_price,
            target_size=target_size,
            detected_at=detected_at,
            status=TradeStatus.DETECTED,
            **kwargs,
        )

    def update_status(
        self,
        trade_id: int,
        status: TradeStatus,
        **kwargs,
    ) -> Optional[TradeLog]:
        """Update trade status.

        Args:
            trade_id: Trade ID
            status: New status
            **kwargs: Additional updates

        Returns:
            Updated trade or None
        """
        updates = {"status": status, **kwargs}

        if status == TradeStatus.FILLED:
            updates["executed_at"] = datetime.utcnow()

        return self.update(trade_id, **updates)

    def get_recent_trades(
        self,
        account_id: Optional[int] = None,
        limit: int = 50,
        status: Optional[TradeStatus] = None,
    ) -> List[TradeLog]:
        """Get recent trades.

        Args:
            account_id: Optional account filter
            limit: Maximum results
            status: Optional status filter

        Returns:
            List of trades
        """
        with self.db.session() as session:
            query = session.query(TradeLog)

            if account_id is not None:
                query = query.filter(TradeLog.account_id == account_id)

            if status is not None:
                query = query.filter(TradeLog.status == status)

            return (
                query.order_by(desc(TradeLog.created_at))
                .limit(limit)
                .all()
            )

    def get_trades_in_period(
        self,
        start: datetime,
        end: Optional[datetime] = None,
        account_id: Optional[int] = None,
    ) -> List[TradeLog]:
        """Get trades in a time period.

        Args:
            start: Start datetime
            end: End datetime (default: now)
            account_id: Optional account filter

        Returns:
            List of trades
        """
        if end is None:
            end = datetime.utcnow()

        with self.db.session() as session:
            query = session.query(TradeLog).filter(
                and_(
                    TradeLog.created_at >= start,
                    TradeLog.created_at <= end,
                )
            )

            if account_id is not None:
                query = query.filter(TradeLog.account_id == account_id)

            return query.order_by(TradeLog.created_at).all()

    def get_trade_stats(
        self,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> dict:
        """Get trade statistics for a period.

        Args:
            start: Start datetime
            end: End datetime (default: now)

        Returns:
            Statistics dictionary
        """
        if end is None:
            end = datetime.utcnow()

        with self.db.session() as session:
            trades = (
                session.query(TradeLog)
                .filter(
                    and_(
                        TradeLog.created_at >= start,
                        TradeLog.created_at <= end,
                    )
                )
                .all()
            )

            total = len(trades)
            filled = sum(1 for t in trades if t.status == TradeStatus.FILLED)
            failed = sum(1 for t in trades if t.status == TradeStatus.FAILED)
            skipped = sum(1 for t in trades if t.status.value.startswith("skipped"))

            avg_slippage = None
            slippages = [t.slippage_percent for t in trades if t.slippage_percent]
            if slippages:
                avg_slippage = sum(slippages) / len(slippages)

            avg_latency = None
            latencies = [t.total_latency_ms for t in trades if t.total_latency_ms]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)

            return {
                "total": total,
                "filled": filled,
                "failed": failed,
                "skipped": skipped,
                "fill_rate": filled / total if total > 0 else 0,
                "avg_slippage_percent": float(avg_slippage) if avg_slippage else None,
                "avg_latency_ms": avg_latency,
            }


class SystemEventRepository(BaseRepository[SystemEvent]):
    """Repository for system events."""

    def __init__(self, db: DatabaseManager):
        super().__init__(db, SystemEvent)

    def log_event(
        self,
        event_type: str,
        severity: EventSeverity,
        message: str,
        metadata: Optional[dict] = None,
        source: Optional[str] = None,
    ) -> SystemEvent:
        """Log a system event.

        Args:
            event_type: Event type
            severity: Severity level
            message: Event message
            metadata: Additional metadata
            source: Event source

        Returns:
            Created event
        """
        return self.create(
            event_type=event_type,
            severity=severity,
            message=message,
            metadata=metadata,
            source=source,
        )

    def get_recent_events(
        self,
        limit: int = 100,
        severity: Optional[EventSeverity] = None,
        event_type: Optional[str] = None,
    ) -> List[SystemEvent]:
        """Get recent events.

        Args:
            limit: Maximum results
            severity: Optional severity filter
            event_type: Optional type filter

        Returns:
            List of events
        """
        with self.db.session() as session:
            query = session.query(SystemEvent)

            if severity is not None:
                query = query.filter(SystemEvent.severity == severity)

            if event_type is not None:
                query = query.filter(SystemEvent.event_type == event_type)

            return (
                query.order_by(desc(SystemEvent.created_at))
                .limit(limit)
                .all()
            )

    def get_errors_in_period(
        self,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> List[SystemEvent]:
        """Get error events in a period.

        Args:
            start: Start datetime
            end: End datetime (default: now)

        Returns:
            List of error events
        """
        if end is None:
            end = datetime.utcnow()

        with self.db.session() as session:
            return (
                session.query(SystemEvent)
                .filter(
                    and_(
                        SystemEvent.created_at >= start,
                        SystemEvent.created_at <= end,
                        SystemEvent.severity.in_([
                            EventSeverity.ERROR,
                            EventSeverity.CRITICAL,
                        ]),
                    )
                )
                .order_by(SystemEvent.created_at)
                .all()
            )


class BalanceSnapshotRepository(BaseRepository[BalanceSnapshot]):
    """Repository for balance snapshots."""

    def __init__(self, db: DatabaseManager):
        super().__init__(db, BalanceSnapshot)

    def take_snapshot(
        self,
        total_balance: Decimal,
        available_balance: Decimal,
        **kwargs,
    ) -> BalanceSnapshot:
        """Take a balance snapshot.

        Args:
            total_balance: Total balance
            available_balance: Available balance
            **kwargs: Additional attributes

        Returns:
            Created snapshot
        """
        return self.create(
            total_balance=total_balance,
            available_balance=available_balance,
            **kwargs,
        )

    def get_latest(self) -> Optional[BalanceSnapshot]:
        """Get the latest snapshot.

        Returns:
            Latest snapshot or None
        """
        with self.db.session() as session:
            return (
                session.query(BalanceSnapshot)
                .order_by(desc(BalanceSnapshot.created_at))
                .first()
            )

    def get_history(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List[BalanceSnapshot]:
        """Get snapshot history.

        Args:
            hours: Hours of history
            limit: Maximum results

        Returns:
            List of snapshots
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self.db.session() as session:
            return (
                session.query(BalanceSnapshot)
                .filter(BalanceSnapshot.created_at >= cutoff)
                .order_by(BalanceSnapshot.created_at)
                .limit(limit)
                .all()
            )


class CircuitBreakerEventRepository(BaseRepository[CircuitBreakerEvent]):
    """Repository for circuit breaker events."""

    def __init__(self, db: DatabaseManager):
        super().__init__(db, CircuitBreakerEvent)

    def log_trip(
        self,
        reason: str,
        details: Optional[str] = None,
        state_snapshot: Optional[dict] = None,
        auto_reset: bool = False,
        reset_after_s: Optional[int] = None,
    ) -> CircuitBreakerEvent:
        """Log a circuit breaker trip.

        Args:
            reason: Trip reason
            details: Trip details
            state_snapshot: State at time of trip
            auto_reset: Whether auto-reset is enabled
            reset_after_s: Seconds until auto-reset

        Returns:
            Created event
        """
        return self.create(
            event_type="trip",
            reason=reason,
            details=details,
            state_snapshot=state_snapshot,
            auto_reset=auto_reset,
            reset_after_s=reset_after_s,
        )

    def log_reset(self, reason: str = "manual") -> CircuitBreakerEvent:
        """Log a circuit breaker reset.

        Args:
            reason: Reset reason

        Returns:
            Created event
        """
        return self.create(
            event_type="reset",
            reason=reason,
        )

    def get_recent_trips(self, limit: int = 10) -> List[CircuitBreakerEvent]:
        """Get recent circuit breaker trips.

        Args:
            limit: Maximum results

        Returns:
            List of trip events
        """
        with self.db.session() as session:
            return (
                session.query(CircuitBreakerEvent)
                .filter(CircuitBreakerEvent.event_type == "trip")
                .order_by(desc(CircuitBreakerEvent.created_at))
                .limit(limit)
                .all()
            )


class DailyStatsRepository(BaseRepository[DailyStats]):
    """Repository for daily statistics."""

    def __init__(self, db: DatabaseManager):
        super().__init__(db, DailyStats)

    def get_or_create_today(self) -> DailyStats:
        """Get or create today's stats record.

        Returns:
            Today's stats
        """
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        with self.db.session() as session:
            stats = (
                session.query(DailyStats)
                .filter(DailyStats.date == today)
                .first()
            )

            if not stats:
                stats = DailyStats(date=today)
                session.add(stats)
                session.flush()
                session.refresh(stats)

            return stats

    def increment_trades_detected(self) -> None:
        """Increment today's trades detected count."""
        with self.db.session() as session:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            session.query(DailyStats).filter(
                DailyStats.date == today
            ).update({
                DailyStats.trades_detected: DailyStats.trades_detected + 1
            })

    def increment_trades_executed(self) -> None:
        """Increment today's trades executed count."""
        with self.db.session() as session:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            session.query(DailyStats).filter(
                DailyStats.date == today
            ).update({
                DailyStats.trades_executed: DailyStats.trades_executed + 1
            })

    def get_history(self, days: int = 30) -> List[DailyStats]:
        """Get daily stats history.

        Args:
            days: Number of days

        Returns:
            List of daily stats
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        with self.db.session() as session:
            return (
                session.query(DailyStats)
                .filter(DailyStats.date >= cutoff)
                .order_by(DailyStats.date)
                .all()
            )
