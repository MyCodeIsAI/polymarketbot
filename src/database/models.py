"""SQLAlchemy database models for PolymarketBot.

Defines all database tables for:
- Tracked accounts configuration
- Position tracking
- Trade execution logs
- System events
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Any

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Numeric,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
    JSON,
    Enum as SQLEnum,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class PositionStatus(str, Enum):
    """Status of a tracked position."""

    SYNCED = "synced"
    PENDING = "pending"
    DRIFT = "drift"
    CLOSED = "closed"


class TradeStatus(str, Enum):
    """Status of a trade execution."""

    DETECTED = "detected"
    QUEUED = "queued"
    EXECUTING = "executing"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SKIPPED_SLIPPAGE = "skipped_slippage"
    SKIPPED_BALANCE = "skipped_balance"
    SKIPPED_SIZE = "skipped_size"


class EventSeverity(str, Enum):
    """Severity level for system events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TrackedAccount(Base):
    """Configuration for a tracked target account."""

    __tablename__ = "tracked_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    target_wallet = Column(String(42), nullable=False)
    position_ratio = Column(Numeric(10, 6), nullable=False)
    max_position_usd = Column(Numeric(20, 2), nullable=False)
    slippage_tolerance = Column(Numeric(5, 4), nullable=False)
    min_position_usd = Column(Numeric(20, 2), default=Decimal("5"))
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    positions = relationship("Position", back_populates="account", lazy="dynamic")
    trades = relationship("TradeLog", back_populates="account", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<TrackedAccount(name='{self.name}', wallet='{self.target_wallet[:10]}...')>"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "target_wallet": self.target_wallet,
            "position_ratio": str(self.position_ratio),
            "max_position_usd": str(self.max_position_usd),
            "slippage_tolerance": str(self.slippage_tolerance),
            "min_position_usd": str(self.min_position_usd),
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Position(Base):
    """Tracked position for copy-trading."""

    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("account_id", "token_id", name="uq_account_token"),
        Index("idx_positions_account_status", "account_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("tracked_accounts.id"), nullable=False)
    market_id = Column(String(66), nullable=False)
    condition_id = Column(String(66), nullable=False)
    token_id = Column(String(66), nullable=False)
    outcome = Column(String(20), nullable=False)

    # Target position
    target_size = Column(Numeric(30, 18), nullable=False, default=Decimal("0"))

    # Our position
    our_size = Column(Numeric(30, 18), nullable=False, default=Decimal("0"))
    average_price = Column(Numeric(20, 18), nullable=False, default=Decimal("0"))
    total_cost = Column(Numeric(20, 6), nullable=False, default=Decimal("0"))

    # Current state
    current_price = Column(Numeric(20, 18), nullable=True)
    unrealized_pnl = Column(Numeric(20, 6), nullable=True)

    status = Column(SQLEnum(PositionStatus), default=PositionStatus.SYNCED)
    drift_percent = Column(Numeric(10, 6), nullable=True)

    # Timestamps
    opened_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    closed_at = Column(DateTime, nullable=True)

    # Relationship
    account = relationship("TrackedAccount", back_populates="positions")

    def __repr__(self) -> str:
        return f"<Position(token='{self.token_id[:16]}...', size={self.our_size})>"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "account_id": self.account_id,
            "market_id": self.market_id,
            "token_id": self.token_id,
            "outcome": self.outcome,
            "target_size": str(self.target_size),
            "our_size": str(self.our_size),
            "average_price": str(self.average_price),
            "total_cost": str(self.total_cost),
            "current_price": str(self.current_price) if self.current_price else None,
            "unrealized_pnl": str(self.unrealized_pnl) if self.unrealized_pnl else None,
            "status": self.status.value,
            "drift_percent": str(self.drift_percent) if self.drift_percent else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TradeLog(Base):
    """Audit log for trade execution attempts."""

    __tablename__ = "trade_log"
    __table_args__ = (
        Index("idx_trade_log_account_created", "account_id", "created_at"),
        Index("idx_trade_log_status", "status"),
        Index("idx_trade_log_market", "market_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("tracked_accounts.id"), nullable=False)

    # Market info
    market_id = Column(String(66), nullable=False)
    condition_id = Column(String(66), nullable=False)
    token_id = Column(String(66), nullable=False)
    outcome = Column(String(20), nullable=False)
    side = Column(String(4), nullable=False)  # BUY or SELL

    # Target trade info
    target_price = Column(Numeric(20, 18), nullable=False)
    target_size = Column(Numeric(30, 18), nullable=False)

    # Our execution info
    execution_price = Column(Numeric(20, 18), nullable=True)
    execution_size = Column(Numeric(30, 18), nullable=True)
    execution_cost = Column(Numeric(20, 6), nullable=True)

    # Slippage info
    slippage_percent = Column(Numeric(10, 6), nullable=True)
    slippage_allowed = Column(Numeric(10, 6), nullable=True)

    # Order info
    order_id = Column(String(100), nullable=True)
    tx_hash = Column(String(66), nullable=True)

    # Status and errors
    status = Column(SQLEnum(TradeStatus), nullable=False, default=TradeStatus.DETECTED)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Timing
    detected_at = Column(DateTime, nullable=False)
    queued_at = Column(DateTime, nullable=True)
    executed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())

    # Latency metrics (milliseconds)
    detection_latency_ms = Column(Integer, nullable=True)
    execution_latency_ms = Column(Integer, nullable=True)
    total_latency_ms = Column(Integer, nullable=True)

    # Relationship
    account = relationship("TrackedAccount", back_populates="trades")

    def __repr__(self) -> str:
        return f"<TradeLog(id={self.id}, side='{self.side}', status='{self.status.value}')>"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "account_id": self.account_id,
            "market_id": self.market_id,
            "token_id": self.token_id,
            "outcome": self.outcome,
            "side": self.side,
            "target_price": str(self.target_price),
            "target_size": str(self.target_size),
            "execution_price": str(self.execution_price) if self.execution_price else None,
            "execution_size": str(self.execution_size) if self.execution_size else None,
            "slippage_percent": str(self.slippage_percent) if self.slippage_percent else None,
            "order_id": self.order_id,
            "tx_hash": self.tx_hash,
            "status": self.status.value,
            "error_message": self.error_message,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "detection_latency_ms": self.detection_latency_ms,
            "execution_latency_ms": self.execution_latency_ms,
        }


class SystemEvent(Base):
    """System event log for audit trail."""

    __tablename__ = "system_events"
    __table_args__ = (
        Index("idx_system_events_type_created", "event_type", "created_at"),
        Index("idx_system_events_severity", "severity"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    severity = Column(SQLEnum(EventSeverity), nullable=False)
    message = Column(Text, nullable=False)
    event_metadata = Column(JSON, nullable=True)
    source = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self) -> str:
        return f"<SystemEvent(type='{self.event_type}', severity='{self.severity.value}')>"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity.value,
            "message": self.message,
            "metadata": self.event_metadata,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BalanceSnapshot(Base):
    """Periodic snapshots of account balance."""

    __tablename__ = "balance_snapshots"
    __table_args__ = (
        Index("idx_balance_snapshots_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Balance state
    total_balance = Column(Numeric(20, 6), nullable=False)
    available_balance = Column(Numeric(20, 6), nullable=False)
    reserved_balance = Column(Numeric(20, 6), nullable=False, default=Decimal("0"))

    # Position value
    total_position_value = Column(Numeric(20, 6), nullable=False, default=Decimal("0"))
    open_positions = Column(Integer, default=0)

    # P&L state
    realized_pnl = Column(Numeric(20, 6), nullable=False, default=Decimal("0"))
    unrealized_pnl = Column(Numeric(20, 6), nullable=False, default=Decimal("0"))

    # Statistics
    trades_today = Column(Integer, default=0)
    daily_pnl = Column(Numeric(20, 6), nullable=False, default=Decimal("0"))

    created_at = Column(DateTime, default=func.now())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "total_balance": str(self.total_balance),
            "available_balance": str(self.available_balance),
            "reserved_balance": str(self.reserved_balance),
            "total_position_value": str(self.total_position_value),
            "open_positions": self.open_positions,
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "trades_today": self.trades_today,
            "daily_pnl": str(self.daily_pnl),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CircuitBreakerEvent(Base):
    """Record of circuit breaker trips and resets."""

    __tablename__ = "circuit_breaker_events"
    __table_args__ = (
        Index("idx_cb_events_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(20), nullable=False)  # trip, reset, force_reset
    reason = Column(String(50), nullable=False)
    details = Column(Text, nullable=True)

    # State snapshot at time of event
    state_snapshot = Column(JSON, nullable=True)

    # Recovery info
    auto_reset = Column(Boolean, default=False)
    reset_after_s = Column(Integer, nullable=True)
    recovered_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=func.now())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "reason": self.reason,
            "details": self.details,
            "state_snapshot": self.state_snapshot,
            "auto_reset": self.auto_reset,
            "recovered_at": self.recovered_at.isoformat() if self.recovered_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DailyStats(Base):
    """Daily aggregated statistics."""

    __tablename__ = "daily_stats"
    __table_args__ = (
        UniqueConstraint("date", name="uq_daily_stats_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)

    # Trade statistics
    trades_detected = Column(Integer, default=0)
    trades_executed = Column(Integer, default=0)
    trades_skipped = Column(Integer, default=0)
    trades_failed = Column(Integer, default=0)

    # Volume
    total_volume_usd = Column(Numeric(20, 6), default=Decimal("0"))

    # P&L
    realized_pnl = Column(Numeric(20, 6), default=Decimal("0"))
    ending_balance = Column(Numeric(20, 6), nullable=True)

    # Performance
    avg_detection_latency_ms = Column(Integer, nullable=True)
    avg_execution_latency_ms = Column(Integer, nullable=True)
    avg_slippage_percent = Column(Numeric(10, 6), nullable=True)

    # System health
    api_errors = Column(Integer, default=0)
    ws_disconnects = Column(Integer, default=0)
    circuit_breaker_trips = Column(Integer, default=0)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat() if self.date else None,
            "trades_detected": self.trades_detected,
            "trades_executed": self.trades_executed,
            "trades_skipped": self.trades_skipped,
            "trades_failed": self.trades_failed,
            "total_volume_usd": str(self.total_volume_usd),
            "realized_pnl": str(self.realized_pnl),
            "avg_detection_latency_ms": self.avg_detection_latency_ms,
            "avg_execution_latency_ms": self.avg_execution_latency_ms,
            "avg_slippage_percent": str(self.avg_slippage_percent) if self.avg_slippage_percent else None,
        }


# =============================================================================
# DISCOVERY PERSISTENCE MODELS
# =============================================================================


class DiscoveryScanStatus(str, Enum):
    """Status of a discovery scan."""

    PENDING = "pending"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DiscoveredAccountStatus(str, Enum):
    """Status of a discovered account in the pipeline."""

    COLLECTED = "collected"  # Basic info from leaderboard
    LIGHT_SCANNED = "light_scanned"  # Phase 2 metrics available
    DEEP_ANALYZED = "deep_analyzed"  # Full analysis complete
    PROMOTED = "promoted"  # Promoted to tracked accounts
    REJECTED = "rejected"  # Manually rejected
    STALE = "stale"  # Metrics outdated, needs refresh


class DiscoveryScanRecord(Base):
    """Record of a discovery scan with checkpoint support."""

    __tablename__ = "discovery_scans"
    __table_args__ = (
        Index("idx_discovery_scans_status", "status"),
        Index("idx_discovery_scans_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Scan configuration
    scan_name = Column(String(100), nullable=True)
    mode = Column(String(50), nullable=False)
    config_json = Column(JSON, nullable=False)

    # Progress tracking
    status = Column(SQLEnum(DiscoveryScanStatus), default=DiscoveryScanStatus.PENDING)
    progress_pct = Column(Integer, default=0)
    current_phase = Column(String(50), nullable=True)
    checkpoint_data = Column(JSON, nullable=True)  # For resume capability

    # Results summary
    total_candidates = Column(Integer, default=0)
    collected_count = Column(Integer, default=0)
    light_scanned_count = Column(Integer, default=0)
    deep_analyzed_count = Column(Integer, default=0)
    passed_count = Column(Integer, default=0)

    # API efficiency
    api_calls_made = Column(Integer, default=0)

    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    last_checkpoint_at = Column(DateTime, nullable=True)

    # Errors
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    accounts = relationship("DiscoveredAccount", back_populates="scan", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<DiscoveryScanRecord(id={self.id}, status='{self.status.value}')>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scan_name": self.scan_name,
            "mode": self.mode,
            "config": self.config_json,
            "status": self.status.value,
            "progress_pct": self.progress_pct,
            "current_phase": self.current_phase,
            "total_candidates": self.total_candidates,
            "collected_count": self.collected_count,
            "light_scanned_count": self.light_scanned_count,
            "deep_analyzed_count": self.deep_analyzed_count,
            "passed_count": self.passed_count,
            "api_calls_made": self.api_calls_made,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DiscoveredAccount(Base):
    """A discovered account with profitability metrics.

    This is the main persistence layer for the wide-net collection.
    Accounts progress through statuses as they're analyzed more deeply.
    """

    __tablename__ = "discovered_accounts"
    __table_args__ = (
        UniqueConstraint("wallet_address", name="uq_discovered_wallet"),
        Index("idx_discovered_accounts_status", "status"),
        Index("idx_discovered_accounts_score", "composite_score"),
        Index("idx_discovered_accounts_pnl", "total_pnl"),
        Index("idx_discovered_accounts_scan", "scan_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey("discovery_scans.id"), nullable=True)

    wallet_address = Column(String(42), nullable=False)
    status = Column(SQLEnum(DiscoveredAccountStatus), default=DiscoveredAccountStatus.COLLECTED)

    # Basic leaderboard metrics (Phase 1 - Collection)
    leaderboard_rank = Column(Integer, nullable=True)
    leaderboard_category = Column(String(50), nullable=True)
    total_pnl = Column(Numeric(20, 6), nullable=True)
    total_volume = Column(Numeric(20, 6), nullable=True)
    num_trades = Column(Integer, nullable=True)
    position_count = Column(Integer, nullable=True)

    # Light scan metrics (Phase 2 - 1 API call)
    win_rate = Column(Numeric(10, 6), nullable=True)
    avg_position_size = Column(Numeric(20, 6), nullable=True)
    account_age_days = Column(Integer, nullable=True)
    active_days = Column(Integer, nullable=True)
    unique_markets = Column(Integer, nullable=True)

    # Deep analysis metrics (Phase 3 - Full analysis)
    sharpe_ratio = Column(Numeric(10, 4), nullable=True)
    sortino_ratio = Column(Numeric(10, 4), nullable=True)
    profit_factor = Column(Numeric(10, 4), nullable=True)
    max_drawdown_pct = Column(Numeric(10, 6), nullable=True)
    largest_win_pct = Column(Numeric(10, 6), nullable=True)
    top3_wins_pct = Column(Numeric(10, 6), nullable=True)

    # Category breakdown (stored as JSON)
    category_breakdown = Column(JSON, nullable=True)

    # Scoring
    composite_score = Column(Numeric(10, 4), nullable=True)
    passes_threshold = Column(Boolean, default=False)

    # Red flags (stored as JSON array)
    red_flags = Column(JSON, nullable=True)
    red_flag_count = Column(Integer, default=0)

    # Full analysis snapshot (for detailed review)
    full_analysis_json = Column(JSON, nullable=True)

    # User actions
    notes = Column(Text, nullable=True)
    starred = Column(Boolean, default=False)

    # Timing
    collected_at = Column(DateTime, default=func.now())
    light_scanned_at = Column(DateTime, nullable=True)
    deep_analyzed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    scan = relationship("DiscoveryScanRecord", back_populates="accounts")

    def __repr__(self) -> str:
        return f"<DiscoveredAccount(wallet='{self.wallet_address[:10]}...', score={self.composite_score})>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "wallet_address": self.wallet_address,
            "status": self.status.value,
            "leaderboard_rank": self.leaderboard_rank,
            "leaderboard_category": self.leaderboard_category,
            "total_pnl": str(self.total_pnl) if self.total_pnl else None,
            "total_volume": str(self.total_volume) if self.total_volume else None,
            "num_trades": self.num_trades,
            "win_rate": float(self.win_rate) if self.win_rate else None,
            "avg_position_size": str(self.avg_position_size) if self.avg_position_size else None,
            "account_age_days": self.account_age_days,
            "sharpe_ratio": float(self.sharpe_ratio) if self.sharpe_ratio else None,
            "sortino_ratio": float(self.sortino_ratio) if self.sortino_ratio else None,
            "profit_factor": float(self.profit_factor) if self.profit_factor else None,
            "max_drawdown_pct": float(self.max_drawdown_pct) if self.max_drawdown_pct else None,
            "category_breakdown": self.category_breakdown,
            "composite_score": float(self.composite_score) if self.composite_score else None,
            "passes_threshold": self.passes_threshold,
            "red_flags": self.red_flags,
            "red_flag_count": self.red_flag_count,
            "starred": self.starred,
            "notes": self.notes,
            "collected_at": self.collected_at.isoformat() if self.collected_at else None,
            "deep_analyzed_at": self.deep_analyzed_at.isoformat() if self.deep_analyzed_at else None,
        }

    def to_summary_dict(self) -> dict:
        """Compact summary for list views."""
        return {
            "id": self.id,
            "wallet": self.wallet_address[:10] + "...",
            "pnl": f"${float(self.total_pnl):,.0f}" if self.total_pnl else "N/A",
            "trades": self.num_trades,
            "win_rate": f"{float(self.win_rate)*100:.0f}%" if self.win_rate else "N/A",
            "score": float(self.composite_score) if self.composite_score else 0,
            "status": self.status.value,
            "starred": self.starred,
        }
