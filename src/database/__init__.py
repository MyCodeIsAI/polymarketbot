"""Database module for PolymarketBot.

This module provides:
- SQLAlchemy models for all entities
- Database connection management
- Migration system
- Repository pattern for data access
- High-level services for trade logging, persistence, and recovery
"""

from .models import (
    Base,
    PositionStatus,
    TradeStatus,
    EventSeverity,
    TrackedAccount,
    Position,
    TradeLog,
    SystemEvent,
    BalanceSnapshot,
    CircuitBreakerEvent,
    DailyStats,
)
from .connection import (
    DatabaseConfig,
    DatabaseManager,
    get_database,
    configure_database,
    close_database,
)
from .migrations import (
    Migration,
    MigrationHistory,
    MigrationManager,
    get_default_migrations,
    setup_migrations,
)
from .repositories import (
    BaseRepository,
    AccountRepository,
    PositionRepository,
    TradeLogRepository,
    SystemEventRepository,
    BalanceSnapshotRepository,
    CircuitBreakerEventRepository,
    DailyStatsRepository,
)
from .services import (
    TradeContext,
    TradeResult,
    TradeLogger,
    PositionPersistence,
    RecoveryState,
    RecoveryService,
    AuditService,
)

__all__ = [
    # Models
    "Base",
    "PositionStatus",
    "TradeStatus",
    "EventSeverity",
    "TrackedAccount",
    "Position",
    "TradeLog",
    "SystemEvent",
    "BalanceSnapshot",
    "CircuitBreakerEvent",
    "DailyStats",
    # Connection
    "DatabaseConfig",
    "DatabaseManager",
    "get_database",
    "configure_database",
    "close_database",
    # Migrations
    "Migration",
    "MigrationHistory",
    "MigrationManager",
    "get_default_migrations",
    "setup_migrations",
    # Repositories
    "BaseRepository",
    "AccountRepository",
    "PositionRepository",
    "TradeLogRepository",
    "SystemEventRepository",
    "BalanceSnapshotRepository",
    "CircuitBreakerEventRepository",
    "DailyStatsRepository",
    # Services
    "TradeContext",
    "TradeResult",
    "TradeLogger",
    "PositionPersistence",
    "RecoveryState",
    "RecoveryService",
    "AuditService",
]
