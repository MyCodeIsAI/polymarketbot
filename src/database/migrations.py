"""Database migration system.

Provides:
- Schema version tracking
- Forward migrations
- Rollback support
- Migration validation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, List

from sqlalchemy import Column, Integer, String, DateTime, Text, func
from sqlalchemy.orm import Session

from ..utils.logging import get_logger
from .models import Base
from .connection import DatabaseManager

logger = get_logger(__name__)


class MigrationHistory(Base):
    """Tracks applied migrations."""

    __tablename__ = "migration_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    applied_at = Column(DateTime, default=func.now())
    description = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Migration(version={self.version}, name='{self.name}')>"


@dataclass
class Migration:
    """Defines a single migration."""

    version: int
    name: str
    description: str
    up: Callable[[Session], None]
    down: Optional[Callable[[Session], None]] = None


class MigrationManager:
    """Manages database migrations.

    Example:
        manager = MigrationManager(db)

        # Register migrations
        manager.register(Migration(
            version=1,
            name="add_user_notes",
            description="Add notes column to tracked_accounts",
            up=lambda s: s.execute("ALTER TABLE tracked_accounts ADD COLUMN notes TEXT"),
            down=lambda s: s.execute("ALTER TABLE tracked_accounts DROP COLUMN notes"),
        ))

        # Apply all pending
        manager.migrate()

        # Rollback one version
        manager.rollback()
    """

    def __init__(self, db: DatabaseManager):
        """Initialize migration manager.

        Args:
            db: Database manager
        """
        self.db = db
        self._migrations: dict[int, Migration] = {}
        self._ensure_history_table()

    def _ensure_history_table(self) -> None:
        """Ensure migration history table exists."""
        MigrationHistory.__table__.create(self.db.engine, checkfirst=True)

    def register(self, migration: Migration) -> None:
        """Register a migration.

        Args:
            migration: Migration to register
        """
        if migration.version in self._migrations:
            raise ValueError(f"Migration version {migration.version} already registered")

        self._migrations[migration.version] = migration

        logger.debug(
            "migration_registered",
            version=migration.version,
            name=migration.name,
        )

    def get_current_version(self) -> int:
        """Get the current database version.

        Returns:
            Current version number (0 if no migrations applied)
        """
        with self.db.session() as session:
            latest = (
                session.query(MigrationHistory)
                .order_by(MigrationHistory.version.desc())
                .first()
            )
            return latest.version if latest else 0

    def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations.

        Returns:
            List of migrations not yet applied
        """
        current = self.get_current_version()
        pending = [
            m for v, m in sorted(self._migrations.items())
            if v > current
        ]
        return pending

    def migrate(self, target_version: Optional[int] = None) -> int:
        """Apply pending migrations.

        Args:
            target_version: Specific version to migrate to (None = latest)

        Returns:
            Number of migrations applied
        """
        current = self.get_current_version()
        pending = self.get_pending_migrations()

        if target_version is not None:
            pending = [m for m in pending if m.version <= target_version]

        if not pending:
            logger.info("no_pending_migrations", current_version=current)
            return 0

        applied = 0
        for migration in pending:
            try:
                self._apply_migration(migration)
                applied += 1

                logger.info(
                    "migration_applied",
                    version=migration.version,
                    name=migration.name,
                )

            except Exception as e:
                logger.error(
                    "migration_failed",
                    version=migration.version,
                    name=migration.name,
                    error=str(e),
                )
                raise

        return applied

    def _apply_migration(self, migration: Migration) -> None:
        """Apply a single migration.

        Args:
            migration: Migration to apply
        """
        with self.db.session() as session:
            # Run the up migration
            migration.up(session)

            # Record in history
            history = MigrationHistory(
                version=migration.version,
                name=migration.name,
                description=migration.description,
            )
            session.add(history)

    def rollback(self, steps: int = 1) -> int:
        """Rollback migrations.

        Args:
            steps: Number of migrations to rollback

        Returns:
            Number of migrations rolled back
        """
        current = self.get_current_version()

        if current == 0:
            logger.info("no_migrations_to_rollback")
            return 0

        rolled_back = 0
        for _ in range(steps):
            version = self.get_current_version()
            if version == 0:
                break

            migration = self._migrations.get(version)
            if migration is None:
                logger.error(
                    "migration_not_found",
                    version=version,
                )
                break

            if migration.down is None:
                logger.error(
                    "migration_not_reversible",
                    version=version,
                    name=migration.name,
                )
                break

            try:
                self._rollback_migration(migration)
                rolled_back += 1

                logger.info(
                    "migration_rolled_back",
                    version=migration.version,
                    name=migration.name,
                )

            except Exception as e:
                logger.error(
                    "rollback_failed",
                    version=migration.version,
                    name=migration.name,
                    error=str(e),
                )
                raise

        return rolled_back

    def _rollback_migration(self, migration: Migration) -> None:
        """Rollback a single migration.

        Args:
            migration: Migration to rollback
        """
        with self.db.session() as session:
            # Run the down migration
            if migration.down:
                migration.down(session)

            # Remove from history
            session.query(MigrationHistory).filter(
                MigrationHistory.version == migration.version
            ).delete()

    def get_history(self) -> List[MigrationHistory]:
        """Get migration history.

        Returns:
            List of applied migrations
        """
        with self.db.session() as session:
            return (
                session.query(MigrationHistory)
                .order_by(MigrationHistory.version)
                .all()
            )

    def get_status(self) -> dict:
        """Get migration status.

        Returns:
            Status dictionary
        """
        current = self.get_current_version()
        pending = self.get_pending_migrations()
        history = self.get_history()

        return {
            "current_version": current,
            "latest_version": max(self._migrations.keys()) if self._migrations else 0,
            "pending_count": len(pending),
            "applied_count": len(history),
            "pending_migrations": [
                {"version": m.version, "name": m.name}
                for m in pending
            ],
        }


# Pre-defined migrations for initial schema evolution
def get_default_migrations() -> List[Migration]:
    """Get the default set of migrations.

    Returns:
        List of migrations
    """
    migrations = []

    # Migration 1: Add market_name to positions
    def up_001(session: Session) -> None:
        from sqlalchemy import text
        try:
            session.execute(text(
                "ALTER TABLE positions ADD COLUMN market_name VARCHAR(500)"
            ))
        except Exception:
            pass  # Column may already exist

    def down_001(session: Session) -> None:
        from sqlalchemy import text
        session.execute(text(
            "ALTER TABLE positions DROP COLUMN market_name"
        ))

    migrations.append(Migration(
        version=1,
        name="add_market_name_to_positions",
        description="Add market_name column to positions for display",
        up=up_001,
        down=down_001,
    ))

    # Migration 2: Add target trade info to trade_log
    def up_002(session: Session) -> None:
        from sqlalchemy import text
        try:
            session.execute(text(
                "ALTER TABLE trade_log ADD COLUMN target_order_id VARCHAR(100)"
            ))
            session.execute(text(
                "ALTER TABLE trade_log ADD COLUMN target_tx_hash VARCHAR(66)"
            ))
        except Exception:
            pass

    def down_002(session: Session) -> None:
        from sqlalchemy import text
        session.execute(text(
            "ALTER TABLE trade_log DROP COLUMN target_order_id"
        ))
        session.execute(text(
            "ALTER TABLE trade_log DROP COLUMN target_tx_hash"
        ))

    migrations.append(Migration(
        version=2,
        name="add_target_trade_info",
        description="Add target order_id and tx_hash to trade_log",
        up=up_002,
        down=down_002,
    ))

    # Migration 3: Add fill statistics to trade_log
    def up_003(session: Session) -> None:
        from sqlalchemy import text
        try:
            session.execute(text(
                "ALTER TABLE trade_log ADD COLUMN fill_percent NUMERIC(5,2)"
            ))
            session.execute(text(
                "ALTER TABLE trade_log ADD COLUMN partial_fills INTEGER DEFAULT 0"
            ))
        except Exception:
            pass

    def down_003(session: Session) -> None:
        from sqlalchemy import text
        session.execute(text(
            "ALTER TABLE trade_log DROP COLUMN fill_percent"
        ))
        session.execute(text(
            "ALTER TABLE trade_log DROP COLUMN partial_fills"
        ))

    migrations.append(Migration(
        version=3,
        name="add_fill_statistics",
        description="Add fill percentage and partial fill count to trade_log",
        up=up_003,
        down=down_003,
    ))

    return migrations


def setup_migrations(db: DatabaseManager) -> MigrationManager:
    """Setup migration manager with default migrations.

    Args:
        db: Database manager

    Returns:
        Configured migration manager
    """
    manager = MigrationManager(db)

    for migration in get_default_migrations():
        manager.register(migration)

    return manager
