"""Tests for database module."""

import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch
import pytest

from src.database.models import (
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
from src.database.connection import (
    DatabaseConfig,
    DatabaseManager,
    configure_database,
    close_database,
)
from src.database.migrations import (
    Migration,
    MigrationManager,
    setup_migrations,
)
from src.database.repositories import (
    AccountRepository,
    PositionRepository,
    TradeLogRepository,
    SystemEventRepository,
    BalanceSnapshotRepository,
    CircuitBreakerEventRepository,
    DailyStatsRepository,
)
from src.database.services import (
    TradeContext,
    TradeResult,
    TradeLogger,
    PositionPersistence,
    RecoveryService,
    AuditService,
)


@pytest.fixture
def db():
    """Create a temporary in-memory database."""
    config = DatabaseConfig(url="sqlite:///:memory:", echo=False)
    db_manager = DatabaseManager(config)
    db_manager.initialize()
    yield db_manager
    db_manager.close()


@pytest.fixture
def db_file():
    """Create a temporary file-based database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    config = DatabaseConfig(url=f"sqlite:///{db_path}", echo=False)
    db_manager = DatabaseManager(config)
    db_manager.initialize()
    yield db_manager
    db_manager.close()
    os.unlink(db_path)


# =============================================================================
# Model Tests
# =============================================================================

class TestTrackedAccount:
    """Tests for TrackedAccount model."""

    def test_create_account(self, db):
        """Test creating an account."""
        with db.session() as session:
            account = TrackedAccount(
                name="test_whale",
                target_wallet="0x1234567890abcdef1234567890abcdef12345678",
                position_ratio=Decimal("0.01"),
                max_position_usd=Decimal("500"),
                slippage_tolerance=Decimal("0.05"),
            )
            session.add(account)
            session.flush()

            assert account.id is not None
            assert account.enabled is True

    def test_to_dict(self, db):
        """Test converting account to dict."""
        with db.session() as session:
            account = TrackedAccount(
                name="test_whale",
                target_wallet="0x1234",
                position_ratio=Decimal("0.01"),
                max_position_usd=Decimal("500"),
                slippage_tolerance=Decimal("0.05"),
            )
            session.add(account)
            session.flush()

            data = account.to_dict()
            assert data["name"] == "test_whale"
            assert data["position_ratio"] == "0.01"


class TestPosition:
    """Tests for Position model."""

    def test_create_position(self, db):
        """Test creating a position."""
        with db.session() as session:
            # Create account first
            account = TrackedAccount(
                name="test",
                target_wallet="0x1234",
                position_ratio=Decimal("0.01"),
                max_position_usd=Decimal("500"),
                slippage_tolerance=Decimal("0.05"),
            )
            session.add(account)
            session.flush()

            position = Position(
                account_id=account.id,
                market_id="0xmarket",
                condition_id="0xcondition",
                token_id="0xtoken",
                outcome="Yes",
                target_size=Decimal("100"),
                our_size=Decimal("1"),
                average_price=Decimal("0.5"),
            )
            session.add(position)
            session.flush()

            assert position.id is not None
            assert position.status == PositionStatus.SYNCED


class TestTradeLog:
    """Tests for TradeLog model."""

    def test_create_trade_log(self, db):
        """Test creating a trade log."""
        with db.session() as session:
            # Create account
            account = TrackedAccount(
                name="test",
                target_wallet="0x1234",
                position_ratio=Decimal("0.01"),
                max_position_usd=Decimal("500"),
                slippage_tolerance=Decimal("0.05"),
            )
            session.add(account)
            session.flush()

            trade = TradeLog(
                account_id=account.id,
                market_id="0xmarket",
                condition_id="0xcondition",
                token_id="0xtoken",
                outcome="Yes",
                side="BUY",
                target_price=Decimal("0.5"),
                target_size=Decimal("100"),
                detected_at=datetime.utcnow(),
            )
            session.add(trade)
            session.flush()

            assert trade.id is not None
            assert trade.status == TradeStatus.DETECTED


# =============================================================================
# Connection Tests
# =============================================================================

class TestDatabaseConnection:
    """Tests for database connection management."""

    def test_initialize(self, db):
        """Test database initialization."""
        assert db._initialized is True
        assert db._engine is not None

    def test_session_context_manager(self, db):
        """Test session context manager."""
        with db.session() as session:
            account = TrackedAccount(
                name="test",
                target_wallet="0x1234",
                position_ratio=Decimal("0.01"),
                max_position_usd=Decimal("500"),
                slippage_tolerance=Decimal("0.05"),
            )
            session.add(account)

        # Verify committed
        with db.session() as session:
            accounts = session.query(TrackedAccount).all()
            assert len(accounts) == 1

    def test_session_rollback_on_error(self, db):
        """Test session rollback on error."""
        try:
            with db.session() as session:
                account = TrackedAccount(
                    name="test",
                    target_wallet="0x1234",
                    position_ratio=Decimal("0.01"),
                    max_position_usd=Decimal("500"),
                    slippage_tolerance=Decimal("0.05"),
                )
                session.add(account)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify rolled back
        with db.session() as session:
            accounts = session.query(TrackedAccount).all()
            assert len(accounts) == 0

    def test_close(self, db):
        """Test closing database."""
        db.close()
        assert db._engine is None
        assert db._initialized is False


# =============================================================================
# Migration Tests
# =============================================================================

class TestMigrations:
    """Tests for migration system."""

    def test_migration_manager_init(self, db):
        """Test migration manager initialization."""
        manager = MigrationManager(db)
        assert manager.get_current_version() == 0

    def test_register_migration(self, db):
        """Test registering a migration."""
        manager = MigrationManager(db)

        migration = Migration(
            version=1,
            name="test_migration",
            description="Test",
            up=lambda s: None,
        )
        manager.register(migration)

        assert len(manager._migrations) == 1

    def test_apply_migration(self, db):
        """Test applying a migration."""
        manager = MigrationManager(db)

        applied = []

        def up_func(session):
            applied.append(True)

        migration = Migration(
            version=1,
            name="test_migration",
            description="Test",
            up=up_func,
        )
        manager.register(migration)

        count = manager.migrate()

        assert count == 1
        assert len(applied) == 1
        assert manager.get_current_version() == 1

    def test_rollback_migration(self, db):
        """Test rolling back a migration."""
        manager = MigrationManager(db)

        migration = Migration(
            version=1,
            name="test_migration",
            description="Test",
            up=lambda s: None,
            down=lambda s: None,
        )
        manager.register(migration)
        manager.migrate()

        count = manager.rollback()

        assert count == 1
        assert manager.get_current_version() == 0

    def test_setup_default_migrations(self, db):
        """Test setting up default migrations."""
        manager = setup_migrations(db)
        assert len(manager._migrations) > 0


# =============================================================================
# Repository Tests
# =============================================================================

class TestAccountRepository:
    """Tests for AccountRepository."""

    def test_create_account(self, db):
        """Test creating an account."""
        repo = AccountRepository(db)

        account = repo.create(
            name="whale1",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
        )

        assert account.id is not None
        assert account.name == "whale1"

    def test_get_by_name(self, db):
        """Test getting account by name."""
        repo = AccountRepository(db)

        repo.create(
            name="whale1",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
        )

        account = repo.get_by_name("whale1")
        assert account is not None
        assert account.name == "whale1"

        not_found = repo.get_by_name("nonexistent")
        assert not_found is None

    def test_get_enabled(self, db):
        """Test getting enabled accounts."""
        repo = AccountRepository(db)

        repo.create(
            name="enabled",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
            enabled=True,
        )
        repo.create(
            name="disabled",
            target_wallet="0x5678",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
            enabled=False,
        )

        enabled = repo.get_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "enabled"


class TestPositionRepository:
    """Tests for PositionRepository."""

    @pytest.fixture
    def account(self, db):
        """Create a test account."""
        repo = AccountRepository(db)
        return repo.create(
            name="test",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
        )

    def test_upsert_position_create(self, db, account):
        """Test upserting creates new position."""
        repo = PositionRepository(db)

        position = repo.upsert_position(
            account_id=account.id,
            token_id="0xtoken1",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="Yes",
            our_size=Decimal("10"),
            target_size=Decimal("1000"),
            average_price=Decimal("0.5"),
        )

        assert position.id is not None
        assert position.our_size == Decimal("10")

    def test_upsert_position_update(self, db, account):
        """Test upserting updates existing position."""
        repo = PositionRepository(db)

        # Create
        repo.upsert_position(
            account_id=account.id,
            token_id="0xtoken1",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="Yes",
            our_size=Decimal("10"),
            target_size=Decimal("1000"),
            average_price=Decimal("0.5"),
        )

        # Update
        position = repo.upsert_position(
            account_id=account.id,
            token_id="0xtoken1",
            our_size=Decimal("20"),
        )

        assert position.our_size == Decimal("20")

        # Verify only one record
        assert repo.count() == 1

    def test_get_open_positions(self, db, account):
        """Test getting open positions."""
        repo = PositionRepository(db)

        repo.upsert_position(
            account_id=account.id,
            token_id="0xtoken1",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="Yes",
            our_size=Decimal("10"),
            target_size=Decimal("1000"),
            average_price=Decimal("0.5"),
            status=PositionStatus.SYNCED,
        )
        repo.upsert_position(
            account_id=account.id,
            token_id="0xtoken2",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="No",
            our_size=Decimal("0"),
            target_size=Decimal("0"),
            average_price=Decimal("0.5"),
            status=PositionStatus.CLOSED,
        )

        open_positions = repo.get_open_positions()
        assert len(open_positions) == 1

    def test_close_position(self, db, account):
        """Test closing a position."""
        repo = PositionRepository(db)

        repo.upsert_position(
            account_id=account.id,
            token_id="0xtoken1",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="Yes",
            our_size=Decimal("10"),
            target_size=Decimal("1000"),
            average_price=Decimal("0.5"),
        )

        result = repo.close_position(account.id, "0xtoken1")
        assert result is True

        position = repo.get_by_token(account.id, "0xtoken1")
        assert position.status == PositionStatus.CLOSED
        assert position.our_size == Decimal("0")


class TestTradeLogRepository:
    """Tests for TradeLogRepository."""

    @pytest.fixture
    def account(self, db):
        """Create a test account."""
        repo = AccountRepository(db)
        return repo.create(
            name="test",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
        )

    def test_log_trade(self, db, account):
        """Test logging a trade."""
        repo = TradeLogRepository(db)

        trade = repo.log_trade(
            account_id=account.id,
            market_id="0xmarket",
            condition_id="0xcondition",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            target_price=Decimal("0.5"),
            target_size=Decimal("100"),
            detected_at=datetime.utcnow(),
        )

        assert trade.id is not None
        assert trade.status == TradeStatus.DETECTED

    def test_update_status(self, db, account):
        """Test updating trade status."""
        repo = TradeLogRepository(db)

        trade = repo.log_trade(
            account_id=account.id,
            market_id="0xmarket",
            condition_id="0xcondition",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            target_price=Decimal("0.5"),
            target_size=Decimal("100"),
            detected_at=datetime.utcnow(),
        )

        updated = repo.update_status(
            trade.id,
            TradeStatus.FILLED,
            execution_price=Decimal("0.51"),
            execution_size=Decimal("100"),
        )

        assert updated.status == TradeStatus.FILLED
        assert updated.execution_price == Decimal("0.51")
        assert updated.executed_at is not None

    def test_get_trade_stats(self, db, account):
        """Test getting trade statistics."""
        repo = TradeLogRepository(db)

        # Create some trades
        for i in range(5):
            trade = repo.log_trade(
                account_id=account.id,
                market_id="0xmarket",
                condition_id="0xcondition",
                token_id="0xtoken",
                outcome="Yes",
                side="BUY",
                target_price=Decimal("0.5"),
                target_size=Decimal("100"),
                detected_at=datetime.utcnow(),
            )

            if i < 3:
                repo.update_status(trade.id, TradeStatus.FILLED)
            else:
                repo.update_status(trade.id, TradeStatus.FAILED)

        stats = repo.get_trade_stats(datetime.utcnow() - timedelta(hours=1))

        assert stats["total"] == 5
        assert stats["filled"] == 3
        assert stats["failed"] == 2
        assert stats["fill_rate"] == 0.6


# =============================================================================
# Service Tests
# =============================================================================

class TestTradeLogger:
    """Tests for TradeLogger service."""

    @pytest.fixture
    def account(self, db):
        """Create a test account."""
        repo = AccountRepository(db)
        return repo.create(
            name="test",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
        )

    def test_log_detected(self, db, account):
        """Test logging detected trade."""
        logger = TradeLogger(db)

        context = TradeContext(
            account_id=account.id,
            account_name=account.name,
            market_id="0xmarket",
            condition_id="0xcondition",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            target_price=Decimal("0.5"),
            target_size=Decimal("100"),
            detected_at=datetime.utcnow(),
        )

        trade_id = logger.log_detected(context)
        assert trade_id is not None

    def test_log_result(self, db, account):
        """Test logging trade result."""
        logger = TradeLogger(db)

        context = TradeContext(
            account_id=account.id,
            account_name=account.name,
            market_id="0xmarket",
            condition_id="0xcondition",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            target_price=Decimal("0.5"),
            target_size=Decimal("100"),
            detected_at=datetime.utcnow(),
        )
        trade_id = logger.log_detected(context)

        result = TradeResult(
            success=True,
            trade_id=trade_id,
            status=TradeStatus.FILLED,
            execution_price=Decimal("0.51"),
            execution_size=Decimal("100"),
        )
        logger.log_result(trade_id, result)

        # Verify in database
        trade_repo = TradeLogRepository(db)
        trade = trade_repo.get_by_id(trade_id)
        assert trade.status == TradeStatus.FILLED


class TestPositionPersistence:
    """Tests for PositionPersistence service."""

    @pytest.fixture
    def account(self, db):
        """Create a test account."""
        repo = AccountRepository(db)
        return repo.create(
            name="test",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
        )

    def test_sync_position(self, db, account):
        """Test syncing a position."""
        persistence = PositionPersistence(db)

        position = persistence.sync_position(
            account_id=account.id,
            token_id="0xtoken",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="Yes",
            our_size=Decimal("10"),
            target_size=Decimal("1000"),
            average_price=Decimal("0.5"),
            current_price=Decimal("0.6"),
        )

        assert position.id is not None
        assert position.unrealized_pnl == Decimal("1")  # (0.6 - 0.5) * 10

    def test_load_positions(self, db, account):
        """Test loading positions."""
        persistence = PositionPersistence(db)

        persistence.sync_position(
            account_id=account.id,
            token_id="0xtoken1",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="Yes",
            our_size=Decimal("10"),
            target_size=Decimal("1000"),
            average_price=Decimal("0.5"),
        )
        persistence.sync_position(
            account_id=account.id,
            token_id="0xtoken2",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="No",
            our_size=Decimal("20"),
            target_size=Decimal("2000"),
            average_price=Decimal("0.5"),
        )

        positions = persistence.load_positions(account.id)
        assert len(positions) == 2


class TestRecoveryService:
    """Tests for RecoveryService."""

    @pytest.fixture
    def populated_db(self, db):
        """Create a database with some data."""
        # Create account
        account_repo = AccountRepository(db)
        account = account_repo.create(
            name="test",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
        )

        # Create position
        persistence = PositionPersistence(db)
        persistence.sync_position(
            account_id=account.id,
            token_id="0xtoken",
            market_id="0xmarket",
            condition_id="0xcondition",
            outcome="Yes",
            our_size=Decimal("10"),
            target_size=Decimal("1000"),
            average_price=Decimal("0.5"),
        )

        # Create trade
        trade_repo = TradeLogRepository(db)
        trade_repo.log_trade(
            account_id=account.id,
            market_id="0xmarket",
            condition_id="0xcondition",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            target_price=Decimal("0.5"),
            target_size=Decimal("100"),
            detected_at=datetime.utcnow(),
        )

        return db

    def test_recover(self, populated_db):
        """Test full recovery."""
        recovery = RecoveryService(populated_db)
        state = recovery.recover()

        assert len(state.accounts) == 1
        assert len(state.positions) == 1
        assert state.last_trade_timestamp is not None

    def test_mark_incomplete_trades_failed(self, populated_db):
        """Test marking incomplete trades as failed."""
        # Create a queued trade
        trade_repo = TradeLogRepository(populated_db)
        account_repo = AccountRepository(populated_db)
        account = account_repo.get_by_name("test")

        trade = trade_repo.log_trade(
            account_id=account.id,
            market_id="0xmarket",
            condition_id="0xcondition",
            token_id="0xtoken",
            outcome="Yes",
            side="BUY",
            target_price=Decimal("0.5"),
            target_size=Decimal("100"),
            detected_at=datetime.utcnow(),
        )
        trade_repo.update_status(trade.id, TradeStatus.QUEUED)

        recovery = RecoveryService(populated_db)
        count = recovery.mark_incomplete_trades_failed("Test restart")

        assert count == 1

        # Verify updated
        updated_trade = trade_repo.get_by_id(trade.id)
        assert updated_trade.status == TradeStatus.FAILED


class TestAuditService:
    """Tests for AuditService."""

    @pytest.fixture
    def populated_db(self, db):
        """Create a database with some data."""
        account_repo = AccountRepository(db)
        account = account_repo.create(
            name="test",
            target_wallet="0x1234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0.05"),
        )

        trade_repo = TradeLogRepository(db)
        for i in range(3):
            trade_repo.log_trade(
                account_id=account.id,
                market_id="0xmarket",
                condition_id="0xcondition",
                token_id="0xtoken",
                outcome="Yes",
                side="BUY",
                target_price=Decimal("0.5"),
                target_size=Decimal("100"),
                detected_at=datetime.utcnow(),
            )

        event_repo = SystemEventRepository(db)
        event_repo.log_event(
            event_type="test_event",
            severity=EventSeverity.INFO,
            message="Test message",
        )

        return db

    def test_get_trade_history(self, populated_db):
        """Test getting trade history."""
        audit = AuditService(populated_db)
        trades = audit.get_trade_history(days=1)

        assert len(trades) == 3

    def test_get_events(self, populated_db):
        """Test getting events."""
        audit = AuditService(populated_db)
        events = audit.get_events(days=1)

        assert len(events) >= 1

    def test_export_audit_data(self, populated_db):
        """Test exporting audit data."""
        audit = AuditService(populated_db)

        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow()

        data = audit.export_audit_data(start, end)

        assert "trades" in data
        assert "events" in data
        assert "statistics" in data
        assert len(data["trades"]) == 3
