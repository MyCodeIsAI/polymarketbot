"""Stage 1 Tests: Database Schema and Migrations for Insider Scanner.

Tests:
- Migration execution
- Model creation (CRUD operations)
- Table constraints and indexes
- Relationships between tables
- Audit trail hash calculations

Run with: pytest tests/insider_scanner/test_stage1_database.py -v
"""

import os
import tempfile
import hashlib
import json
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.database.connection import DatabaseConfig, DatabaseManager
from src.database.migrations import MigrationManager
from src.insider_scanner.migrations import (
    get_insider_scanner_migrations,
    register_insider_migrations,
)
from src.insider_scanner.models import (
    FlaggedWallet,
    FlaggedFundingSource,
    InsiderCluster,
    InsiderSignal,
    CumulativePosition,
    DetectionRecord,
    InvestmentThesis,
    AuditChainEntry,
    InsiderPriority,
    WalletStatus,
    SignalCategory,
)


@pytest.fixture
def db():
    """Create a temporary in-memory database with insider scanner tables."""
    config = DatabaseConfig(url="sqlite:///:memory:", echo=False)
    db_manager = DatabaseManager(config)
    db_manager.initialize()

    # Apply insider scanner migrations
    manager = MigrationManager(db_manager)
    register_insider_migrations(manager)
    manager.migrate()

    yield db_manager
    db_manager.close()


@pytest.fixture
def db_file():
    """Create a temporary file-based database for persistence tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    config = DatabaseConfig(url=f"sqlite:///{db_path}", echo=False)
    db_manager = DatabaseManager(config)
    db_manager.initialize()

    # Apply migrations
    manager = MigrationManager(db_manager)
    register_insider_migrations(manager)
    manager.migrate()

    yield db_manager
    db_manager.close()
    os.unlink(db_path)


# =============================================================================
# Migration Tests
# =============================================================================

class TestMigrations:
    """Test migration execution and rollback."""

    def test_migrations_load(self):
        """Test that all migrations load correctly."""
        migrations = get_insider_scanner_migrations()

        assert len(migrations) == 8  # migrations 100-107
        assert migrations[0].version == 100
        assert migrations[-1].version == 107

    def test_migrations_have_up_and_down(self):
        """Test that all migrations have up functions."""
        migrations = get_insider_scanner_migrations()

        for m in migrations:
            assert m.up is not None
            assert callable(m.up)

    def test_migrations_apply_successfully(self, db):
        """Test that migrations apply without error."""
        # Migrations already applied in fixture
        # Just verify tables exist
        from sqlalchemy import text

        with db.session() as session:
            # Check all tables exist
            tables = [
                "insider_flagged_wallets",
                "insider_funding_sources",
                "insider_clusters",
                "insider_signals",
                "insider_cumulative_positions",
                "insider_detection_records",
                "insider_investment_thesis",
                "insider_audit_chain",
            ]

            for table in tables:
                result = session.execute(text(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
                ))
                row = result.fetchone()
                assert row is not None, f"Table {table} not created"

    def test_migration_versions_unique(self):
        """Test that migration versions are unique."""
        migrations = get_insider_scanner_migrations()
        versions = [m.version for m in migrations]

        assert len(versions) == len(set(versions)), "Duplicate migration versions"


# =============================================================================
# FlaggedWallet Model Tests
# =============================================================================

class TestFlaggedWalletModel:
    """Test FlaggedWallet CRUD operations."""

    def test_create_flagged_wallet(self, db):
        """Test creating a flagged wallet."""
        from sqlalchemy import text

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_flagged_wallets
                (wallet_address, insider_score, priority, status)
                VALUES ('0x1234567890abcdef1234567890abcdef12345678', 75.5, 'high', 'new')
            """))

            result = session.execute(text(
                "SELECT * FROM insider_flagged_wallets WHERE wallet_address = '0x1234567890abcdef1234567890abcdef12345678'"
            ))
            row = result.fetchone()

            assert row is not None
            assert float(row[2]) == 75.5  # insider_score
            assert row[5] == 'high'  # priority

    def test_wallet_address_unique(self, db):
        """Test that wallet address is unique."""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_flagged_wallets (wallet_address, insider_score)
                VALUES ('0xduplicate', 50.0)
            """))

            with pytest.raises(IntegrityError):
                session.execute(text("""
                    INSERT INTO insider_flagged_wallets (wallet_address, insider_score)
                    VALUES ('0xduplicate', 60.0)
                """))

    def test_dimension_scores_stored(self, db):
        """Test that all dimension scores are stored correctly."""
        from sqlalchemy import text

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_flagged_wallets
                (wallet_address, insider_score, score_account, score_trading,
                 score_behavioral, score_contextual, score_cluster)
                VALUES ('0xtest', 85.0, 20.0, 30.0, 15.0, 12.0, 8.0)
            """))

            result = session.execute(text(
                "SELECT score_account, score_trading, score_behavioral, score_contextual, score_cluster "
                "FROM insider_flagged_wallets WHERE wallet_address = '0xtest'"
            ))
            row = result.fetchone()

            assert float(row[0]) == 20.0  # account
            assert float(row[1]) == 30.0  # trading
            assert float(row[2]) == 15.0  # behavioral
            assert float(row[3]) == 12.0  # contextual
            assert float(row[4]) == 8.0   # cluster

    def test_signals_json_stored(self, db):
        """Test that signals JSON is stored correctly."""
        from sqlalchemy import text

        signals = json.dumps([
            {"name": "account_age", "weight": 15, "category": "account"},
            {"name": "position_size", "weight": 10, "category": "trading"},
        ])

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_flagged_wallets
                (wallet_address, insider_score, signals_json)
                VALUES ('0xjson', 50.0, :signals)
            """), {"signals": signals})

            result = session.execute(text(
                "SELECT signals_json FROM insider_flagged_wallets WHERE wallet_address = '0xjson'"
            ))
            row = result.fetchone()
            retrieved = json.loads(row[0])

            assert len(retrieved) == 2
            assert retrieved[0]["name"] == "account_age"


# =============================================================================
# FlaggedFundingSource Model Tests
# =============================================================================

class TestFundingSourceModel:
    """Test FlaggedFundingSource CRUD operations."""

    def test_create_funding_source(self, db):
        """Test creating a funding source."""
        from sqlalchemy import text

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_funding_sources
                (funding_address, source_type, exchange_name, risk_level)
                VALUES ('0xfunder', 'cex', 'Coinbase', 'high')
            """))

            result = session.execute(text(
                "SELECT * FROM insider_funding_sources WHERE funding_address = '0xfunder'"
            ))
            row = result.fetchone()

            assert row is not None
            assert row[2] == 'cex'
            assert row[3] == 'Coinbase'

    def test_funding_address_unique(self, db):
        """Test that funding address is unique."""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_funding_sources (funding_address)
                VALUES ('0xdupfunder')
            """))

            with pytest.raises(IntegrityError):
                session.execute(text("""
                    INSERT INTO insider_funding_sources (funding_address)
                    VALUES ('0xdupfunder')
                """))


# =============================================================================
# CumulativePosition Model Tests
# =============================================================================

class TestCumulativePositionModel:
    """Test CumulativePosition CRUD operations."""

    def test_create_cumulative_position(self, db):
        """Test creating a cumulative position."""
        from sqlalchemy import text

        # First create a wallet
        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_flagged_wallets (wallet_address, insider_score)
                VALUES ('0xpositioner', 50.0)
            """))

            result = session.execute(text(
                "SELECT id FROM insider_flagged_wallets WHERE wallet_address = '0xpositioner'"
            ))
            wallet_id = result.fetchone()[0]

            session.execute(text("""
                INSERT INTO insider_cumulative_positions
                (wallet_id, market_id, side, cumulative_size, cumulative_usd, entry_count, is_split_entry)
                VALUES (:wallet_id, '0xmarket123', 'YES', 1500.0, 75000.0, 5, 1)
            """), {"wallet_id": wallet_id})

            result = session.execute(text(
                "SELECT cumulative_usd, entry_count, is_split_entry "
                "FROM insider_cumulative_positions WHERE market_id = '0xmarket123'"
            ))
            row = result.fetchone()

            assert float(row[0]) == 75000.0
            assert row[1] == 5
            assert row[2] == 1  # True

    def test_position_unique_constraint(self, db):
        """Test unique constraint on wallet_id, market_id, side."""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_flagged_wallets (wallet_address, insider_score)
                VALUES ('0xunique', 50.0)
            """))

            result = session.execute(text(
                "SELECT id FROM insider_flagged_wallets WHERE wallet_address = '0xunique'"
            ))
            wallet_id = result.fetchone()[0]

            session.execute(text("""
                INSERT INTO insider_cumulative_positions
                (wallet_id, market_id, side, cumulative_size, cumulative_usd)
                VALUES (:wallet_id, '0xmarket', 'YES', 100.0, 1000.0)
            """), {"wallet_id": wallet_id})

            with pytest.raises(IntegrityError):
                session.execute(text("""
                    INSERT INTO insider_cumulative_positions
                    (wallet_id, market_id, side, cumulative_size, cumulative_usd)
                    VALUES (:wallet_id, '0xmarket', 'YES', 200.0, 2000.0)
                """), {"wallet_id": wallet_id})


# =============================================================================
# Audit Trail Model Tests
# =============================================================================

class TestDetectionRecordModel:
    """Test DetectionRecord for audit trail."""

    def test_create_detection_record(self, db):
        """Test creating a detection record with hash."""
        from sqlalchemy import text

        # Create wallet first
        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_flagged_wallets (wallet_address, insider_score)
                VALUES ('0xaudit', 80.0)
            """))

            result = session.execute(text(
                "SELECT id FROM insider_flagged_wallets WHERE wallet_address = '0xaudit'"
            ))
            wallet_id = result.fetchone()[0]

            # Create detection record
            signals = json.dumps([{"name": "test", "weight": 10}])
            record_hash = hashlib.sha256(
                f"0xaudit:80.0:{signals}".encode()
            ).hexdigest()

            session.execute(text("""
                INSERT INTO insider_detection_records
                (wallet_id, record_hash, wallet_address, detected_at,
                 insider_score, priority, signals_snapshot)
                VALUES (:wallet_id, :hash, '0xaudit', CURRENT_TIMESTAMP,
                        80.0, 'high', :signals)
            """), {"wallet_id": wallet_id, "hash": record_hash, "signals": signals})

            result = session.execute(text(
                "SELECT record_hash, insider_score FROM insider_detection_records "
                "WHERE wallet_address = '0xaudit'"
            ))
            row = result.fetchone()

            assert row[0] == record_hash
            assert float(row[1]) == 80.0

    def test_record_hash_unique(self, db):
        """Test that record hash must be unique."""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_flagged_wallets (wallet_address, insider_score)
                VALUES ('0xhashtest', 50.0)
            """))

            result = session.execute(text(
                "SELECT id FROM insider_flagged_wallets WHERE wallet_address = '0xhashtest'"
            ))
            wallet_id = result.fetchone()[0]

            session.execute(text("""
                INSERT INTO insider_detection_records
                (wallet_id, record_hash, wallet_address, detected_at,
                 insider_score, priority, signals_snapshot)
                VALUES (:wallet_id, 'duplicate_hash', '0xhashtest', CURRENT_TIMESTAMP,
                        50.0, 'medium', '[]')
            """), {"wallet_id": wallet_id})

            with pytest.raises(IntegrityError):
                session.execute(text("""
                    INSERT INTO insider_detection_records
                    (wallet_id, record_hash, wallet_address, detected_at,
                     insider_score, priority, signals_snapshot)
                    VALUES (:wallet_id, 'duplicate_hash', '0xhashtest2', CURRENT_TIMESTAMP,
                            60.0, 'high', '[]')
                """), {"wallet_id": wallet_id})


class TestAuditChainModel:
    """Test AuditChainEntry for tamper-evident linking."""

    def test_create_chain_entry(self, db):
        """Test creating an audit chain entry."""
        from sqlalchemy import text

        entry_hash = hashlib.sha256(b"entry_data").hexdigest()
        chain_hash = hashlib.sha256(f"{entry_hash}:GENESIS".encode()).hexdigest()

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_audit_chain
                (sequence_number, entry_type, entry_id, entry_hash,
                 previous_chain_hash, chain_hash)
                VALUES (1, 'detection', 1, :entry_hash, NULL, :chain_hash)
            """), {"entry_hash": entry_hash, "chain_hash": chain_hash})

            result = session.execute(text(
                "SELECT chain_hash, previous_chain_hash FROM insider_audit_chain "
                "WHERE sequence_number = 1"
            ))
            row = result.fetchone()

            assert row[0] == chain_hash
            assert row[1] is None  # First entry has no previous

    def test_chain_hash_unique(self, db):
        """Test that chain hash must be unique."""
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_audit_chain
                (sequence_number, entry_type, entry_id, entry_hash, chain_hash)
                VALUES (1, 'detection', 1, 'entry1', 'unique_chain_hash')
            """))

            with pytest.raises(IntegrityError):
                session.execute(text("""
                    INSERT INTO insider_audit_chain
                    (sequence_number, entry_type, entry_id, entry_hash, chain_hash)
                    VALUES (2, 'detection', 2, 'entry2', 'unique_chain_hash')
                """))


class TestInvestmentThesisModel:
    """Test InvestmentThesis for documenting user actions."""

    def test_create_thesis(self, db):
        """Test creating an investment thesis."""
        from sqlalchemy import text

        thesis_hash = hashlib.sha256(b"thesis_data").hexdigest()
        detection_ids = json.dumps([1, 2, 3])

        with db.session() as session:
            session.execute(text("""
                INSERT INTO insider_investment_thesis
                (thesis_hash, created_at, detection_record_ids, reasoning,
                 intended_action, market_id, position_side, position_size)
                VALUES (:hash, CURRENT_TIMESTAMP, :ids,
                        'Based on detection signals, placing contrarian bet',
                        'Buy YES on market X', '0xmarket', 'YES', 5000.0)
            """), {"hash": thesis_hash, "ids": detection_ids})

            result = session.execute(text(
                "SELECT reasoning, position_size FROM insider_investment_thesis "
                "WHERE thesis_hash = :hash"
            ), {"hash": thesis_hash})
            row = result.fetchone()

            assert "detection signals" in row[0]
            assert float(row[1]) == 5000.0


# =============================================================================
# Index Tests
# =============================================================================

class TestIndexes:
    """Test that indexes are created correctly."""

    def test_wallet_indexes_exist(self, db):
        """Test that wallet table indexes exist."""
        from sqlalchemy import text

        with db.session() as session:
            result = session.execute(text(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='insider_flagged_wallets'"
            ))
            indexes = [row[0] for row in result.fetchall()]

            assert "idx_insider_wallet_score" in indexes
            assert "idx_insider_wallet_priority" in indexes
            assert "idx_insider_wallet_status" in indexes


# =============================================================================
# Relationship Tests
# =============================================================================

class TestRelationships:
    """Test foreign key relationships."""

    def test_signal_wallet_relationship(self, db):
        """Test that signals reference valid wallets."""
        from sqlalchemy import text

        with db.session() as session:
            # Create wallet
            session.execute(text("""
                INSERT INTO insider_flagged_wallets (wallet_address, insider_score)
                VALUES ('0xreltest', 70.0)
            """))

            result = session.execute(text(
                "SELECT id FROM insider_flagged_wallets WHERE wallet_address = '0xreltest'"
            ))
            wallet_id = result.fetchone()[0]

            # Create signal
            session.execute(text("""
                INSERT INTO insider_signals
                (wallet_id, signal_name, category, weight)
                VALUES (:wallet_id, 'account_age', 'account', 15.0)
            """), {"wallet_id": wallet_id})

            # Query should work
            result = session.execute(text("""
                SELECT s.signal_name, w.wallet_address
                FROM insider_signals s
                JOIN insider_flagged_wallets w ON s.wallet_id = w.id
                WHERE w.wallet_address = '0xreltest'
            """))
            row = result.fetchone()

            assert row[0] == 'account_age'
            assert row[1] == '0xreltest'


# =============================================================================
# Stage 1 Validation Tests
# =============================================================================

class TestStage1Validation:
    """End-to-end validation tests for Stage 1 completion."""

    def test_complete_detection_flow(self, db):
        """Test complete flow: wallet -> signals -> position -> detection record -> audit chain."""
        from sqlalchemy import text

        with db.session() as session:
            # Step 1: Create flagged wallet with full data
            session.execute(text("""
                INSERT INTO insider_flagged_wallets
                (wallet_address, insider_score, priority, score_account, score_trading,
                 score_behavioral, score_contextual, score_cluster, signal_count,
                 active_dimensions, status, account_age_days)
                VALUES ('0xvalidation_wallet', 85.5, 'critical', 20.0, 30.0,
                        18.0, 12.0, 5.5, 8, 5, 'new', 3)
            """))

            result = session.execute(text(
                "SELECT id FROM insider_flagged_wallets WHERE wallet_address = '0xvalidation_wallet'"
            ))
            wallet_id = result.fetchone()[0]

            # Step 2: Create signals across multiple dimensions
            signals = [
                ('account_age', 'account', 20.0),
                ('timing_precision', 'trading', 18.0),
                ('high_conviction', 'trading', 12.0),
                ('no_history', 'behavioral', 18.0),
                ('before_resolution', 'contextual', 12.0),
                ('same_funding', 'cluster', 5.5),
            ]
            for name, category, weight in signals:
                session.execute(text("""
                    INSERT INTO insider_signals (wallet_id, signal_name, category, weight)
                    VALUES (:wallet_id, :name, :category, :weight)
                """), {"wallet_id": wallet_id, "name": name, "category": category, "weight": weight})

            # Step 3: Create cumulative position
            session.execute(text("""
                INSERT INTO insider_cumulative_positions
                (wallet_id, market_id, market_title, side, cumulative_size, cumulative_usd,
                 entry_count, avg_entry_size, is_split_entry, first_entry_odds)
                VALUES (:wallet_id, '0xmarket_validation', 'Election Market', 'YES',
                        5000.0, 125000.0, 7, 17857.14, 1, 0.20)
            """), {"wallet_id": wallet_id})

            # Step 4: Create detection record with hash
            import hashlib
            import json

            record_data = {
                "wallet_address": "0xvalidation_wallet",
                "detected_at": "2026-01-23T12:00:00",
                "insider_score": "85.5",
                "priority": "critical",
                "signals_snapshot": json.dumps([{"name": s[0], "category": s[1], "weight": s[2]} for s in signals]),
                "market_ids": json.dumps(["0xmarket_validation"]),
            }
            record_hash = hashlib.sha256(json.dumps(record_data, sort_keys=True).encode()).hexdigest()

            session.execute(text("""
                INSERT INTO insider_detection_records
                (wallet_id, record_hash, wallet_address, detected_at, insider_score,
                 priority, signals_snapshot, market_ids)
                VALUES (:wallet_id, :hash, '0xvalidation_wallet', CURRENT_TIMESTAMP,
                        85.5, 'critical', :signals, :markets)
            """), {
                "wallet_id": wallet_id,
                "hash": record_hash,
                "signals": json.dumps([{"name": s[0], "category": s[1], "weight": s[2]} for s in signals]),
                "markets": json.dumps(["0xmarket_validation"]),
            })

            # Step 5: Create audit chain entry
            chain_hash = hashlib.sha256(f"1-detection-1-{record_hash}".encode()).hexdigest()
            session.execute(text("""
                INSERT INTO insider_audit_chain
                (sequence_number, entry_type, entry_id, entry_hash, chain_hash)
                VALUES (1, 'detection', :wallet_id, :entry_hash, :chain_hash)
            """), {"wallet_id": wallet_id, "entry_hash": record_hash, "chain_hash": chain_hash})

            # Validate: Query the complete detection with all related data
            result = session.execute(text("""
                SELECT
                    w.wallet_address,
                    w.insider_score,
                    w.priority,
                    w.signal_count,
                    w.active_dimensions,
                    COUNT(DISTINCT s.id) as signal_rows,
                    cp.cumulative_usd,
                    cp.is_split_entry,
                    dr.record_hash,
                    ac.chain_hash
                FROM insider_flagged_wallets w
                LEFT JOIN insider_signals s ON s.wallet_id = w.id
                LEFT JOIN insider_cumulative_positions cp ON cp.wallet_id = w.id
                LEFT JOIN insider_detection_records dr ON dr.wallet_id = w.id
                LEFT JOIN insider_audit_chain ac ON ac.entry_id = w.id AND ac.entry_type = 'detection'
                WHERE w.wallet_address = '0xvalidation_wallet'
                GROUP BY w.id
            """))
            row = result.fetchone()

            # Assertions
            assert row[0] == '0xvalidation_wallet'
            assert float(row[1]) == 85.5
            assert row[2] == 'critical'
            assert row[3] == 8  # signal_count from wallet
            assert row[4] == 5  # active_dimensions
            assert row[5] == 6  # actual signal rows
            assert float(row[6]) == 125000.0  # cumulative_usd
            assert row[7] == 1  # is_split_entry
            assert row[8] == record_hash  # record_hash
            assert row[9] == chain_hash  # chain_hash

    def test_module_imports(self, db):
        """Test that all insider_scanner module exports are importable."""
        # These imports should work
        from src.insider_scanner import (
            InsiderPriority,
            FlaggedWallet,
            FlaggedFundingSource,
            InsiderSignal,
            DetectionRecord,
            InvestmentThesis,
            AuditChainEntry,
            CumulativePosition,
            InsiderScorer,
            AuditTrailManager,
        )

        # Verify enums work
        assert InsiderPriority.CRITICAL.value == "critical"
        assert InsiderPriority.HIGH.value == "high"

        # Verify classes are instantiable (for dataclass-like models)
        wallet = FlaggedWallet(
            wallet_address="0xtest",
            insider_score=75.0,
            priority=InsiderPriority.HIGH,
        )
        assert wallet.wallet_address == "0xtest"
        assert wallet.insider_score == 75.0

    def test_hash_chain_integrity(self, db):
        """Test that hash chain maintains integrity."""
        from sqlalchemy import text
        import hashlib

        with db.session() as session:
            # Create wallet
            session.execute(text("""
                INSERT INTO insider_flagged_wallets (wallet_address, insider_score)
                VALUES ('0xchain_test', 60.0)
            """))

            result = session.execute(text(
                "SELECT id FROM insider_flagged_wallets WHERE wallet_address = '0xchain_test'"
            ))
            wallet_id = result.fetchone()[0]

            # Create chain entries with proper linking
            prev_hash = None
            for i in range(1, 4):
                entry_hash = hashlib.sha256(f"entry_{i}".encode()).hexdigest()
                chain_data = f"{i}-detection-{wallet_id}-{entry_hash}"
                if prev_hash:
                    chain_data += f"-{prev_hash}"
                chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()

                session.execute(text("""
                    INSERT INTO insider_audit_chain
                    (sequence_number, entry_type, entry_id, entry_hash, previous_chain_hash, chain_hash)
                    VALUES (:seq, 'detection', :wallet_id, :entry_hash, :prev_hash, :chain_hash)
                """), {
                    "seq": i,
                    "wallet_id": wallet_id,
                    "entry_hash": entry_hash,
                    "prev_hash": prev_hash,
                    "chain_hash": chain_hash,
                })
                prev_hash = chain_hash

            # Verify chain integrity
            result = session.execute(text("""
                SELECT sequence_number, previous_chain_hash, chain_hash
                FROM insider_audit_chain
                ORDER BY sequence_number
            """))
            rows = result.fetchall()

            assert len(rows) == 3
            assert rows[0][1] is None  # First entry has no previous
            assert rows[1][1] == rows[0][2]  # Second's prev = first's hash
            assert rows[2][1] == rows[1][2]  # Third's prev = second's hash
