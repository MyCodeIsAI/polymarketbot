"""Stage 8 Tests: Audit Trail System for Legal Protection.

Critical tests ensuring the audit trail system:
1. Creates immutable detection records with proper hashing
2. Maintains tamper-evident hash chain
3. Links investment theses to detection records
4. Verifies chain integrity and detects tampering
5. Exports legal-ready documentation

These tests are CRITICAL for legal protection when acting on detection signals.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.database.connection import DatabaseConfig, DatabaseManager
from src.database.migrations import MigrationManager
from src.insider_scanner.migrations import register_insider_migrations
from src.insider_scanner.models import (
    FlaggedWallet,
    DetectionRecord,
    InvestmentThesis,
    AuditChainEntry,
    InsiderPriority,
    WalletStatus,
)
from src.insider_scanner.audit import AuditTrailManager
from src.insider_scanner.scoring import ScoringResult, Signal


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
def audit_manager(db):
    """Create an AuditTrailManager with a test session."""
    with db.session() as session:
        yield AuditTrailManager(session)


@pytest.fixture
def sample_wallet(db):
    """Create a sample flagged wallet for testing."""
    with db.session() as session:
        wallet = FlaggedWallet(
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            insider_score=85.5,
            priority=InsiderPriority.CRITICAL,
            status=WalletStatus.NEW,
            score_account=20.0,
            score_trading=30.0,
            score_behavioral=18.0,
            score_contextual=12.0,
            score_cluster=5.5,
            signal_count=8,
            active_dimensions=5,
            account_age_days=3,
        )
        session.add(wallet)
        session.commit()
        session.refresh(wallet)
        yield wallet


@pytest.fixture
def sample_scoring_result():
    """Create a sample scoring result for testing."""
    signals = [
        Signal(
            name="account_age",
            category="account",
            weight=20.0,
            raw_value=3,
            description="Account is 3 days old",
        ),
        Signal(
            name="perfect_win_rate",
            category="trading",
            weight=15.0,
            raw_value=1.0,
            description="100% win rate",
        ),
        Signal(
            name="single_market_focus",
            category="behavioral",
            weight=18.0,
            raw_value=1.0,
            description="100% concentration in one market",
        ),
        Signal(
            name="before_announcement",
            category="contextual",
            weight=12.0,
            raw_value=2,
            description="Entry 2 hours before announcement",
        ),
    ]

    return ScoringResult(
        score=85.5,
        confidence_low=75.5,
        confidence_high=95.5,
        priority="critical",
        dimensions={
            "account": 20.0,
            "trading": 30.0,
            "behavioral": 18.0,
            "contextual": 12.0,
            "cluster": 5.5,
        },
        signals=signals,
        signal_count=8,
        active_dimensions=5,
        downgraded=False,
        downgrade_reason=None,
    )


# =============================================================================
# Detection Record Tests
# =============================================================================


class TestDetectionRecordCreation:
    """Tests for creating detection records."""

    def test_create_detection_record_basic(self, db, sample_scoring_result):
        """Test creating a basic detection record."""
        with db.session() as session:
            # Create wallet first
            wallet = FlaggedWallet(
                wallet_address="0xtest_basic",
                insider_score=85.5,
                priority=InsiderPriority.CRITICAL,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            # Create audit manager and detection record
            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
                market_ids=["0xmarket123"],
            )
            session.commit()

            # Verify record created
            assert record.id is not None
            assert record.wallet_address == "0xtest_basic"
            assert record.insider_score == 85.5
            assert record.priority == "critical"
            assert record.record_hash is not None
            assert len(record.record_hash) == 64  # SHA-256 hex

    def test_detection_record_has_signals_snapshot(self, db, sample_scoring_result):
        """Test that detection record captures signals snapshot."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xtest_signals",
                insider_score=75.0,
                priority=InsiderPriority.HIGH,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            # Verify signals snapshot
            assert record.signals_snapshot is not None
            assert len(record.signals_snapshot) == 4
            assert record.signals_snapshot[0]["name"] == "account_age"
            assert record.signals_snapshot[1]["name"] == "perfect_win_rate"

    def test_detection_record_hash_deterministic(self, db, sample_scoring_result):
        """Test that the same inputs produce the same hash."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xtest_hash",
                insider_score=85.5,
                priority=InsiderPriority.CRITICAL,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)

            # Create first record
            record1 = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            hash1 = record1.calculate_hash()

            # Create second record with same data
            record2 = DetectionRecord(
                wallet_id=wallet.id,
                wallet_address=wallet.wallet_address,
                detected_at=record1.detected_at,
                insider_score=record1.insider_score,
                priority=record1.priority,
                signals_snapshot=record1.signals_snapshot,
            )
            hash2 = record2.calculate_hash()

            # Same inputs should produce same hash
            assert hash1 == hash2

    def test_detection_record_captures_market_data(self, db, sample_scoring_result):
        """Test that detection record captures market position data."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xtest_market",
                insider_score=80.0,
                priority=InsiderPriority.HIGH,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            market_positions = {
                "0xmarket123": {
                    "side": "YES",
                    "size_usd": 50000,
                    "entry_odds": 0.15,
                }
            }

            raw_api = {
                "timestamp": "2026-01-23T12:00:00Z",
                "source": "polymarket_api",
                "data": {"positions": market_positions},
            }

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
                market_ids=["0xmarket123"],
                market_positions=market_positions,
                raw_api_snapshot=raw_api,
            )
            session.commit()

            # Verify market data captured
            assert record.market_ids == ["0xmarket123"]
            assert record.market_positions["0xmarket123"]["size_usd"] == 50000
            assert record.raw_api_snapshot["source"] == "polymarket_api"

    def test_detection_records_link_with_previous_hash(self, db, sample_scoring_result):
        """Test that detection records form a hash chain."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Create first wallet and record
            wallet1 = FlaggedWallet(
                wallet_address="0xfirst",
                insider_score=80.0,
                priority=InsiderPriority.HIGH,
            )
            session.add(wallet1)
            session.commit()
            session.refresh(wallet1)

            record1 = audit.create_detection_record(
                wallet=wallet1,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            # First record should have no previous hash
            assert record1.previous_record_hash is None

            # Create second wallet and record
            wallet2 = FlaggedWallet(
                wallet_address="0xsecond",
                insider_score=75.0,
                priority=InsiderPriority.HIGH,
            )
            session.add(wallet2)
            session.commit()
            session.refresh(wallet2)

            record2 = audit.create_detection_record(
                wallet=wallet2,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            # Second record should reference first record's hash
            assert record2.previous_record_hash == record1.record_hash


# =============================================================================
# Hash Chain Tests
# =============================================================================


class TestAuditChain:
    """Tests for the audit chain integrity."""

    def test_chain_entry_created_with_detection(self, db, sample_scoring_result):
        """Test that creating a detection also creates a chain entry."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xchain_test",
                insider_score=85.0,
                priority=InsiderPriority.CRITICAL,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            # Check chain entry exists
            chain_entry = (
                session.query(AuditChainEntry)
                .filter_by(entry_type="detection", entry_id=record.id)
                .first()
            )

            assert chain_entry is not None
            assert chain_entry.entry_hash == record.record_hash
            assert chain_entry.sequence_number == 1

    def test_chain_links_entries_correctly(self, db, sample_scoring_result):
        """Test that chain entries link correctly."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Create multiple detections
            hashes = []
            for i in range(3):
                wallet = FlaggedWallet(
                    wallet_address=f"0xchain_{i}",
                    insider_score=70.0 + i * 5,
                    priority=InsiderPriority.HIGH,
                )
                session.add(wallet)
                session.commit()
                session.refresh(wallet)

                record = audit.create_detection_record(
                    wallet=wallet,
                    scoring_result=sample_scoring_result,
                )
                session.commit()
                hashes.append(record.record_hash)

            # Verify chain linkage
            entries = (
                session.query(AuditChainEntry)
                .order_by(AuditChainEntry.sequence_number.asc())
                .all()
            )

            assert len(entries) == 3
            assert entries[0].previous_chain_hash is None
            assert entries[1].previous_chain_hash == entries[0].chain_hash
            assert entries[2].previous_chain_hash == entries[1].chain_hash

    def test_chain_integrity_verification_passes(self, db, sample_scoring_result):
        """Test that valid chain passes integrity check."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Create valid chain
            for i in range(5):
                wallet = FlaggedWallet(
                    wallet_address=f"0xvalid_{i}",
                    insider_score=75.0,
                    priority=InsiderPriority.HIGH,
                )
                session.add(wallet)
                session.commit()
                session.refresh(wallet)

                audit.create_detection_record(
                    wallet=wallet,
                    scoring_result=sample_scoring_result,
                )
                session.commit()

            # Verify chain
            result = audit.verify_chain_integrity()

            assert result["valid"] is True
            assert result["entries_checked"] == 5
            assert len(result["errors"]) == 0

    def test_chain_integrity_detects_broken_link(self, db, sample_scoring_result):
        """Test that broken chain link is detected."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Create valid chain
            for i in range(3):
                wallet = FlaggedWallet(
                    wallet_address=f"0xbroken_{i}",
                    insider_score=75.0,
                    priority=InsiderPriority.HIGH,
                )
                session.add(wallet)
                session.commit()
                session.refresh(wallet)

                audit.create_detection_record(
                    wallet=wallet,
                    scoring_result=sample_scoring_result,
                )
                session.commit()

            # Tamper with middle entry's previous_chain_hash
            entry = (
                session.query(AuditChainEntry)
                .filter_by(sequence_number=2)
                .first()
            )
            entry.previous_chain_hash = "tampered_hash"
            session.commit()

            # Verify should detect the tampering
            result = audit.verify_chain_integrity()

            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert any("Chain link broken" in e["error"] for e in result["errors"])

    def test_chain_integrity_detects_hash_modification(self, db, sample_scoring_result):
        """Test that modified chain hash is detected."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Create valid chain
            for i in range(3):
                wallet = FlaggedWallet(
                    wallet_address=f"0xmodified_{i}",
                    insider_score=75.0,
                    priority=InsiderPriority.HIGH,
                )
                session.add(wallet)
                session.commit()
                session.refresh(wallet)

                audit.create_detection_record(
                    wallet=wallet,
                    scoring_result=sample_scoring_result,
                )
                session.commit()

            # Tamper with chain hash
            entry = (
                session.query(AuditChainEntry)
                .filter_by(sequence_number=2)
                .first()
            )
            entry.chain_hash = "modified_chain_hash"
            session.commit()

            # Verify should detect the tampering
            result = audit.verify_chain_integrity()

            assert result["valid"] is False
            assert len(result["errors"]) > 0


class TestDetectionRecordVerification:
    """Tests for verifying individual detection records."""

    def test_verify_valid_record(self, db, sample_scoring_result):
        """Test verification of a valid detection record."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xverify_valid",
                insider_score=80.0,
                priority=InsiderPriority.HIGH,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            # Verify record
            result = audit.verify_detection_record(record.id)

            assert result["valid"] is True
            assert result["hash_valid"] is True
            assert result["in_chain"] is True
            assert result["chain_hash_matches"] is True

    def test_verify_tampered_record_detected(self, db, sample_scoring_result):
        """Test that tampering with a record is detected."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xverify_tampered",
                insider_score=80.0,
                priority=InsiderPriority.HIGH,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            # Tamper with the record (change the score)
            record.insider_score = 95.0  # Changed from 80.0
            session.commit()

            # Verify should detect tampering
            result = audit.verify_detection_record(record.id)

            assert result["valid"] is False
            assert result["hash_valid"] is False  # Hash no longer matches content

    def test_verify_nonexistent_record(self, db):
        """Test verification of nonexistent record."""
        with db.session() as session:
            audit = AuditTrailManager(session)
            result = audit.verify_detection_record(99999)

            assert result["valid"] is False
            assert result["error"] == "Record not found"


# =============================================================================
# Investment Thesis Tests
# =============================================================================


class TestInvestmentThesis:
    """Tests for investment thesis documentation."""

    def test_create_investment_thesis(self, db, sample_scoring_result):
        """Test creating an investment thesis."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xthesis_test",
                insider_score=85.0,
                priority=InsiderPriority.CRITICAL,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)

            # Create detection first
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            # Create thesis linked to detection
            thesis = audit.create_investment_thesis(
                detection_record_ids=[record.id],
                reasoning="Based on 85+ score with fresh account and perfect win rate signals, "
                          "placing contrarian YES bet on the same market the insider is targeting.",
                intended_action="Buy YES",
                market_id="0xmarket123",
                position_side="YES",
                position_size=1000.0,
            )
            session.commit()

            # Verify thesis
            assert thesis.id is not None
            assert thesis.thesis_hash is not None
            assert thesis.detection_record_ids == [record.id]
            assert "85+ score" in thesis.reasoning

    def test_thesis_links_to_multiple_detections(self, db, sample_scoring_result):
        """Test thesis can link to multiple detection records."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Create multiple detections
            record_ids = []
            for i in range(3):
                wallet = FlaggedWallet(
                    wallet_address=f"0xmulti_{i}",
                    insider_score=75.0 + i * 5,
                    priority=InsiderPriority.HIGH,
                )
                session.add(wallet)
                session.commit()
                session.refresh(wallet)

                record = audit.create_detection_record(
                    wallet=wallet,
                    scoring_result=sample_scoring_result,
                )
                session.commit()
                record_ids.append(record.id)

            # Create thesis linking all detections
            thesis = audit.create_investment_thesis(
                detection_record_ids=record_ids,
                reasoning="Cluster of 3 related wallets detected, all showing insider patterns.",
                intended_action="Monitor and potentially fade",
            )
            session.commit()

            assert len(thesis.detection_record_ids) == 3
            assert thesis.detection_record_ids == record_ids

    def test_thesis_in_audit_chain(self, db, sample_scoring_result):
        """Test that thesis is added to audit chain."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xthesis_chain",
                insider_score=80.0,
                priority=InsiderPriority.HIGH,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)

            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            thesis = audit.create_investment_thesis(
                detection_record_ids=[record.id],
                reasoning="Test reasoning",
            )
            session.commit()

            # Check thesis is in chain
            chain_entry = (
                session.query(AuditChainEntry)
                .filter_by(entry_type="thesis", entry_id=thesis.id)
                .first()
            )

            assert chain_entry is not None
            assert chain_entry.entry_hash == thesis.thesis_hash

    def test_record_thesis_outcome(self, db, sample_scoring_result):
        """Test recording the outcome of an investment thesis."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xthesis_outcome",
                insider_score=85.0,
                priority=InsiderPriority.CRITICAL,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)

            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            session.commit()

            thesis = audit.create_investment_thesis(
                detection_record_ids=[record.id],
                reasoning="Test",
                position_size=1000.0,
            )
            session.commit()

            # Record outcome
            updated = audit.record_thesis_outcome(
                thesis_id=thesis.id,
                action_taken=True,
                actual_position_size=950.0,  # Slightly different due to slippage
                actual_outcome="Won $500 profit",
            )

            assert updated.action_taken is True
            assert updated.actual_position_size == 950.0
            assert "Won" in updated.actual_outcome


# =============================================================================
# Export Tests
# =============================================================================


class TestAuditExport:
    """Tests for exporting audit trail for legal purposes."""

    def test_export_detection_record(self, db, sample_scoring_result):
        """Test exporting a single detection record."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xexport_test",
                insider_score=85.0,
                priority=InsiderPriority.CRITICAL,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
                market_ids=["0xmarket"],
            )
            session.commit()

            # Export
            export = audit.export_detection_record(record.id)

            assert "record" in export
            assert "verification" in export
            assert "chain_position" in export
            assert "exported_at" in export
            assert export["record"]["wallet_address"] == "0xexport_test"
            assert export["verification"]["valid"] is True

    def test_export_full_audit_trail(self, db, sample_scoring_result):
        """Test exporting full audit trail."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Create multiple detections
            for i in range(3):
                wallet = FlaggedWallet(
                    wallet_address=f"0xfull_export_{i}",
                    insider_score=70.0 + i * 10,
                    priority=InsiderPriority.HIGH,
                )
                session.add(wallet)
                session.commit()
                session.refresh(wallet)

                record = audit.create_detection_record(
                    wallet=wallet,
                    scoring_result=sample_scoring_result,
                )
                session.commit()

            # Export full trail
            export = audit.export_full_audit_trail()

            assert export["record_count"] == 3
            assert len(export["detection_records"]) == 3
            assert export["chain_verification"]["valid"] is True
            assert "exported_at" in export

    def test_export_filtered_by_wallet(self, db, sample_scoring_result):
        """Test exporting audit trail filtered by wallet."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Create detections for different wallets
            for i in range(3):
                wallet = FlaggedWallet(
                    wallet_address=f"0xfiltered_{i}",
                    insider_score=75.0,
                    priority=InsiderPriority.HIGH,
                )
                session.add(wallet)
                session.commit()
                session.refresh(wallet)

                audit.create_detection_record(
                    wallet=wallet,
                    scoring_result=sample_scoring_result,
                )
                session.commit()

            # Export filtered
            export = audit.export_full_audit_trail(wallet_address="0xfiltered_1")

            assert export["record_count"] == 1
            assert export["detection_records"][0]["wallet_address"] == "0xfiltered_1"


# =============================================================================
# Integration Tests
# =============================================================================


class TestAuditTrailIntegration:
    """Integration tests for complete audit trail flow."""

    def test_complete_legal_protection_flow(self, db, sample_scoring_result):
        """Test complete flow from detection to legal documentation."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # Step 1: Detect suspicious wallet
            wallet = FlaggedWallet(
                wallet_address="0xsuspect_integration",
                insider_score=88.0,
                priority=InsiderPriority.CRITICAL,
                account_age_days=2,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            # Step 2: Create immutable detection record BEFORE taking action
            detection = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
                market_ids=["0xelection_market"],
                market_positions={
                    "0xelection_market": {
                        "side": "YES",
                        "size_usd": 125000,
                        "entry_odds": 0.08,
                    }
                },
                raw_api_snapshot={
                    "fetched_at": "2026-01-23T12:00:00Z",
                    "api_version": "v2",
                    "positions_response": "...",
                },
            )
            session.commit()
            detection_time = detection.detected_at

            # Step 3: Document investment thesis BEFORE placing bet
            thesis = audit.create_investment_thesis(
                detection_record_ids=[detection.id],
                reasoning=(
                    "Detected 88/100 insider score on wallet 0xsuspect_integration. "
                    "Wallet is 2 days old, has perfect win rate, and placed $125K on "
                    "election market at 8% odds. This indicates probable insider knowledge. "
                    "Taking contrarian position on YES side of same market based on "
                    "publicly observable detection signals."
                ),
                intended_action="Place YES bet on 0xelection_market",
                market_id="0xelection_market",
                position_side="YES",
                position_size=5000.0,
            )
            session.commit()
            thesis_time = thesis.created_at

            # Step 4: Record actual outcome
            thesis = audit.record_thesis_outcome(
                thesis_id=thesis.id,
                action_taken=True,
                actual_position_size=4800.0,  # After slippage
                actual_outcome="Market resolved YES. Profit: $8,500",
            )
            session.commit()

            # Step 5: Verify everything for legal export
            chain_valid = audit.verify_chain_integrity()
            detection_valid = audit.verify_detection_record(detection.id)
            export = audit.export_full_audit_trail(wallet_address="0xsuspect_integration")

            # Assertions for legal protection
            assert chain_valid["valid"] is True
            assert detection_valid["valid"] is True
            assert detection.detected_at < thesis.created_at  # Proves detection came first
            assert detection.record_hash is not None  # Immutable proof
            assert thesis.thesis_hash is not None  # Documented reasoning
            assert len(export["detection_records"]) == 1
            assert export["chain_verification"]["valid"] is True

    def test_audit_trail_survives_multiple_sessions(self, db, sample_scoring_result):
        """Test that audit trail persists across sessions."""
        # First session: create detection
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xpersist_test",
                insider_score=80.0,
                priority=InsiderPriority.HIGH,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=sample_scoring_result,
            )
            session.commit()
            record_id = record.id
            record_hash = record.record_hash

        # Second session: verify record still valid
        with db.session() as session:
            audit = AuditTrailManager(session)
            result = audit.verify_detection_record(record_id)

            assert result["valid"] is True
            assert result["record_hash"] == record_hash

    def test_multiple_detections_same_wallet_tracked(self, db, sample_scoring_result):
        """Test tracking multiple detections for the same wallet over time."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xrepeat_offender",
                insider_score=70.0,
                priority=InsiderPriority.MEDIUM,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            audit = AuditTrailManager(session)

            # Create multiple detections as score increases
            record_ids = []
            for i, score in enumerate([70.0, 78.0, 85.0]):
                # Update wallet score
                wallet.insider_score = score
                if score >= 85:
                    wallet.priority = InsiderPriority.CRITICAL
                session.commit()

                # Create detection record
                result = ScoringResult(
                    score=score,
                    confidence_low=score - 10.0,
                    confidence_high=min(score + 10.0, 100.0),
                    priority="critical" if score >= 85 else "high" if score >= 70 else "medium",
                    dimensions=sample_scoring_result.dimensions,
                    signals=sample_scoring_result.signals,
                    signal_count=sample_scoring_result.signal_count,
                    active_dimensions=sample_scoring_result.active_dimensions,
                    downgraded=False,
                    downgrade_reason=None,
                )

                record = audit.create_detection_record(
                    wallet=wallet,
                    scoring_result=result,
                )
                session.commit()
                record_ids.append(record.id)

            # Export trail for this wallet
            export = audit.export_full_audit_trail(wallet_address="0xrepeat_offender")

            assert export["record_count"] == 3
            # Scores should be in order
            scores = [r["insider_score"] for r in export["detection_records"]]
            assert scores == [70.0, 78.0, 85.0]


# =============================================================================
# Edge Cases
# =============================================================================


class TestAuditTrailEdgeCases:
    """Edge case tests for audit trail system."""

    def test_empty_chain_verification(self, db):
        """Test verifying empty audit chain."""
        with db.session() as session:
            audit = AuditTrailManager(session)
            result = audit.verify_chain_integrity()

            assert result["valid"] is True
            assert result["entries_checked"] == 0

    def test_detection_with_no_signals(self, db):
        """Test detection record with empty signals list."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xno_signals",
                insider_score=30.0,
                priority=InsiderPriority.LOW,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            empty_result = ScoringResult(
                score=30.0,
                confidence_low=20.0,
                confidence_high=40.0,
                priority="low",
                dimensions={},
                signals=[],
                signal_count=0,
                active_dimensions=0,
                downgraded=False,
                downgrade_reason=None,
            )

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=empty_result,
            )
            session.commit()

            assert record.signals_snapshot == []
            assert audit.verify_detection_record(record.id)["valid"] is True

    def test_thesis_with_invalid_detection_id(self, db, sample_scoring_result):
        """Test creating thesis with nonexistent detection ID."""
        with db.session() as session:
            audit = AuditTrailManager(session)

            # This should work but link to invalid IDs
            # The thesis itself is valid, just references missing records
            thesis = audit.create_investment_thesis(
                detection_record_ids=[99999, 99998],  # Nonexistent
                reasoning="Test with invalid IDs",
            )
            session.commit()

            assert thesis.id is not None
            # The thesis is created but references invalid detections

    def test_large_signals_snapshot(self, db):
        """Test detection with many signals."""
        with db.session() as session:
            wallet = FlaggedWallet(
                wallet_address="0xmany_signals",
                insider_score=95.0,
                priority=InsiderPriority.CRITICAL,
            )
            session.add(wallet)
            session.commit()
            session.refresh(wallet)

            # Create many signals
            signals = [
                Signal(
                    name=f"signal_{i}",
                    category="test",
                    weight=1.0,
                    raw_value=i,
                    description=f"Test signal {i}",
                )
                for i in range(50)
            ]

            large_result = ScoringResult(
                score=95.0,
                confidence_low=90.0,
                confidence_high=100.0,
                priority="critical",
                dimensions={"test": 50.0},
                signals=signals,
                signal_count=50,
                active_dimensions=1,
                downgraded=False,
                downgrade_reason=None,
            )

            audit = AuditTrailManager(session)
            record = audit.create_detection_record(
                wallet=wallet,
                scoring_result=large_result,
            )
            session.commit()

            assert len(record.signals_snapshot) == 50
            assert audit.verify_detection_record(record.id)["valid"] is True
