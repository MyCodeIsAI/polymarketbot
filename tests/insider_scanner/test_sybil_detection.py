"""Comprehensive tests for sybil detection system.

Tests cover:
1. Database models (KnownProfitableTrader, KnownFundingSource, etc.)
2. SybilDetector import functions
3. Sybil detection logic (funding source match, withdrawal dest match, chain link)
4. API endpoints for sybil detection
5. Integration tests with real data files
6. UI filter functionality (flag_type filtering)

Run with: pytest tests/insider_scanner/test_sybil_detection.py -v
"""

import json
import os
import pytest
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import models - MUST import ALL models so Base.metadata has all tables registered
from src.insider_scanner.models import (
    Base,
    FlagType,
    FlaggedWallet,
    InsiderPriority,
    WalletStatus,
    KnownProfitableTrader,
    KnownFundingSource,
    KnownWithdrawalDest,
    FundFlowEdge,
    SybilDetection,
    # Also import these so tables are created
    FlaggedFundingSource,
    InsiderCluster,
    DetectionRecord,
    InvestmentThesis,
)

# Import sybil detector
from src.insider_scanner.sybil_detector import SybilDetector, SybilMatch


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def db_engine():
    """Create in-memory SQLite database for testing."""
    # check_same_thread=False is needed for FastAPI TestClient which runs in different thread
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Create a database session for testing."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def session_factory(db_engine):
    """Create a session factory for SybilDetector."""
    Session = sessionmaker(bind=db_engine)
    return Session


@pytest.fixture
def sybil_detector(session_factory):
    """Create a SybilDetector instance for testing."""
    return SybilDetector(session_factory)


@pytest.fixture
def sample_profitable_wallets_json():
    """Sample profitable_wallets_full.json data."""
    return {
        "generated_at": "2026-01-23T18:00:00.000000",
        "total_wallets": 3,
        "wallets": [
            {
                "wallet": "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
                "profit_usd": 326147.99,
                "funding_source": "0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d",
                "funding_source_type": "eoa:USDC",
                "funding_timestamp": "2024-08-20T22:48:52",
                "funding_amount_matic": 13852.45,
                "primary_withdrawal_dest": "0xc5d563a36ae78145c45a50134d48a1215220f80a",
                "total_withdrawn_matic": 186122.72,
                "funding_count": 30,
                "withdrawal_count": 68
            },
            {
                "wallet": "0xcb3143ee858e14d0b3fe40ffeaea78416e646b02",
                "profit_usd": 425933.67,
                "funding_source": "0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215",
                "funding_source_type": "eoa:USDC.e",
                "funding_timestamp": "2025-05-09T10:52:11",
                "funding_amount_matic": 9.91,
                "primary_withdrawal_dest": "0xd91e80cf2e7be2e162c6513ced06f1dd0da35296",
                "total_withdrawn_matic": 70.0,
                "funding_count": 17,
                "withdrawal_count": 12
            },
            {
                "wallet": "0xf743f416caa37f672e8434a9132f681a8fa0ac84",
                "profit_usd": 292501.43,
                "funding_source": "0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215",
                "funding_source_type": "eoa:USDC.e",
                "funding_timestamp": "2025-02-17T13:39:21",
                "funding_amount_matic": 1234.68,
                "primary_withdrawal_dest": "0xd36ec33c8bed5a9f7b6630855f1533455b98a418",
                "total_withdrawn_matic": 356409.22,
                "funding_count": 9,
                "withdrawal_count": 88
            }
        ]
    }


@pytest.fixture
def sample_funding_sources_json():
    """Sample funding_sources.json data."""
    return {
        "generated_at": "2026-01-23T18:00:00.000000",
        "total_unique_sources": 2,
        "sources": [
            {
                "address": "0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215",
                "source_type": "eoa",
                "labels": [],
                "funded_wallets": [
                    "0xcb3143ee858e14d0b3fe40ffeaea78416e646b02",
                    "0xf743f416caa37f672e8434a9132f681a8fa0ac84"
                ],
                "total_profit_funded": 718435.11
            },
            {
                "address": "0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d",
                "source_type": "eoa",
                "labels": [],
                "funded_wallets": [
                    "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817"
                ],
                "total_profit_funded": 326147.99
            },
            {
                "address": "0xe7804c37c13166ff0b37f5ae0bb07a3aebb6e245",
                "source_type": "exchange",
                "labels": ["Binance Hot 2"],
                "funded_wallets": [
                    "0x5bffcf561bcae83af680ad600cb99f1184d6ffbe"
                ],
                "total_profit_funded": 254547.38
            }
        ]
    }


@pytest.fixture
def sample_withdrawal_destinations_json():
    """Sample withdrawal_destinations.json data."""
    return {
        "generated_at": "2026-01-23T18:00:00.000000",
        "total_unique_destinations": 2,
        "destinations": [
            {
                "address": "0xc5d563a36ae78145c45a50134d48a1215220f80a",
                "dest_type": "eoa",
                "labels": [],
                "received_from_traders": [
                    "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817"
                ],
                "total_profit_source": 326147.99,
                "also_funded_traders": []
            },
            {
                "address": "0xbridge1234567890abcdef1234567890abcdef12",
                "dest_type": "eoa",
                "labels": [],
                "received_from_traders": [
                    "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817"
                ],
                "total_profit_source": 326147.99,
                "also_funded_traders": [
                    "0xnewtrader12345678901234567890123456789012"
                ]
            }
        ]
    }


@pytest.fixture
def sample_fund_flow_graph_json():
    """Sample fund_flow_graph.json data."""
    return {
        "generated_at": "2026-01-23T18:00:00.000000",
        "edges": [
            {
                "from": "0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d",
                "to": "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
                "value_matic": 13852.45,
                "tx_hash": "0xabc123",
                "timestamp": "2024-08-20T22:48:52",
                "type": "funding"
            },
            {
                "from": "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
                "to": "0xc5d563a36ae78145c45a50134d48a1215220f80a",
                "value_matic": 186122.72,
                "tx_hash": "0xdef456",
                "timestamp": "2024-09-15T10:00:00",
                "type": "withdrawal"
            }
        ]
    }


# =============================================================================
# MODEL TESTS
# =============================================================================


class TestFlagTypeEnum:
    """Tests for FlagType enum."""

    def test_flag_type_values(self):
        """Test FlagType enum has correct values."""
        assert FlagType.INSIDER.value == "insider"
        assert FlagType.SYBIL.value == "sybil"
        assert FlagType.UNKNOWN.value == "unknown"

    def test_flag_type_from_string(self):
        """Test creating FlagType from string."""
        assert FlagType("insider") == FlagType.INSIDER
        assert FlagType("sybil") == FlagType.SYBIL
        assert FlagType("unknown") == FlagType.UNKNOWN

    def test_flag_type_invalid_raises(self):
        """Test invalid FlagType string raises error."""
        with pytest.raises(ValueError):
            FlagType("invalid")


class TestKnownProfitableTrader:
    """Tests for KnownProfitableTrader model."""

    def test_create_trader(self, db_session):
        """Test creating a KnownProfitableTrader."""
        trader = KnownProfitableTrader(
            wallet_address="0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
            profit_usd=Decimal("326147.99"),
            funding_source="0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d",
            funding_source_type="eoa:USDC",
            primary_withdrawal_dest="0xc5d563a36ae78145c45a50134d48a1215220f80a",
        )
        db_session.add(trader)
        db_session.commit()

        assert trader.id is not None
        assert trader.wallet_address == "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817"
        assert float(trader.profit_usd) == 326147.99

    def test_trader_to_dict(self, db_session):
        """Test KnownProfitableTrader.to_dict()."""
        trader = KnownProfitableTrader(
            wallet_address="0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
            profit_usd=Decimal("326147.99"),
        )
        db_session.add(trader)
        db_session.commit()

        d = trader.to_dict()
        assert d["wallet_address"] == "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817"
        assert d["profit_usd"] == 326147.99

    def test_trader_unique_constraint(self, db_session):
        """Test wallet address unique constraint."""
        trader1 = KnownProfitableTrader(
            wallet_address="0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
            profit_usd=Decimal("100000"),
        )
        trader2 = KnownProfitableTrader(
            wallet_address="0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
            profit_usd=Decimal("200000"),
        )
        db_session.add(trader1)
        db_session.commit()

        db_session.add(trader2)
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()


class TestKnownFundingSource:
    """Tests for KnownFundingSource model."""

    def test_create_funding_source(self, db_session):
        """Test creating a KnownFundingSource."""
        source = KnownFundingSource(
            address="0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215",
            source_type="eoa",
            funded_trader_wallets=["0xcb3143ee858e14d0b3fe40ffeaea78416e646b02"],
            funded_trader_count=1,
            total_profit_funded=Decimal("425933.67"),
            is_exchange=False,
        )
        db_session.add(source)
        db_session.commit()

        assert source.id is not None
        assert source.funded_trader_count == 1
        assert not source.is_exchange

    def test_funding_source_exchange_flag(self, db_session):
        """Test exchange flag on funding source."""
        source = KnownFundingSource(
            address="0xe7804c37c13166ff0b37f5ae0bb07a3aebb6e245",
            source_type="exchange",
            labels=["Binance Hot 2"],
            funded_trader_wallets=["0x5bffcf561bcae83af680ad600cb99f1184d6ffbe"],
            funded_trader_count=1,
            is_exchange=True,
            risk_score=20,
        )
        db_session.add(source)
        db_session.commit()

        assert source.is_exchange is True
        assert source.risk_score == 20
        assert "Binance Hot 2" in source.labels


class TestKnownWithdrawalDest:
    """Tests for KnownWithdrawalDest model."""

    def test_create_withdrawal_dest(self, db_session):
        """Test creating a KnownWithdrawalDest."""
        dest = KnownWithdrawalDest(
            address="0xc5d563a36ae78145c45a50134d48a1215220f80a",
            dest_type="eoa",
            received_from_traders=["0xf2f6af4f27ec2dcf4072095ab804016e14cd5817"],
            received_from_count=1,
            total_profit_source=Decimal("326147.99"),
            is_bridge_wallet=False,
        )
        db_session.add(dest)
        db_session.commit()

        assert dest.id is not None
        assert not dest.is_bridge_wallet

    def test_withdrawal_dest_bridge_wallet(self, db_session):
        """Test bridge wallet detection."""
        dest = KnownWithdrawalDest(
            address="0xbridge1234567890abcdef1234567890abcdef12",
            dest_type="eoa",
            received_from_traders=["0xf2f6af4f27ec2dcf4072095ab804016e14cd5817"],
            received_from_count=1,
            also_funded_traders=["0xnewtrader12345678901234567890123456789012"],
            is_bridge_wallet=True,
        )
        db_session.add(dest)
        db_session.commit()

        assert dest.is_bridge_wallet is True
        assert len(dest.also_funded_traders) == 1


class TestFlaggedWalletWithFlagType:
    """Tests for FlaggedWallet with flag_type field."""

    def test_flagged_wallet_default_flag_type(self, db_session):
        """Test default flag_type is UNKNOWN."""
        wallet = FlaggedWallet(
            wallet_address="0x1234567890123456789012345678901234567890",
            insider_score=Decimal("75"),
            priority=InsiderPriority.HIGH,
            status=WalletStatus.NEW,
        )
        db_session.add(wallet)
        db_session.commit()

        assert wallet.flag_type == FlagType.UNKNOWN

    def test_flagged_wallet_sybil_flag(self, db_session):
        """Test setting SYBIL flag type."""
        wallet = FlaggedWallet(
            wallet_address="0x1234567890123456789012345678901234567890",
            insider_score=Decimal("75"),
            priority=InsiderPriority.HIGH,
            status=WalletStatus.NEW,
            flag_type=FlagType.SYBIL,
            funding_match_address="0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d",
        )
        db_session.add(wallet)
        db_session.commit()

        assert wallet.flag_type == FlagType.SYBIL
        assert wallet.funding_match_address == "0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d"

    def test_flagged_wallet_insider_flag(self, db_session):
        """Test setting INSIDER flag type."""
        wallet = FlaggedWallet(
            wallet_address="0x1234567890123456789012345678901234567890",
            insider_score=Decimal("85"),
            priority=InsiderPriority.CRITICAL,
            status=WalletStatus.NEW,
            flag_type=FlagType.INSIDER,
        )
        db_session.add(wallet)
        db_session.commit()

        assert wallet.flag_type == FlagType.INSIDER

    def test_flagged_wallet_to_dict_includes_flag_type(self, db_session):
        """Test to_dict includes flag_type."""
        wallet = FlaggedWallet(
            wallet_address="0x1234567890123456789012345678901234567890",
            insider_score=Decimal("75"),
            priority=InsiderPriority.HIGH,
            status=WalletStatus.NEW,
            flag_type=FlagType.SYBIL,
            funding_match_address="0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d",
        )
        db_session.add(wallet)
        db_session.commit()

        d = wallet.to_dict()
        assert d["flag_type"] == "sybil"
        assert d["funding_match_address"] == "0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d"


class TestSybilDetectionModel:
    """Tests for SybilDetection model."""

    def test_create_sybil_detection(self, db_session):
        """Test creating a SybilDetection record."""
        detection = SybilDetection(
            new_wallet_address="0xnewwallet1234567890123456789012345678901",
            linked_trader_wallet="0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
            linked_trader_profit=Decimal("326147.99"),
            match_type="funding_source",
            match_address="0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d",
        )
        db_session.add(detection)
        db_session.commit()

        assert detection.id is not None
        assert detection.match_type == "funding_source"

    def test_sybil_detection_chain_link(self, db_session):
        """Test SybilDetection with chain_path."""
        detection = SybilDetection(
            new_wallet_address="0xnewwallet1234567890123456789012345678901",
            linked_trader_wallet="0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
            match_type="chain_link",
            match_address="0xbridge1234567890abcdef1234567890abcdef12",
            chain_path=[
                "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817",
                "0xbridge1234567890abcdef1234567890abcdef12",
                "0xnewwallet1234567890123456789012345678901"
            ],
        )
        db_session.add(detection)
        db_session.commit()

        assert detection.chain_path is not None
        assert len(detection.chain_path) == 3


# =============================================================================
# SYBIL DETECTOR TESTS
# =============================================================================


class TestSybilDetectorImport:
    """Tests for SybilDetector import functions."""

    def test_import_profitable_wallets(self, sybil_detector, session_factory, sample_profitable_wallets_json):
        """Test importing profitable_wallets_full.json."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_profitable_wallets_json, f)
            f.flush()

            try:
                stats = sybil_detector.import_profitable_wallets(f.name)

                assert stats["imported"] == 3
                assert stats["skipped"] == 0

                # Verify in database
                with session_factory() as session:
                    traders = session.query(KnownProfitableTrader).all()
                    assert len(traders) == 3

                    # Check specific trader
                    trader = session.query(KnownProfitableTrader).filter(
                        KnownProfitableTrader.wallet_address == "0xf2f6af4f27ec2dcf4072095ab804016e14cd5817"
                    ).first()
                    assert trader is not None
                    assert float(trader.profit_usd) == 326147.99
            finally:
                os.unlink(f.name)

    def test_import_funding_sources(self, sybil_detector, session_factory, sample_funding_sources_json):
        """Test importing funding_sources.json."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()

            try:
                stats = sybil_detector.import_funding_sources(f.name)

                assert stats["imported"] == 3
                assert stats["skipped"] == 0

                # Verify exchange flag
                with session_factory() as session:
                    exchange = session.query(KnownFundingSource).filter(
                        KnownFundingSource.address == "0xe7804c37c13166ff0b37f5ae0bb07a3aebb6e245"
                    ).first()
                    assert exchange is not None
                    assert exchange.is_exchange is True

                    # Check non-exchange
                    eoa = session.query(KnownFundingSource).filter(
                        KnownFundingSource.address == "0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215"
                    ).first()
                    assert eoa is not None
                    assert eoa.is_exchange is False
            finally:
                os.unlink(f.name)

    def test_import_withdrawal_destinations(self, sybil_detector, session_factory, sample_withdrawal_destinations_json):
        """Test importing withdrawal_destinations.json."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_withdrawal_destinations_json, f)
            f.flush()

            try:
                stats = sybil_detector.import_withdrawal_destinations(f.name)

                assert stats["imported"] == 2

                # Verify bridge wallet
                with session_factory() as session:
                    bridge = session.query(KnownWithdrawalDest).filter(
                        KnownWithdrawalDest.address == "0xbridge1234567890abcdef1234567890abcdef12"
                    ).first()
                    assert bridge is not None
                    assert bridge.is_bridge_wallet is True
            finally:
                os.unlink(f.name)

    def test_import_fund_flow_edges(self, sybil_detector, session_factory, sample_fund_flow_graph_json):
        """Test importing fund_flow_graph.json edges."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_fund_flow_graph_json, f)
            f.flush()

            try:
                stats = sybil_detector.import_fund_flow_edges(f.name)

                assert stats["imported"] == 2

                # Verify edges
                with session_factory() as session:
                    edges = session.query(FundFlowEdge).all()
                    assert len(edges) == 2

                    funding = session.query(FundFlowEdge).filter(
                        FundFlowEdge.edge_type == "funding"
                    ).first()
                    assert funding is not None
                    assert funding.from_address == "0xdd42ffb8aabe818f7538d93c175a9f9e2da9990d"
            finally:
                os.unlink(f.name)

    def test_import_file_not_found(self, sybil_detector):
        """Test import with non-existent file."""
        with pytest.raises(FileNotFoundError):
            sybil_detector.import_profitable_wallets("/nonexistent/file.json")


class TestSybilDetectorRebuildIndex:
    """Tests for SybilDetector.rebuild_index()."""

    def test_rebuild_index_empty(self, sybil_detector):
        """Test rebuild_index with empty database."""
        stats = sybil_detector.rebuild_index()

        assert stats["funding_sources"] == 0
        assert stats["withdrawal_dests"] == 0
        assert stats["bridge_wallets"] == 0

    def test_rebuild_index_with_data(self, sybil_detector, session_factory, sample_funding_sources_json, sample_withdrawal_destinations_json):
        """Test rebuild_index after importing data."""
        # Import funding sources
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        # Import withdrawal destinations
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_withdrawal_destinations_json, f)
            f.flush()
            sybil_detector.import_withdrawal_destinations(f.name)
            os.unlink(f.name)

        # Rebuild index
        stats = sybil_detector.rebuild_index()

        # 3 funding sources total, but 1 is exchange (excluded from index)
        assert stats["funding_sources"] == 2  # 2 non-exchange funding sources indexed
        assert stats["withdrawal_dests"] == 2  # 2 withdrawal destinations
        assert stats["exchanges"] == 1  # 1 exchange (Binance) tracked but not indexed
        assert "high_volume_excluded" in stats  # New field for high-volume sources


class TestSybilDetectorCheckFundingSource:
    """Tests for SybilDetector.check_funding_source()."""

    def test_check_funding_source_match(self, sybil_detector, session_factory, sample_funding_sources_json, sample_profitable_wallets_json):
        """Test detecting a funding source match."""
        # Import data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_profitable_wallets_json, f)
            f.flush()
            sybil_detector.import_profitable_wallets(f.name)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Check known funding source
        match = sybil_detector.check_funding_source("0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215")

        assert match.matched is True
        assert match.match_type == "funding_source"
        assert match.confidence >= 0.9

    def test_check_funding_source_no_match(self, sybil_detector, session_factory, sample_funding_sources_json):
        """Test checking an unknown funding source."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Check unknown address
        match = sybil_detector.check_funding_source("0xunknown1234567890123456789012345678901234")

        assert match.matched is False

    def test_check_funding_source_exchange_skipped(self, sybil_detector, session_factory, sample_funding_sources_json):
        """Test that exchange addresses are skipped (less suspicious)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Check exchange address (Binance)
        match = sybil_detector.check_funding_source("0xe7804c37c13166ff0b37f5ae0bb07a3aebb6e245")

        # Exchange addresses are skipped to avoid false positives
        assert match.matched is False

    def test_check_funding_source_case_insensitive(self, sybil_detector, session_factory, sample_funding_sources_json, sample_profitable_wallets_json):
        """Test that address matching is case-insensitive."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_profitable_wallets_json, f)
            f.flush()
            sybil_detector.import_profitable_wallets(f.name)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Check with uppercase
        match = sybil_detector.check_funding_source("0x1929347E025D4F5F8D6B2BD2261E2F4EFCACD215")

        assert match.matched is True

    def test_check_withdrawal_dest_match(self, sybil_detector, session_factory, sample_withdrawal_destinations_json, sample_profitable_wallets_json):
        """Test detecting a withdrawal destination match (chain link)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_profitable_wallets_json, f)
            f.flush()
            sybil_detector.import_profitable_wallets(f.name)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_withdrawal_destinations_json, f)
            f.flush()
            sybil_detector.import_withdrawal_destinations(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Check withdrawal destination
        match = sybil_detector.check_funding_source("0xc5d563a36ae78145c45a50134d48a1215220f80a")

        assert match.matched is True
        assert match.match_type == "withdrawal_dest"
        assert match.chain_path is not None


class TestSybilDetectorFlagWallet:
    """Tests for SybilDetector.flag_sybil_wallet()."""

    def test_flag_sybil_wallet(self, sybil_detector, session_factory, sample_funding_sources_json, sample_profitable_wallets_json):
        """Test flagging a new wallet as sybil."""
        # Import data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_profitable_wallets_json, f)
            f.flush()
            sybil_detector.import_profitable_wallets(f.name)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Flag a new wallet funded by known source
        flagged = sybil_detector.flag_sybil_wallet(
            new_wallet="0xnewwallet1234567890123456789012345678901",
            funding_address="0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215",
            funding_amount_matic=Decimal("100"),
        )

        assert flagged is not None

        # Verify flagged wallet in database (access within session to avoid DetachedInstanceError)
        with session_factory() as session:
            wallet = session.query(FlaggedWallet).filter(
                FlaggedWallet.wallet_address == "0xnewwallet1234567890123456789012345678901"
            ).first()
            assert wallet is not None
            assert wallet.flag_type == FlagType.SYBIL
            assert wallet.funding_match_address == "0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215"

            # Verify SybilDetection record was created
            detection = session.query(SybilDetection).filter(
                SybilDetection.new_wallet_address == "0xnewwallet1234567890123456789012345678901"
            ).first()
            assert detection is not None
            assert detection.match_type == "funding_source"

    def test_flag_sybil_wallet_no_match(self, sybil_detector, session_factory, sample_funding_sources_json):
        """Test flagging fails when no match."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Try to flag with unknown funding source
        flagged = sybil_detector.flag_sybil_wallet(
            new_wallet="0xnewwallet1234567890123456789012345678901",
            funding_address="0xunknown1234567890123456789012345678901234",
        )

        assert flagged is None


class TestSybilDetectorStats:
    """Tests for SybilDetector.get_stats()."""

    def test_get_stats_empty(self, sybil_detector):
        """Test get_stats with empty database."""
        stats = sybil_detector.get_stats()

        assert stats["known_profitable_traders"] == 0
        assert stats["known_funding_sources"] == 0
        assert stats["sybil_detections"] == 0

    def test_get_stats_with_data(self, sybil_detector, session_factory, sample_profitable_wallets_json, sample_funding_sources_json):
        """Test get_stats after importing data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_profitable_wallets_json, f)
            f.flush()
            sybil_detector.import_profitable_wallets(f.name)
            os.unlink(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        stats = sybil_detector.get_stats()

        assert stats["known_profitable_traders"] == 3
        assert stats["known_funding_sources"] == 3  # Total in database
        assert stats["indexed_funding_sources"] == 2  # Exchanges are not indexed (filtered out)
        assert "high_volume_sources_excluded" in stats  # New field
        assert "max_funded_traders_threshold" in stats  # New config field


# =============================================================================
# INTEGRATION TESTS WITH REAL DATA
# =============================================================================


class TestIntegrationWithRealData:
    """Integration tests using real data files."""

    @pytest.fixture
    def real_data_path(self):
        """Path to real data files."""
        # Get project root from test file location (tests/insider_scanner/)
        return Path(__file__).parent.parent.parent / "data"

    def test_import_real_profitable_wallets(self, sybil_detector, session_factory, real_data_path):
        """Test importing real profitable_wallets_full.json."""
        json_file = real_data_path / "profitable_wallets_full.json"
        if not json_file.exists():
            pytest.skip("Real data file not found")

        stats = sybil_detector.import_profitable_wallets(str(json_file))

        assert stats["imported"] > 0 or stats["updated"] > 0
        print(f"Imported {stats['imported']} traders, updated {stats['updated']}")

        # Verify top traders
        with session_factory() as session:
            top_trader = session.query(KnownProfitableTrader).order_by(
                KnownProfitableTrader.profit_usd.desc()
            ).first()
            assert top_trader is not None
            print(f"Top trader: {top_trader.wallet_address[:16]}... profit=${float(top_trader.profit_usd):,.2f}")

    def test_import_real_funding_sources(self, sybil_detector, session_factory, real_data_path):
        """Test importing real funding_sources.json."""
        json_file = real_data_path / "funding_sources.json"
        if not json_file.exists():
            pytest.skip("Real data file not found")

        stats = sybil_detector.import_funding_sources(str(json_file))

        assert stats["imported"] > 0 or stats["updated"] > 0
        print(f"Imported {stats['imported']} funding sources")

    def test_import_real_withdrawal_destinations(self, sybil_detector, session_factory, real_data_path):
        """Test importing real withdrawal_destinations.json."""
        json_file = real_data_path / "withdrawal_destinations.json"
        if not json_file.exists():
            pytest.skip("Real data file not found")

        stats = sybil_detector.import_withdrawal_destinations(str(json_file))

        assert stats["imported"] > 0 or stats["updated"] > 0
        print(f"Imported {stats['imported']} withdrawal destinations")

    def test_full_pipeline_with_real_data(self, sybil_detector, session_factory, real_data_path):
        """Test complete sybil detection pipeline with real data."""
        # Import all data files
        files_imported = 0

        for file_name, import_func in [
            ("profitable_wallets_full.json", sybil_detector.import_profitable_wallets),
            ("funding_sources.json", sybil_detector.import_funding_sources),
            ("withdrawal_destinations.json", sybil_detector.import_withdrawal_destinations),
        ]:
            json_file = real_data_path / file_name
            if json_file.exists():
                import_func(str(json_file))
                files_imported += 1

        if files_imported == 0:
            pytest.skip("No real data files found")

        # Rebuild index
        index_stats = sybil_detector.rebuild_index()
        print(f"Index stats: {index_stats}")

        # Test detection with a known funding source from real data
        with session_factory() as session:
            # Get a non-exchange funding source
            source = session.query(KnownFundingSource).filter(
                KnownFundingSource.is_exchange == False
            ).first()

            if source:
                match = sybil_detector.check_funding_source(source.address)
                print(f"Testing source: {source.address[:16]}...")
                print(f"Match result: {match.matched}, type: {match.match_type}")
                assert match.matched is True

    def test_detect_cluster_from_real_data(self, sybil_detector, session_factory, real_data_path):
        """Test that we can detect the known cluster in real data.

        From cluster_report.txt:
        CLUSTER 1: SHARED_FUNDER
          Bridge Address: 0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215
          Traders funded by this address:
            - 0xcb3143ee858e14d0b3fe40ffeaea78416e646b02 ($425,934)
            - 0xf743f416caa37f672e8434a9132f681a8fa0ac84 ($292,501)
        """
        json_file = real_data_path / "funding_sources.json"
        if not json_file.exists():
            pytest.skip("Real data file not found")

        sybil_detector.import_funding_sources(str(json_file))
        sybil_detector.rebuild_index()

        # Check that the shared funder from cluster 1 is indexed
        match = sybil_detector.check_funding_source("0x1929347e025d4f5f8d6b2bd2261e2f4efcacd215")

        assert match.matched is True
        print(f"Cluster detection: matched={match.matched}, type={match.match_type}")


# =============================================================================
# API ENDPOINT TESTS
# =============================================================================


class TestSybilAPIEndpoints:
    """Tests for sybil detection API endpoints.

    Note: The actual get_db dependency returns a DatabaseManager, but the routes
    use it like a session (db.query()). For testing, we override get_db to return
    an actual session.
    """

    @pytest.fixture
    def api_test_env(self):
        """Create complete API test environment with client and session factory."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        # Import ALL models to ensure they're registered with Base
        from src.insider_scanner import models as insider_models
        from src.database import models as db_models
        from sqlalchemy.orm import scoped_session
        from sqlalchemy.pool import StaticPool
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.web.api.insider_routes import router
        from src.web.api.dependencies import get_db

        # Create engine with StaticPool to share in-memory database across threads
        # This is essential for TestClient which runs in a different thread
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,  # Share connection across threads
        )
        # Create all tables - this must happen AFTER importing all models
        db_models.Base.metadata.create_all(engine)

        # Create scoped session factory for thread-safety
        Session = sessionmaker(bind=engine)
        session_factory = scoped_session(Session)

        # Create FastAPI app
        app = FastAPI()
        app.include_router(router, prefix="/api")

        # Override get_db to yield a proper session (not DatabaseManager)
        # This fixes the mismatch between get_db returning DatabaseManager
        # and routes using it like a session with db.query()
        def override_get_db():
            session = session_factory()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        # Create client
        client = TestClient(app)
        yield client, session_factory, engine

        # Cleanup
        client.close()
        session_factory.remove()
        engine.dispose()

    @pytest.fixture
    def test_client(self, api_test_env):
        """Get test client from environment."""
        client, _, _ = api_test_env
        return client

    @pytest.fixture
    def api_session_factory(self, api_test_env):
        """Get session factory from environment."""
        _, session_factory, _ = api_test_env
        return session_factory

    def test_get_sybil_stats_empty(self, test_client):
        """Test GET /sybil/stats with empty database."""
        response = test_client.get("/api/insider/sybil/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["known_profitable_traders"] == 0
        assert data["sybil_flagged_wallets"] == 0

    def test_list_watchlist_with_flag_type_filter(self, test_client, api_session_factory):
        """Test GET /watchlist with flag_type filter."""
        # Create test wallets using scoped session
        session = api_session_factory()
        try:
            sybil_wallet = FlaggedWallet(
                wallet_address="0xsybil12345678901234567890123456789012345",
                insider_score=Decimal("75"),
                priority=InsiderPriority.HIGH,
                status=WalletStatus.NEW,
                flag_type=FlagType.SYBIL,
            )
            insider_wallet = FlaggedWallet(
                wallet_address="0xinsider2345678901234567890123456789012345",
                insider_score=Decimal("85"),
                priority=InsiderPriority.CRITICAL,
                status=WalletStatus.NEW,
                flag_type=FlagType.INSIDER,
            )
            session.add_all([sybil_wallet, insider_wallet])
            session.commit()
        finally:
            api_session_factory.remove()

        # Test filter by sybil
        response = test_client.get("/api/insider/watchlist?flag_type=sybil")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["flag_type"] == "sybil"

        # Test filter by insider
        response = test_client.get("/api/insider/watchlist?flag_type=insider")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["flag_type"] == "insider"

        # Test no filter
        response = test_client.get("/api/insider/watchlist")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_wallet_detail_with_linked_trader(self, test_client, api_session_factory):
        """Test GET /watchlist/{id} includes linked trader info."""
        session = api_session_factory()
        try:
            # Create linked trader
            trader = KnownProfitableTrader(
                wallet_address="0xtrader12345678901234567890123456789012345",
                profit_usd=Decimal("326147.99"),
            )
            session.add(trader)
            session.commit()
            trader_id = trader.id

            # Create sybil wallet linked to trader
            wallet = FlaggedWallet(
                wallet_address="0xsybil12345678901234567890123456789012345",
                insider_score=Decimal("75"),
                priority=InsiderPriority.HIGH,
                status=WalletStatus.NEW,
                flag_type=FlagType.SYBIL,
                linked_trader_id=trader_id,
                funding_match_address="0xfunding234567890123456789012345678901234",
            )
            session.add(wallet)
            session.commit()
            wallet_id = wallet.id
        finally:
            api_session_factory.remove()

        # Get wallet detail
        response = test_client.get(f"/api/insider/watchlist/{wallet_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["flag_type"] == "sybil"
        assert data["linked_trader_id"] == trader_id
        assert data["linked_trader_wallet"] == "0xtrader12345678901234567890123456789012345"
        assert data["linked_trader_profit"] == 326147.99
        assert data["funding_match_address"] == "0xfunding234567890123456789012345678901234"


# =============================================================================
# UI FILTER TESTS
# =============================================================================


class TestUIFlagTypeFilter:
    """Tests for UI flag type filtering logic."""

    def test_filter_wallets_by_flag_type_sql(self, db_session):
        """Test SQL filtering by flag_type."""
        # Create test wallets
        wallets = [
            FlaggedWallet(wallet_address="0xa1", insider_score=Decimal("80"), priority=InsiderPriority.HIGH, status=WalletStatus.NEW, flag_type=FlagType.SYBIL),
            FlaggedWallet(wallet_address="0xa2", insider_score=Decimal("85"), priority=InsiderPriority.CRITICAL, status=WalletStatus.NEW, flag_type=FlagType.INSIDER),
            FlaggedWallet(wallet_address="0xa3", insider_score=Decimal("70"), priority=InsiderPriority.HIGH, status=WalletStatus.NEW, flag_type=FlagType.UNKNOWN),
            FlaggedWallet(wallet_address="0xa4", insider_score=Decimal("75"), priority=InsiderPriority.HIGH, status=WalletStatus.NEW, flag_type=FlagType.SYBIL),
        ]
        for w in wallets:
            db_session.add(w)
        db_session.commit()

        # Filter by SYBIL
        sybil_wallets = db_session.query(FlaggedWallet).filter(
            FlaggedWallet.flag_type == FlagType.SYBIL
        ).all()
        assert len(sybil_wallets) == 2

        # Filter by INSIDER
        insider_wallets = db_session.query(FlaggedWallet).filter(
            FlaggedWallet.flag_type == FlagType.INSIDER
        ).all()
        assert len(insider_wallets) == 1

        # Filter by UNKNOWN
        unknown_wallets = db_session.query(FlaggedWallet).filter(
            FlaggedWallet.flag_type == FlagType.UNKNOWN
        ).all()
        assert len(unknown_wallets) == 1

    def test_count_wallets_by_flag_type(self, db_session):
        """Test counting wallets by flag type for stats display."""
        # Create test wallets
        for i in range(5):
            db_session.add(FlaggedWallet(
                wallet_address=f"0xsybil{i:039d}",
                insider_score=Decimal("75"),
                priority=InsiderPriority.HIGH,
                status=WalletStatus.NEW,
                flag_type=FlagType.SYBIL,
            ))
        for i in range(3):
            db_session.add(FlaggedWallet(
                wallet_address=f"0xinsider{i:037d}",
                insider_score=Decimal("85"),
                priority=InsiderPriority.CRITICAL,
                status=WalletStatus.NEW,
                flag_type=FlagType.INSIDER,
            ))
        db_session.commit()

        # Count by flag type
        sybil_count = db_session.query(FlaggedWallet).filter(
            FlaggedWallet.flag_type == FlagType.SYBIL
        ).count()
        insider_count = db_session.query(FlaggedWallet).filter(
            FlaggedWallet.flag_type == FlagType.INSIDER
        ).count()

        assert sybil_count == 5
        assert insider_count == 3


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_wallet_address(self, db_session):
        """Test handling of empty wallet address in import."""
        data = {
            "wallets": [
                {"wallet": "", "profit_usd": 100000},
                {"wallet": "0xvalid123456789012345678901234567890123456", "profit_usd": 200000},
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                Session = sessionmaker(bind=db_session.get_bind())
                detector = SybilDetector(Session)
                stats = detector.import_profitable_wallets(f.name)

                assert stats["skipped"] == 1
                assert stats["imported"] == 1
            finally:
                os.unlink(f.name)

    def test_null_funding_source(self, sybil_detector, session_factory):
        """Test wallet with null funding source."""
        data = {
            "wallets": [
                {
                    "wallet": "0xnofunding1234567890123456789012345678901",
                    "profit_usd": 100000,
                    "funding_source": None,
                    "primary_withdrawal_dest": None,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                stats = sybil_detector.import_profitable_wallets(f.name)
                assert stats["imported"] == 1

                with session_factory() as session:
                    trader = session.query(KnownProfitableTrader).first()
                    assert trader.funding_source is None
            finally:
                os.unlink(f.name)

    def test_zero_address_handling(self, sybil_detector, session_factory):
        """Test handling of zero address (0x0000...0000) funding source."""
        data = {
            "wallets": [
                {
                    "wallet": "0xwallet12345678901234567890123456789012345",
                    "profit_usd": 224772.31,
                    "funding_source": "0x0000000000000000000000000000000000000000",
                    "funding_source_type": "eoa:USDC.e",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                stats = sybil_detector.import_profitable_wallets(f.name)
                assert stats["imported"] == 1

                # Zero address should be stored but likely excluded from matching
                with session_factory() as session:
                    trader = session.query(KnownProfitableTrader).first()
                    assert trader.funding_source == "0x0000000000000000000000000000000000000000"
            finally:
                os.unlink(f.name)

    def test_duplicate_import_updates(self, sybil_detector, session_factory):
        """Test that re-importing updates existing records."""
        data_v1 = {
            "wallets": [
                {"wallet": "0xwallet12345678901234567890123456789012345", "profit_usd": 100000}
            ]
        }
        data_v2 = {
            "wallets": [
                {"wallet": "0xwallet12345678901234567890123456789012345", "profit_usd": 150000}
            ]
        }

        # First import
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data_v1, f)
            f.flush()
            stats1 = sybil_detector.import_profitable_wallets(f.name)
            os.unlink(f.name)

        assert stats1["imported"] == 1

        # Second import (update)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data_v2, f)
            f.flush()
            stats2 = sybil_detector.import_profitable_wallets(f.name)
            os.unlink(f.name)

        assert stats2["updated"] == 1

        # Verify profit was updated
        with session_factory() as session:
            trader = session.query(KnownProfitableTrader).first()
            assert float(trader.profit_usd) == 150000


# =============================================================================
# HIGH-VOLUME SOURCE FILTERING TESTS
# =============================================================================


class TestHighVolumeSourceFiltering:
    """Tests for high-volume source filtering (exchanges, OTC desks, market makers)."""

    @pytest.fixture
    def high_volume_funding_sources_json(self):
        """Sample data with high-volume funding sources."""
        return {
            "sources": [
                {
                    "address": "0xhighvol1234567890123456789012345678901234",
                    "source_type": "eoa",
                    "labels": [],
                    # This source funded 15 traders - exceeds default threshold of 10
                    "funded_wallets": [f"0xtrader{i:037d}" for i in range(15)],
                    "total_profit_funded": 5000000.00
                },
                {
                    "address": "0xlowvol12345678901234567890123456789012345",
                    "source_type": "eoa",
                    "labels": [],
                    # This source funded only 3 traders - below threshold
                    "funded_wallets": ["0xtrader_a", "0xtrader_b", "0xtrader_c"],
                    "total_profit_funded": 500000.00
                },
                {
                    "address": "0xmedium12345678901234567890123456789012345",
                    "source_type": "eoa",
                    "labels": [],
                    # This source funded exactly 10 traders - at threshold
                    "funded_wallets": [f"0xmed{i:040d}" for i in range(10)],
                    "total_profit_funded": 1000000.00
                },
            ]
        }

    def test_high_volume_sources_excluded_from_index(self, sybil_detector, high_volume_funding_sources_json):
        """Test that high-volume sources are excluded from the index."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(high_volume_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        # Default threshold is 10
        stats = sybil_detector.rebuild_index()

        # Only sources with <= 10 traders should be indexed
        # 0xhighvol: 15 traders (excluded)
        # 0xlowvol: 3 traders (indexed)
        # 0xmedium: 10 traders (indexed - not > threshold)
        assert stats["funding_sources"] == 2
        assert stats["high_volume_excluded"] == 1

    def test_high_volume_source_not_matched(self, sybil_detector, high_volume_funding_sources_json):
        """Test that high-volume sources don't trigger sybil matches."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(high_volume_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # High-volume source should not match
        match = sybil_detector.check_funding_source("0xhighvol1234567890123456789012345678901234")
        assert match.matched is False

        # Low-volume source should match
        match = sybil_detector.check_funding_source("0xlowvol12345678901234567890123456789012345")
        assert match.matched is True

    def test_is_high_volume_source_helper(self, sybil_detector, high_volume_funding_sources_json):
        """Test the is_high_volume_source() helper method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(high_volume_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Test helper method
        assert sybil_detector.is_high_volume_source("0xhighvol1234567890123456789012345678901234") is True
        assert sybil_detector.is_high_volume_source("0xlowvol12345678901234567890123456789012345") is False
        assert sybil_detector.is_high_volume_source("0xmedium12345678901234567890123456789012345") is False
        assert sybil_detector.is_high_volume_source("0xunknown") is False

    def test_threshold_adjustment(self, sybil_detector, high_volume_funding_sources_json):
        """Test adjusting the threshold dynamically."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(high_volume_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        # Set threshold to 5 (stricter)
        stats = sybil_detector.set_max_funded_traders_threshold(5)

        # Now both high-volume (15) and medium (10) should be excluded
        assert stats["funding_sources"] == 1  # Only low-vol (3 traders)
        assert stats["high_volume_excluded"] == 2
        assert sybil_detector.max_funded_traders_threshold == 5

        # Set threshold to 20 (more permissive)
        stats = sybil_detector.set_max_funded_traders_threshold(20)

        # Now all should be indexed
        assert stats["funding_sources"] == 3
        assert stats["high_volume_excluded"] == 0

    def test_get_high_volume_sources_list(self, sybil_detector, session_factory, high_volume_funding_sources_json):
        """Test getting the list of high-volume sources."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(high_volume_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()

        # Get high-volume sources
        high_volume = sybil_detector.get_high_volume_sources()

        assert len(high_volume) == 1
        assert high_volume[0]["address"] == "0xhighvol1234567890123456789012345678901234"
        assert high_volume[0]["funded_trader_count"] == 15
        assert high_volume[0]["type"] == "funding_source"

    def test_stats_include_threshold_info(self, sybil_detector, high_volume_funding_sources_json):
        """Test that stats include threshold configuration info."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(high_volume_funding_sources_json, f)
            f.flush()
            sybil_detector.import_funding_sources(f.name)
            os.unlink(f.name)

        sybil_detector.rebuild_index()
        stats = sybil_detector.get_stats()

        assert "high_volume_sources_excluded" in stats
        assert "max_funded_traders_threshold" in stats
        assert stats["max_funded_traders_threshold"] == 10
        assert stats["high_volume_sources_excluded"] == 1

    def test_withdrawal_dest_high_volume_excluded(self, sybil_detector):
        """Test that high-volume withdrawal destinations are also excluded."""
        data = {
            "destinations": [
                {
                    "address": "0xhighvoldest12345678901234567890123456789",
                    # Received from 12 traders - exceeds threshold
                    "received_from_traders": [f"0xfrom{i:039d}" for i in range(12)],
                    "total_profit_source": 2000000.00
                },
                {
                    "address": "0xlowvoldest123456789012345678901234567890",
                    # Received from 2 traders - below threshold
                    "received_from_traders": ["0xfrom_a", "0xfrom_b"],
                    "total_profit_source": 200000.00
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            sybil_detector.import_withdrawal_destinations(f.name)
            os.unlink(f.name)

        stats = sybil_detector.rebuild_index()

        # Only low-volume dest should be indexed
        assert stats["withdrawal_dests"] == 1
        assert stats["high_volume_excluded"] == 1

        # High-volume dest should not match
        match = sybil_detector.check_funding_source("0xhighvoldest12345678901234567890123456789")
        assert match.matched is False

        # Low-volume dest should match
        match = sybil_detector.check_funding_source("0xlowvoldest123456789012345678901234567890")
        assert match.matched is True


class TestHighVolumeAPIEndpoints:
    """Tests for high-volume source API endpoints."""

    @pytest.fixture
    def api_test_env(self):
        """Create complete API test environment."""
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

        from src.insider_scanner import models as insider_models
        from src.database import models as db_models
        from sqlalchemy.orm import scoped_session
        from sqlalchemy.pool import StaticPool
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.web.api.insider_routes import router
        from src.web.api.dependencies import get_db

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        db_models.Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session_factory = scoped_session(Session)

        app = FastAPI()
        app.include_router(router, prefix="/api")

        def override_get_db():
            session = session_factory()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        client = TestClient(app)
        yield client, session_factory, engine

        client.close()
        session_factory.remove()
        engine.dispose()

    def test_get_threshold_endpoint(self, api_test_env):
        """Test GET /sybil/threshold endpoint."""
        client, _, _ = api_test_env

        response = client.get("/api/insider/sybil/threshold")
        assert response.status_code == 200

        data = response.json()
        assert "max_funded_traders_threshold" in data
        assert data["max_funded_traders_threshold"] == 10  # Default

    def test_update_threshold_endpoint(self, api_test_env):
        """Test POST /sybil/threshold endpoint."""
        client, _, _ = api_test_env

        response = client.post(
            "/api/insider/sybil/threshold",
            json={"threshold": 5}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["old_threshold"] == 10
        assert data["new_threshold"] == 5

    def test_get_high_volume_sources_endpoint(self, api_test_env):
        """Test GET /sybil/high-volume-sources endpoint."""
        client, session_factory, _ = api_test_env

        # Add a high-volume source
        session = session_factory()
        try:
            from src.insider_scanner.models import KnownFundingSource
            source = KnownFundingSource(
                address="0xhighvoltest123456789012345678901234567890",
                funded_trader_wallets=[f"0xtr{i:041d}" for i in range(15)],
                funded_trader_count=15,
            )
            session.add(source)
            session.commit()
        finally:
            session_factory.remove()

        response = client.get("/api/insider/sybil/high-volume-sources")
        assert response.status_code == 200

        data = response.json()
        assert "threshold" in data
        assert "count" in data
        assert "sources" in data

    def test_check_high_volume_endpoint(self, api_test_env):
        """Test GET /sybil/check-high-volume/{address} endpoint."""
        client, session_factory, _ = api_test_env

        # Valid Ethereum addresses (0x + 40 hex chars = 42 chars total)
        high_vol_addr = "0x1111111111111111111111111111111111111111"
        low_vol_addr = "0x2222222222222222222222222222222222222222"

        # Add funding sources
        session = session_factory()
        try:
            from src.insider_scanner.models import KnownFundingSource
            # High-volume source
            high_vol = KnownFundingSource(
                address=high_vol_addr,
                funded_trader_wallets=[f"0xaaa{i:037d}" for i in range(15)],
                funded_trader_count=15,
            )
            # Low-volume source
            low_vol = KnownFundingSource(
                address=low_vol_addr,
                funded_trader_wallets=["0xbbb0000000000000000000000000000000000001", "0xbbb0000000000000000000000000000000000002"],
                funded_trader_count=2,
            )
            session.add_all([high_vol, low_vol])
            session.commit()
        finally:
            session_factory.remove()

        # Check high-volume source
        response = client.get(f"/api/insider/sybil/check-high-volume/{high_vol_addr}")
        assert response.status_code == 200
        data = response.json()
        assert data["is_high_volume"] is True
        assert data["funded_trader_count"] == 15

        # Check low-volume source
        response = client.get(f"/api/insider/sybil/check-high-volume/{low_vol_addr}")
        assert response.status_code == 200
        data = response.json()
        assert data["is_high_volume"] is False
        assert data["funded_trader_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
