"""Tests for Insider Scanner API routes.

Tests the REST API endpoints for:
- Watchlist management
- Detection records
- Investment thesis
- Audit trail export
- Scanner statistics
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.database.connection import DatabaseConfig, DatabaseManager
from src.database.migrations import MigrationManager
from src.insider_scanner.migrations import register_insider_migrations
from src.insider_scanner.models import (
    FlaggedWallet,
    FlaggedFundingSource,
    InsiderCluster,
    DetectionRecord,
    InvestmentThesis,
    InsiderPriority,
    WalletStatus,
)
from src.web.app import create_app


@pytest.fixture
def db():
    """Create a temporary in-memory database."""
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
def test_client(db):
    """Create a test client with database dependency override."""
    app = create_app(debug=True)

    # Override database dependency
    from src.web.api.dependencies import get_db

    def override_get_db():
        with db.session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_wallets(db):
    """Create sample flagged wallets for testing."""
    wallet_data = []
    with db.session() as session:
        for i, (score, priority, status) in enumerate([
            (92.5, InsiderPriority.CRITICAL, WalletStatus.NEW),
            (78.3, InsiderPriority.HIGH, WalletStatus.MONITORING),
            (61.0, InsiderPriority.MEDIUM, WalletStatus.MONITORING),
            (45.5, InsiderPriority.LOW, WalletStatus.CLEARED),
            (35.0, InsiderPriority.NORMAL, WalletStatus.ARCHIVED),
        ]):
            wallet = FlaggedWallet(
                wallet_address=f"0x{'0' * 36}{str(i).zfill(4)}",
                insider_score=Decimal(str(score)),
                priority=priority,
                status=status,
                signal_count=5 + i,
                active_dimensions=3,
                account_age_days=i + 1,
                score_account=Decimal("15"),
                score_trading=Decimal("25"),
                score_behavioral=Decimal("12"),
                score_contextual=Decimal("8"),
                score_cluster=Decimal("5"),
            )
            session.add(wallet)
        session.commit()

        # Query back to get IDs as dicts to avoid detached instance issues
        wallets = session.query(FlaggedWallet).all()
        for w in wallets:
            wallet_data.append({
                "id": w.id,
                "wallet_address": w.wallet_address,
                "insider_score": float(w.insider_score),
                "priority": w.priority.value,
                "status": w.status.value,
            })

    # Return simple objects instead of SQLAlchemy models
    class WalletRef:
        def __init__(self, data):
            self.id = data["id"]
            self.wallet_address = data["wallet_address"]
            self.insider_score = data["insider_score"]
            self.priority = data["priority"]
            self.status = data["status"]

    return [WalletRef(d) for d in wallet_data]


@pytest.fixture
def sample_detection_record(db, sample_wallets):
    """Create a sample detection record."""
    wallet = sample_wallets[0]  # Already a simple object with id/wallet_address

    with db.session() as session:
        record = DetectionRecord(
            wallet_id=wallet.id,
            wallet_address=wallet.wallet_address,
            detected_at=datetime.utcnow(),
            insider_score=Decimal("92.5"),
            priority="critical",
            record_hash="a" * 64,
            signals_snapshot=[
                {"name": "account_age", "category": "account", "weight": 15.0},
                {"name": "win_rate", "category": "trading", "weight": 12.0},
            ],
            market_ids=["0xmarket1", "0xmarket2"],
        )
        session.add(record)
        session.commit()

        # Get the data we need before session closes
        record_data = {
            "id": record.id,
            "record_hash": record.record_hash,
            "wallet_address": record.wallet_address,
            "insider_score": float(record.insider_score),
        }

    # Return a simple object
    class RecordRef:
        def __init__(self, data):
            self.id = data["id"]
            self.record_hash = data["record_hash"]
            self.wallet_address = data["wallet_address"]
            self.insider_score = data["insider_score"]

    return RecordRef(record_data)


# =============================================================================
# Watchlist Tests
# =============================================================================


class TestWatchlistEndpoints:
    """Tests for watchlist endpoints."""

    def test_list_watchlist_empty(self, test_client, db):
        """Test listing empty watchlist."""
        response = test_client.get("/api/insider/watchlist")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_watchlist_with_wallets(self, test_client, sample_wallets):
        """Test listing watchlist with wallets."""
        response = test_client.get("/api/insider/watchlist")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 5

        # Should be sorted by score descending
        assert data[0]["insider_score"] > data[1]["insider_score"]

    def test_filter_by_priority(self, test_client, sample_wallets):
        """Test filtering watchlist by priority."""
        response = test_client.get("/api/insider/watchlist?priority=critical")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["priority"] == "critical"

    def test_filter_by_status(self, test_client, sample_wallets):
        """Test filtering watchlist by status."""
        response = test_client.get("/api/insider/watchlist?status=monitoring")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2
        for wallet in data:
            assert wallet["status"] == "monitoring"

    def test_filter_by_min_score(self, test_client, sample_wallets):
        """Test filtering watchlist by minimum score."""
        response = test_client.get("/api/insider/watchlist?min_score=70")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2
        for wallet in data:
            assert wallet["insider_score"] >= 70

    def test_get_wallet_detail(self, test_client, sample_wallets):
        """Test getting wallet details."""
        wallet_id = sample_wallets[0].id
        response = test_client.get(f"/api/insider/watchlist/{wallet_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == wallet_id
        assert "dimensions" in data
        assert data["priority"] == "critical"

    def test_get_wallet_not_found(self, test_client):
        """Test getting nonexistent wallet."""
        response = test_client.get("/api/insider/watchlist/99999")
        assert response.status_code == 404

    def test_add_wallet_to_watchlist(self, test_client, db):
        """Test adding a wallet to watchlist."""
        response = test_client.post(
            "/api/insider/watchlist",
            json={
                "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
                "priority": "high",
                "notes": "Test wallet",
                "auto_score": False,
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["wallet_address"] == "0x1234567890abcdef1234567890abcdef12345678"
        assert data["priority"] == "high"
        assert data["status"] == "new"

    def test_add_wallet_invalid_address(self, test_client):
        """Test adding wallet with invalid address."""
        response = test_client.post(
            "/api/insider/watchlist",
            json={
                "wallet_address": "invalid_address",
                "priority": "high",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_add_duplicate_wallet(self, test_client, sample_wallets):
        """Test adding duplicate wallet."""
        response = test_client.post(
            "/api/insider/watchlist",
            json={
                "wallet_address": sample_wallets[0].wallet_address,
                "priority": "medium",
            },
        )
        assert response.status_code == 400
        assert "already in watchlist" in response.json()["detail"].lower()

    def test_update_wallet(self, test_client, sample_wallets):
        """Test updating a wallet."""
        wallet_id = sample_wallets[0].id
        response = test_client.patch(
            f"/api/insider/watchlist/{wallet_id}",
            json={
                "status": "escalated",
                "notes": "Updated notes",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "escalated"

    def test_delete_wallet(self, test_client, sample_wallets):
        """Test deleting a wallet from watchlist."""
        wallet_id = sample_wallets[4].id  # Use last wallet
        response = test_client.delete(f"/api/insider/watchlist/{wallet_id}")
        assert response.status_code == 204

        # Verify deleted
        response = test_client.get(f"/api/insider/watchlist/{wallet_id}")
        assert response.status_code == 404


# =============================================================================
# Detection Records Tests
# =============================================================================


class TestDetectionEndpoints:
    """Tests for detection record endpoints."""

    def test_list_detections_empty(self, test_client, db):
        """Test listing empty detection records."""
        response = test_client.get("/api/insider/detections")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_detections(self, test_client, sample_detection_record):
        """Test listing detection records."""
        response = test_client.get("/api/insider/detections")
        assert response.status_code == 200

        data = response.json()
        assert len(data) >= 1

    def test_filter_detections_by_wallet(self, test_client, sample_detection_record):
        """Test filtering detections by wallet address."""
        wallet_addr = sample_detection_record.wallet_address
        response = test_client.get(f"/api/insider/detections?wallet_address={wallet_addr}")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["wallet_address"] == wallet_addr

    def test_get_detection_record(self, test_client, sample_detection_record):
        """Test getting a specific detection record."""
        record_id = sample_detection_record.id
        response = test_client.get(f"/api/insider/detections/{record_id}")
        assert response.status_code == 200

        data = response.json()
        assert "record" in data
        assert "verification" in data

    def test_export_detection_record(self, test_client, sample_detection_record):
        """Test exporting a detection record."""
        record_id = sample_detection_record.id
        response = test_client.get(f"/api/insider/detections/{record_id}/export")
        assert response.status_code == 200

        data = response.json()
        assert "record" in data
        assert "exported_at" in data


# =============================================================================
# Investment Thesis Tests
# =============================================================================


class TestThesisEndpoints:
    """Tests for investment thesis endpoints."""

    def test_create_thesis(self, test_client, sample_detection_record):
        """Test creating an investment thesis."""
        response = test_client.post(
            "/api/insider/thesis",
            json={
                "detection_record_ids": [sample_detection_record.id],
                "reasoning": "Based on critical insider score and fresh account signals, "
                            "placing contrarian bet on same market.",
                "intended_action": "Buy YES",
                "market_id": "0xmarket123",
                "position_side": "YES",
                "position_size": 1000.0,
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["thesis_hash"] is not None
        assert data["detection_record_ids"] == [sample_detection_record.id]
        assert "critical insider score" in data["reasoning"]

    def test_list_theses(self, test_client, sample_detection_record):
        """Test listing investment theses."""
        # Create a thesis first
        test_client.post(
            "/api/insider/thesis",
            json={
                "detection_record_ids": [sample_detection_record.id],
                "reasoning": "Test thesis",
            },
        )

        response = test_client.get("/api/insider/thesis")
        assert response.status_code == 200

        data = response.json()
        assert len(data) >= 1


# =============================================================================
# Audit Trail Tests
# =============================================================================


class TestAuditEndpoints:
    """Tests for audit trail endpoints."""

    def test_export_audit_trail_empty(self, test_client, db):
        """Test exporting empty audit trail."""
        response = test_client.get("/api/insider/audit/export")
        assert response.status_code == 200

        data = response.json()
        assert data["record_count"] == 0
        assert data["detection_records"] == []

    def test_export_audit_trail(self, test_client, sample_detection_record):
        """Test exporting audit trail with records."""
        response = test_client.get("/api/insider/audit/export")
        assert response.status_code == 200

        data = response.json()
        assert data["record_count"] >= 1
        assert "exported_at" in data
        assert "chain_verification" in data

    def test_verify_chain_integrity(self, test_client, db):
        """Test verifying chain integrity."""
        response = test_client.get("/api/insider/audit/verify")
        assert response.status_code == 200

        data = response.json()
        assert "valid" in data
        assert "entries_checked" in data


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatsEndpoints:
    """Tests for statistics endpoints."""

    def test_get_stats_empty(self, test_client, db):
        """Test getting stats with no data."""
        response = test_client.get("/api/insider/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_wallets_flagged"] == 0
        assert data["critical_count"] == 0

    def test_get_stats_with_wallets(self, test_client, sample_wallets):
        """Test getting stats with wallets."""
        response = test_client.get("/api/insider/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_wallets_flagged"] == 5
        assert data["critical_count"] == 1
        assert data["high_count"] == 1
        assert data["medium_count"] == 1
        assert data["low_count"] == 1


# =============================================================================
# Scanner Control Tests
# =============================================================================


class TestScannerControlEndpoints:
    """Tests for scanner control endpoints."""

    def test_get_scanner_status(self, test_client):
        """Test getting scanner status."""
        response = test_client.get("/api/insider/status")
        assert response.status_code == 200

        data = response.json()
        assert "is_running" in data
        assert "mode" in data

    def test_start_scanner(self, test_client):
        """Test starting the scanner."""
        response = test_client.post("/api/insider/control/start")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "start_requested"

    def test_stop_scanner(self, test_client):
        """Test stopping the scanner."""
        response = test_client.post("/api/insider/control/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stop_requested"


# =============================================================================
# Funding Source Tests
# =============================================================================


class TestFundingSourceEndpoints:
    """Tests for funding source endpoints."""

    def test_list_funding_sources_empty(self, test_client, db):
        """Test listing empty funding sources."""
        response = test_client.get("/api/insider/funding-sources")
        assert response.status_code == 200
        assert response.json() == []

    def test_add_funding_source(self, test_client, db):
        """Test adding a funding source."""
        response = test_client.post(
            "/api/insider/funding-sources",
            params={
                "funding_address": "0xabcdef0123456789abcdef0123456789abcdef01",
                "reason": "Associated with known insider",
                "risk_level": "high",
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["funding_address"] == "0xabcdef0123456789abcdef0123456789abcdef01"
        assert data["risk_level"] == "high"
        assert data["manually_flagged"] is True

    def test_add_duplicate_funding_source(self, test_client, db):
        """Test adding duplicate funding source."""
        # Add first
        test_client.post(
            "/api/insider/funding-sources",
            params={
                "funding_address": "0xabcdef0123456789abcdef0123456789abcdef02",
                "risk_level": "medium",
            },
        )

        # Try to add duplicate
        response = test_client.post(
            "/api/insider/funding-sources",
            params={
                "funding_address": "0xabcdef0123456789abcdef0123456789abcdef02",
                "risk_level": "high",
            },
        )
        assert response.status_code == 400


# =============================================================================
# Cluster Tests
# =============================================================================


class TestClusterEndpoints:
    """Tests for cluster endpoints."""

    def test_list_clusters_empty(self, test_client, db):
        """Test listing empty clusters."""
        response = test_client.get("/api/insider/clusters")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_cluster_not_found(self, test_client):
        """Test getting nonexistent cluster."""
        response = test_client.get("/api/insider/clusters/99999")
        assert response.status_code == 404


# =============================================================================
# Pagination Tests
# =============================================================================


class TestPagination:
    """Tests for pagination."""

    def test_watchlist_limit(self, test_client, sample_wallets):
        """Test watchlist limit parameter."""
        response = test_client.get("/api/insider/watchlist?limit=2")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2

    def test_watchlist_offset(self, test_client, sample_wallets):
        """Test watchlist offset parameter."""
        # Get first page
        response1 = test_client.get("/api/insider/watchlist?limit=2&offset=0")
        data1 = response1.json()

        # Get second page
        response2 = test_client.get("/api/insider/watchlist?limit=2&offset=2")
        data2 = response2.json()

        # Ensure different results
        assert data1[0]["id"] != data2[0]["id"]
