"""Tests for CLI module."""

import os
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from typer.testing import CliRunner

from src.cli.main import app
from src.cli.formatters import (
    format_decimal,
    format_usd,
    format_percent,
    format_timestamp,
    format_relative_time,
    format_wallet,
    format_table,
    format_accounts,
    format_positions,
    format_trades,
    format_pnl,
)


runner = CliRunner()


# =============================================================================
# Formatter Tests
# =============================================================================

class TestFormatters:
    """Tests for CLI formatters."""

    def test_format_decimal(self):
        """Test decimal formatting."""
        assert format_decimal(Decimal("123.456"), 2) == "123.46"
        assert format_decimal(Decimal("1000.5"), 2) == "1,000.50"
        assert format_decimal(None) == "-"

    def test_format_usd(self):
        """Test USD formatting."""
        assert format_usd(Decimal("100")) == "$100.00"
        assert format_usd(Decimal("1234.56")) == "$1,234.56"
        assert format_usd(Decimal("-50.25")) == "$-50.25"
        assert format_usd(None) == "-"

    def test_format_percent(self):
        """Test percentage formatting."""
        assert format_percent(Decimal("0.05")) == "5.00%"
        assert format_percent(Decimal("0.123")) == "12.30%"
        assert format_percent(Decimal("0.05"), include_sign=True) == "+5.00%"
        assert format_percent(Decimal("-0.05"), include_sign=True) == "-5.00%"
        assert format_percent(None) == "-"

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        assert format_timestamp(dt) == "2024-01-15 10:30:45"
        assert format_timestamp(None) == "-"

    def test_format_relative_time(self):
        """Test relative time formatting."""
        now = datetime.utcnow()

        # Recent
        recent = datetime.utcnow()
        result = format_relative_time(recent)
        assert "s ago" in result or "0s" in result

        # None
        assert format_relative_time(None) == "-"

    def test_format_wallet(self):
        """Test wallet address formatting."""
        wallet = "0x1234567890abcdef1234567890abcdef12345678"

        # Short format
        short = format_wallet(wallet, short=True)
        assert short == "0x123456...345678"

        # Full format
        full = format_wallet(wallet, short=False)
        assert full == wallet

        # Empty
        assert format_wallet("") == "-"
        assert format_wallet(None) == "-"

    def test_format_table(self):
        """Test table formatting."""
        columns = [("Name", "cyan"), ("Value", "white")]
        rows = [["Item 1", "100"], ["Item 2", "200"]]

        table = format_table("Test Table", columns, rows)

        assert table.title == "Test Table"
        assert len(table.columns) == 2

    def test_format_accounts(self):
        """Test accounts table formatting."""
        accounts = [
            {
                "name": "whale1",
                "target_wallet": "0x1234567890abcdef1234567890abcdef12345678",
                "position_ratio": "0.01",
                "max_position_usd": "500",
                "slippage_tolerance": "0.05",
                "enabled": True,
            },
        ]

        table = format_accounts(accounts)
        assert table.title == "Tracked Accounts"

    def test_format_positions(self):
        """Test positions table formatting."""
        positions = [
            {
                "market_name": "Test Market",
                "outcome": "Yes",
                "our_size": "10",
                "average_price": "0.5",
                "current_price": "0.6",
                "unrealized_pnl": "1.0",
                "status": "synced",
            },
        ]

        table = format_positions(positions)
        assert table.title == "Positions"

    def test_format_trades(self):
        """Test trades table formatting."""
        trades = [
            {
                "detected_at": "2024-01-15T10:30:45",
                "account_id": 1,
                "side": "BUY",
                "target_size": "100",
                "target_price": "0.5",
                "execution_size": "100",
                "execution_price": "0.51",
                "slippage_percent": "0.02",
                "status": "filled",
                "total_latency_ms": 250,
            },
        ]

        table = format_trades(trades)
        assert "Trades" in table.title

    def test_format_pnl(self):
        """Test P&L panel formatting."""
        pnl_data = {
            "realized_pnl": "100",
            "unrealized_pnl": "50",
            "total_invested": "1000",
            "roi_percent": 15,
            "total_trades": 20,
            "winning_trades": 12,
            "losing_trades": 8,
            "win_rate_percent": 60,
        }

        panel = format_pnl(pnl_data)
        assert panel.title == "P&L Summary"


# =============================================================================
# CLI Command Tests
# =============================================================================

class TestVersionCommand:
    """Tests for version command."""

    def test_version(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "PolymarketBot" in result.stdout


class TestStatusCommand:
    """Tests for status command."""

    def test_status_not_running(self):
        """Test status when bot not running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = runner.invoke(app, ["status"])
            # Should show not running status
            assert "not running" in result.stdout.lower() or result.exit_code == 0


class TestAccountsCommands:
    """Tests for accounts commands."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        with patch("src.cli.main.get_database") as mock:
            db = MagicMock()
            mock.return_value = db
            yield db

    def test_accounts_list_empty(self, mock_db):
        """Test accounts list with no accounts."""
        from src.database import AccountRepository

        with patch.object(AccountRepository, "get_enabled", return_value=[]):
            with patch.object(AccountRepository, "__init__", return_value=None):
                result = runner.invoke(app, ["accounts", "list"])
                assert result.exit_code == 0
                assert "no accounts" in result.stdout.lower()

    def test_accounts_add_invalid_wallet(self):
        """Test adding account with invalid wallet."""
        result = runner.invoke(
            app,
            ["accounts", "add", "--name", "test", "--wallet", "invalid"],
        )
        assert result.exit_code == 1
        assert "invalid" in result.stdout.lower()


class TestPositionsCommand:
    """Tests for positions command."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        with patch("src.cli.main.get_database") as mock:
            db = MagicMock()
            mock.return_value = db
            yield db

    def test_positions_empty(self, mock_db):
        """Test positions with no positions."""
        from src.database import PositionRepository, AccountRepository

        with patch.object(PositionRepository, "get_open_positions", return_value=[]):
            with patch.object(PositionRepository, "__init__", return_value=None):
                with patch.object(AccountRepository, "__init__", return_value=None):
                    result = runner.invoke(app, ["positions"])
                    assert result.exit_code == 0
                    assert "no open positions" in result.stdout.lower()


class TestTradesCommand:
    """Tests for trades command."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        with patch("src.cli.main.get_database") as mock:
            db = MagicMock()
            mock.return_value = db
            yield db

    def test_trades_empty(self, mock_db):
        """Test trades with no trades."""
        from src.database import TradeLogRepository, AccountRepository

        with patch.object(TradeLogRepository, "get_recent_trades", return_value=[]):
            with patch.object(TradeLogRepository, "__init__", return_value=None):
                with patch.object(AccountRepository, "__init__", return_value=None):
                    result = runner.invoke(app, ["trades"])
                    assert result.exit_code == 0
                    assert "no trades" in result.stdout.lower()


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_missing_config(self):
        """Test validate with missing config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                ["validate", "--config", tmpdir],
            )
            # Should fail since config files don't exist
            assert result.exit_code == 1


class TestReconcileCommand:
    """Tests for reconcile command."""

    def test_reconcile_dry_run(self):
        """Test reconcile in dry-run mode."""
        result = runner.invoke(app, ["reconcile", "--dry-run"])
        assert result.exit_code == 0
        assert "dry-run" in result.stdout.lower()


# =============================================================================
# Integration Tests
# =============================================================================

class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_help_command(self):
        """Test help displays correctly."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "status" in result.stdout
        assert "accounts" in result.stdout
        assert "positions" in result.stdout
        assert "trades" in result.stdout

    def test_accounts_help(self):
        """Test accounts subcommand help."""
        result = runner.invoke(app, ["accounts", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "add" in result.stdout
        assert "remove" in result.stdout
        assert "pause" in result.stdout
        assert "resume" in result.stdout

    def test_invalid_command(self):
        """Test invalid command shows error."""
        result = runner.invoke(app, ["nonexistent"])
        assert result.exit_code != 0

    def test_start_help(self):
        """Test start command help."""
        result = runner.invoke(app, ["start", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.stdout
        assert "--verbose" in result.stdout
        assert "--dry-run" in result.stdout
