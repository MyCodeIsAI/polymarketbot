"""Tests for configuration loading and validation."""

import os
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from src.config.models import (
    AppConfig,
    TargetAccount,
    YourAccount,
    SignatureType,
    SlippageAction,
    PollingConfig,
    ExecutionConfig,
    SafetyConfig,
)
from src.config.loader import load_config, ConfigError, _deep_merge
from src.config.validation import (
    validate_config,
    ConfigValidator,
    ValidationSeverity,
)


# =============================================================================
# Model Tests
# =============================================================================


class TestTargetAccount:
    """Tests for TargetAccount model."""

    def test_valid_target_account(self):
        """Test creating a valid target account."""
        account = TargetAccount(
            name="test_whale",
            wallet="0x1234567890123456789012345678901234567890",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
        )

        assert account.name == "test_whale"
        assert account.wallet == "0x1234567890123456789012345678901234567890"
        assert account.position_ratio == Decimal("0.01")
        assert account.enabled is True  # default
        assert account.slippage_tolerance == Decimal("0.05")  # default
        assert account.slippage_action == SlippageAction.SKIP  # default

    def test_wallet_address_validation(self):
        """Test that invalid wallet addresses are rejected."""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            TargetAccount(
                name="test",
                wallet="not_a_valid_address",
                position_ratio=Decimal("0.01"),
                max_position_usd=Decimal("500"),
            )

    def test_wallet_address_lowercase_conversion(self):
        """Test that wallet addresses are converted to lowercase."""
        account = TargetAccount(
            name="test",
            wallet="0xABCDEF1234567890123456789012345678901234",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
        )
        assert account.wallet == "0xabcdef1234567890123456789012345678901234"

    def test_name_validation(self):
        """Test that invalid names are rejected."""
        with pytest.raises(ValueError, match="alphanumeric"):
            TargetAccount(
                name="test with spaces",
                wallet="0x1234567890123456789012345678901234567890",
                position_ratio=Decimal("0.01"),
                max_position_usd=Decimal("500"),
            )

    def test_position_ratio_bounds(self):
        """Test position ratio validation."""
        # Too low (must be > 0)
        with pytest.raises(ValueError):
            TargetAccount(
                name="test",
                wallet="0x1234567890123456789012345678901234567890",
                position_ratio=Decimal("0"),
                max_position_usd=Decimal("500"),
            )

        # Too high (must be <= 10)
        with pytest.raises(ValueError):
            TargetAccount(
                name="test",
                wallet="0x1234567890123456789012345678901234567890",
                position_ratio=Decimal("11"),
                max_position_usd=Decimal("500"),
            )

    def test_slippage_tolerance_bounds(self):
        """Test slippage tolerance validation."""
        # Valid at boundaries
        account = TargetAccount(
            name="test",
            wallet="0x1234567890123456789012345678901234567890",
            position_ratio=Decimal("0.01"),
            max_position_usd=Decimal("500"),
            slippage_tolerance=Decimal("0"),  # 0% is valid
        )
        assert account.slippage_tolerance == Decimal("0")

        # Too high (must be <= 1)
        with pytest.raises(ValueError):
            TargetAccount(
                name="test",
                wallet="0x1234567890123456789012345678901234567890",
                position_ratio=Decimal("0.01"),
                max_position_usd=Decimal("500"),
                slippage_tolerance=Decimal("1.5"),
            )


class TestYourAccount:
    """Tests for YourAccount model."""

    def test_valid_your_account(self):
        """Test creating a valid your account config."""
        account = YourAccount(
            proxy_wallet="0x1234567890123456789012345678901234567890",
        )

        assert account.private_key_env == "POLYBOT_PRIVATE_KEY"  # default
        assert account.signature_type == SignatureType.GNOSIS_SAFE  # default

    def test_signature_types(self):
        """Test all signature types are valid."""
        for sig_type in SignatureType:
            account = YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
                signature_type=sig_type,
            )
            assert account.signature_type == sig_type


class TestAppConfig:
    """Tests for AppConfig model."""

    def test_minimal_valid_config(self):
        """Test creating minimal valid configuration."""
        config = AppConfig(
            your_account=YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
            ),
        )

        assert config.your_account is not None
        assert len(config.targets) == 0
        assert config.polling is not None  # default
        assert config.execution is not None  # default
        assert config.safety is not None  # default

    def test_config_with_targets(self):
        """Test configuration with target accounts."""
        config = AppConfig(
            your_account=YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
            ),
            targets=[
                TargetAccount(
                    name="whale_1",
                    wallet="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    position_ratio=Decimal("0.01"),
                    max_position_usd=Decimal("500"),
                ),
                TargetAccount(
                    name="whale_2",
                    wallet="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    position_ratio=Decimal("0.02"),
                    max_position_usd=Decimal("1000"),
                    enabled=False,
                ),
            ],
        )

        assert len(config.targets) == 2
        assert len(config.enabled_targets) == 1
        assert config.enabled_targets[0].name == "whale_1"

    def test_duplicate_target_names_rejected(self):
        """Test that duplicate target names are rejected."""
        with pytest.raises(ValueError, match="unique"):
            AppConfig(
                your_account=YourAccount(
                    proxy_wallet="0x1234567890123456789012345678901234567890",
                ),
                targets=[
                    TargetAccount(
                        name="same_name",
                        wallet="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        position_ratio=Decimal("0.01"),
                        max_position_usd=Decimal("500"),
                    ),
                    TargetAccount(
                        name="same_name",  # duplicate
                        wallet="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        position_ratio=Decimal("0.02"),
                        max_position_usd=Decimal("1000"),
                    ),
                ],
            )

    def test_get_target_by_name(self):
        """Test looking up target by name."""
        config = AppConfig(
            your_account=YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
            ),
            targets=[
                TargetAccount(
                    name="whale_1",
                    wallet="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    position_ratio=Decimal("0.01"),
                    max_position_usd=Decimal("500"),
                ),
            ],
        )

        target = config.get_target_by_name("whale_1")
        assert target is not None
        assert target.name == "whale_1"

        assert config.get_target_by_name("nonexistent") is None

    def test_get_target_by_wallet(self):
        """Test looking up target by wallet address."""
        config = AppConfig(
            your_account=YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
            ),
            targets=[
                TargetAccount(
                    name="whale_1",
                    wallet="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    position_ratio=Decimal("0.01"),
                    max_position_usd=Decimal("500"),
                ),
            ],
        )

        # Test with lowercase
        target = config.get_target_by_wallet("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        assert target is not None

        # Test with mixed case (should still find it)
        target = config.get_target_by_wallet("0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        assert target is not None


# =============================================================================
# Loader Tests
# =============================================================================


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_deep_merge(self):
        """Test deep merging of dictionaries."""
        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [1, 2, 3],
        }
        override = {
            "a": 10,
            "b": {"c": 20},
            "f": 4,
        }

        result = _deep_merge(base, override)

        assert result["a"] == 10  # overridden
        assert result["b"]["c"] == 20  # nested override
        assert result["b"]["d"] == 3  # preserved from base
        assert result["e"] == [1, 2, 3]  # preserved
        assert result["f"] == 4  # new key

    def test_load_config_from_files(self):
        """Test loading configuration from YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create accounts.yaml
            accounts_data = {
                "your_account": {
                    "proxy_wallet": "0x1234567890123456789012345678901234567890",
                },
                "targets": [
                    {
                        "name": "test_whale",
                        "wallet": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "position_ratio": 0.01,
                        "max_position_usd": 500,
                    }
                ],
            }
            with open(config_dir / "accounts.yaml", "w") as f:
                yaml.dump(accounts_data, f)

            # Create settings.yaml
            settings_data = {
                "polling": {
                    "activity_interval_ms": 300,
                },
            }
            with open(config_dir / "settings.yaml", "w") as f:
                yaml.dump(settings_data, f)

            # Load config
            config = load_config(config_dir=config_dir, env_file=None)

            assert config.your_account.proxy_wallet == "0x1234567890123456789012345678901234567890"
            assert len(config.targets) == 1
            assert config.targets[0].name == "test_whale"
            assert config.polling.activity_interval_ms == 300

    def test_load_config_missing_file(self):
        """Test that missing config file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ConfigError, match="not found"):
                load_config(config_dir=Path(tmpdir), env_file=None)


# =============================================================================
# Validation Tests
# =============================================================================


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validation_no_targets_warning(self):
        """Test warning when no targets configured."""
        config = AppConfig(
            your_account=YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
            ),
        )

        is_valid, issues = validate_config(config)

        # Should be valid but with warning
        assert is_valid is True
        assert any(
            i.severity == ValidationSeverity.WARNING and "No target" in i.message
            for i in issues
        )

    def test_validation_small_position_ratio_warning(self):
        """Test warning for very small position ratio."""
        config = AppConfig(
            your_account=YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
            ),
            targets=[
                TargetAccount(
                    name="whale_1",
                    wallet="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    position_ratio=Decimal("0.0001"),  # Very small
                    max_position_usd=Decimal("500"),
                ),
            ],
        )

        is_valid, issues = validate_config(config)

        assert any(
            i.severity == ValidationSeverity.WARNING and "small position ratio" in i.message
            for i in issues
        )

    def test_validation_high_slippage_warning(self):
        """Test warning for high slippage tolerance."""
        config = AppConfig(
            your_account=YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
            ),
            targets=[
                TargetAccount(
                    name="whale_1",
                    wallet="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    position_ratio=Decimal("0.01"),
                    max_position_usd=Decimal("500"),
                    slippage_tolerance=Decimal("0.25"),  # 25% - very high
                ),
            ],
        )

        is_valid, issues = validate_config(config)

        assert any(
            i.severity == ValidationSeverity.WARNING and "slippage" in i.message.lower()
            for i in issues
        )

    def test_validation_passes_for_good_config(self):
        """Test that well-configured setup passes validation."""
        config = AppConfig(
            your_account=YourAccount(
                proxy_wallet="0x1234567890123456789012345678901234567890",
            ),
            targets=[
                TargetAccount(
                    name="whale_1",
                    wallet="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    position_ratio=Decimal("0.01"),
                    max_position_usd=Decimal("500"),
                    slippage_tolerance=Decimal("0.05"),
                ),
            ],
        )

        is_valid, issues = validate_config(config)

        # Should pass with no errors (may have info-level issues)
        assert is_valid is True
        assert not any(i.severity == ValidationSeverity.ERROR for i in issues)


# =============================================================================
# Polling Config Tests
# =============================================================================


class TestPollingConfig:
    """Tests for PollingConfig model."""

    def test_default_values(self):
        """Test default polling configuration."""
        config = PollingConfig()

        assert config.activity_interval_ms == 200
        assert config.positions_sync_interval_s == 60
        assert config.balance_check_interval_s == 30

    def test_interval_bounds(self):
        """Test polling interval validation."""
        # Too low
        with pytest.raises(ValueError):
            PollingConfig(activity_interval_ms=50)  # min is 100

        # Too high
        with pytest.raises(ValueError):
            PollingConfig(activity_interval_ms=10000)  # max is 5000


# =============================================================================
# Safety Config Tests
# =============================================================================


class TestSafetyConfig:
    """Tests for SafetyConfig model."""

    def test_default_values(self):
        """Test default safety configuration."""
        config = SafetyConfig()

        assert config.max_daily_loss_usd == Decimal("1000")
        assert config.max_open_positions == 50
        assert config.min_balance_usd == Decimal("50")

    def test_positive_values_required(self):
        """Test that safety values must be positive."""
        with pytest.raises(ValueError):
            SafetyConfig(max_daily_loss_usd=Decimal("-100"))
