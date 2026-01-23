"""Configuration validation utilities.

This module provides additional validation logic that goes beyond
Pydantic's built-in validation, including runtime checks and
cross-field validation.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from .models import AppConfig, TargetAccount


class ValidationSeverity(str, Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Must fix before running
    WARNING = "warning"  # Should fix, but can run
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """A validation issue found in the configuration."""

    severity: ValidationSeverity
    field: str
    message: str
    suggestion: Optional[str] = None


class ConfigValidator:
    """Validates configuration for runtime readiness."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.issues: list[ValidationIssue] = []

    def validate_all(self) -> list[ValidationIssue]:
        """Run all validation checks.

        Returns:
            List of validation issues found
        """
        self.issues = []

        self._validate_targets()
        self._validate_rate_limits()
        self._validate_safety_limits()
        self._validate_execution_params()

        return self.issues

    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    def has_warnings(self) -> bool:
        """Check if any warnings were found."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)

    def _validate_targets(self) -> None:
        """Validate target account configurations."""
        if not self.config.targets:
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="targets",
                    message="No target accounts configured",
                    suggestion="Add at least one target account to copy-trade",
                )
            )
            return

        enabled_count = len(self.config.enabled_targets)
        if enabled_count == 0:
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="targets",
                    message="All target accounts are disabled",
                    suggestion="Enable at least one target account",
                )
            )

        # Check for potential issues with each target
        for target in self.config.targets:
            self._validate_target(target)

    def _validate_target(self, target: TargetAccount) -> None:
        """Validate a single target account configuration."""
        # Check if position ratio is very small
        if target.position_ratio < Decimal("0.001"):
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"targets.{target.name}.position_ratio",
                    message=f"Very small position ratio ({target.position_ratio}) may result in orders below minimum size",
                    suggestion="Consider increasing position_ratio or adjusting min_copy_size_usd",
                )
            )

        # Check if slippage tolerance is very high
        if target.slippage_tolerance > Decimal("0.15"):
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field=f"targets.{target.name}.slippage_tolerance",
                    message=f"High slippage tolerance ({target.slippage_tolerance:.0%}) may result in unfavorable executions",
                    suggestion="Consider reducing slippage_tolerance to 5-10%",
                )
            )

        # Check if slippage tolerance is very low
        if target.slippage_tolerance < Decimal("0.01"):
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    field=f"targets.{target.name}.slippage_tolerance",
                    message=f"Very tight slippage tolerance ({target.slippage_tolerance:.0%}) may result in many skipped trades",
                )
            )

        # Check if max_position_usd is reasonable relative to ratio
        # (e.g., if ratio is 0.01 and max is $10, target would need $1000 position to hit max)
        if target.max_position_usd < Decimal("10"):
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    field=f"targets.{target.name}.max_position_usd",
                    message=f"Low max position size (${target.max_position_usd})",
                )
            )

    def _validate_rate_limits(self) -> None:
        """Validate polling intervals won't exceed rate limits."""
        # Calculate requests per 10 seconds for activity polling
        # Each target is polled at activity_interval_ms
        num_targets = len(self.config.targets)
        interval_s = self.config.polling.activity_interval_ms / 1000

        if interval_s > 0:
            requests_per_10s = (10 / interval_s) * num_targets

            # Data API /activity limit: 1000/10s (general), but we're conservative
            if requests_per_10s > 500:
                self.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field="polling.activity_interval_ms",
                        message=f"Polling {num_targets} targets at {self.config.polling.activity_interval_ms}ms = {requests_per_10s:.0f} req/10s",
                        suggestion="Increase activity_interval_ms or reduce number of targets",
                    )
                )

    def _validate_safety_limits(self) -> None:
        """Validate safety configuration."""
        # Check if max daily loss is reasonable
        if self.config.safety.max_daily_loss_usd > Decimal("10000"):
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    field="safety.max_daily_loss_usd",
                    message=f"High max daily loss limit (${self.config.safety.max_daily_loss_usd})",
                    suggestion="Consider a more conservative limit",
                )
            )

        # Check if min balance is reasonable
        if self.config.safety.min_balance_usd < Decimal("10"):
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    field="safety.min_balance_usd",
                    message=f"Low minimum balance (${self.config.safety.min_balance_usd}) may not cover fees",
                )
            )

    def _validate_execution_params(self) -> None:
        """Validate execution configuration."""
        # Check order timeout
        if self.config.execution.order_timeout_s < 10:
            self.issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    field="execution.order_timeout_s",
                    message=f"Short order timeout ({self.config.execution.order_timeout_s}s) may result in partial fills being cancelled",
                )
            )

        # Check if using FOK with high position ratios (risky)
        if self.config.execution.use_fill_or_kill:
            high_ratio_targets = [
                t for t in self.config.targets if t.position_ratio > Decimal("0.1")
            ]
            if high_ratio_targets:
                self.issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        field="execution.use_fill_or_kill",
                        message="FOK orders with high position ratios may fail frequently due to liquidity",
                        suggestion="Consider disabling FOK or reducing position ratios",
                    )
                )


def validate_config(config: AppConfig) -> tuple[bool, list[ValidationIssue]]:
    """Validate configuration and return issues.

    Args:
        config: The configuration to validate

    Returns:
        Tuple of (is_valid, issues) where is_valid is False if any errors found
    """
    validator = ConfigValidator(config)
    issues = validator.validate_all()
    return not validator.has_errors(), issues


def format_validation_issues(issues: list[ValidationIssue]) -> str:
    """Format validation issues for display.

    Args:
        issues: List of validation issues

    Returns:
        Formatted string for display
    """
    if not issues:
        return "Configuration validation passed with no issues."

    lines = ["Configuration validation found the following issues:", ""]

    for issue in sorted(issues, key=lambda i: (i.severity.value, i.field)):
        icon = {"error": "[X]", "warning": "[!]", "info": "[i]"}[issue.severity.value]
        lines.append(f"{icon} {issue.severity.value.upper()}: {issue.field}")
        lines.append(f"    {issue.message}")
        if issue.suggestion:
            lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")

    return "\n".join(lines)
