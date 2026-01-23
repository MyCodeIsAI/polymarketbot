"""Position sizing for copy trades.

This module calculates the appropriate size for copy trades based on:
- Target's position size
- Configured position ratio (e.g., 1/100th)
- Maximum position USD limit
- Available balance
- Minimum order size requirements
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Optional

from ..config.models import TargetAccount
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Polymarket minimum order size (approximate)
MIN_ORDER_SIZE_USD = Decimal("1.00")
MIN_ORDER_SIZE_SHARES = Decimal("1.0")


class SizeRejectionReason(str, Enum):
    """Reasons why a position size may be rejected."""

    NONE = "none"
    BELOW_MIN_SIZE = "below_min_size"
    EXCEEDS_MAX_POSITION = "exceeds_max_position"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    TARGET_BELOW_THRESHOLD = "target_below_threshold"
    ZERO_SIZE = "zero_size"


@dataclass
class SizingResult:
    """Result of position sizing calculation."""

    # Calculated values
    target_size: Decimal
    calculated_size: Decimal
    final_size: Decimal
    estimated_cost: Decimal

    # Whether the trade should proceed
    approved: bool
    rejection_reason: SizeRejectionReason = SizeRejectionReason.NONE

    # Constraints applied
    ratio_applied: Decimal = Decimal("1")
    max_position_capped: bool = False
    balance_capped: bool = False

    @property
    def size_usd(self) -> Decimal:
        """Get final size in USD terms."""
        return self.estimated_cost

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "target_size": str(self.target_size),
            "calculated_size": str(self.calculated_size),
            "final_size": str(self.final_size),
            "estimated_cost": str(self.estimated_cost),
            "approved": self.approved,
            "rejection_reason": self.rejection_reason.value,
        }


class PositionSizer:
    """Calculates copy trade position sizes.

    Applies all configured constraints to determine the appropriate
    size for a copy trade.

    Example:
        sizer = PositionSizer(target_config)
        result = sizer.calculate_size(
            target_size=Decimal("1000"),
            current_price=Decimal("0.5"),
            available_balance=Decimal("500"),
        )
        if result.approved:
            execute_order(result.final_size)
    """

    def __init__(
        self,
        target: TargetAccount,
        fee_rate: Decimal = Decimal("0"),  # Currently 0 on Polymarket
    ):
        """Initialize the position sizer.

        Args:
            target: Target account configuration
            fee_rate: Trading fee rate (0-1)
        """
        self.target = target
        self.fee_rate = fee_rate

    def calculate_size(
        self,
        target_size: Decimal,
        current_price: Decimal,
        available_balance: Decimal,
        existing_position_size: Decimal = Decimal("0"),
        is_exit: bool = False,
    ) -> SizingResult:
        """Calculate the appropriate copy trade size.

        Args:
            target_size: The target's position/trade size in shares
            current_price: Current market price (0-1)
            available_balance: Available USDC balance
            existing_position_size: Our current position size (for exits)
            is_exit: Whether this is an exit trade (sell)

        Returns:
            SizingResult with calculated size and approval status
        """
        # Handle zero size
        if target_size <= 0:
            return SizingResult(
                target_size=target_size,
                calculated_size=Decimal("0"),
                final_size=Decimal("0"),
                estimated_cost=Decimal("0"),
                approved=False,
                rejection_reason=SizeRejectionReason.ZERO_SIZE,
            )

        # Check if target's position is below our threshold
        target_usd_value = target_size * current_price
        if target_usd_value < self.target.min_position_usd:
            return SizingResult(
                target_size=target_size,
                calculated_size=Decimal("0"),
                final_size=Decimal("0"),
                estimated_cost=Decimal("0"),
                approved=False,
                rejection_reason=SizeRejectionReason.TARGET_BELOW_THRESHOLD,
            )

        # Apply position ratio
        calculated_size = target_size * self.target.position_ratio

        # For exits, we can only sell what we have
        if is_exit:
            calculated_size = min(calculated_size, existing_position_size)

        # Calculate estimated cost
        estimated_cost = calculated_size * current_price

        # Track if constraints were applied
        max_position_capped = False
        balance_capped = False
        final_size = calculated_size

        # Apply max position USD limit
        if estimated_cost > self.target.max_position_usd:
            final_size = self.target.max_position_usd / current_price
            estimated_cost = self.target.max_position_usd
            max_position_capped = True

        # Apply balance constraint (with fee buffer)
        if not is_exit:
            fee_buffer = Decimal("1.01")  # 1% buffer for potential fees
            max_from_balance = available_balance / (current_price * fee_buffer)

            if final_size > max_from_balance:
                final_size = max_from_balance
                estimated_cost = final_size * current_price
                balance_capped = True

        # Round down to reasonable precision
        final_size = final_size.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        estimated_cost = (final_size * current_price).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        )

        # Check minimum size requirements
        if estimated_cost < self.target.min_copy_size_usd:
            return SizingResult(
                target_size=target_size,
                calculated_size=calculated_size,
                final_size=Decimal("0"),
                estimated_cost=Decimal("0"),
                approved=False,
                rejection_reason=SizeRejectionReason.BELOW_MIN_SIZE,
                ratio_applied=self.target.position_ratio,
            )

        if final_size < MIN_ORDER_SIZE_SHARES:
            return SizingResult(
                target_size=target_size,
                calculated_size=calculated_size,
                final_size=Decimal("0"),
                estimated_cost=Decimal("0"),
                approved=False,
                rejection_reason=SizeRejectionReason.BELOW_MIN_SIZE,
                ratio_applied=self.target.position_ratio,
            )

        # Check if balance constraint made it too small
        if balance_capped and estimated_cost < MIN_ORDER_SIZE_USD:
            return SizingResult(
                target_size=target_size,
                calculated_size=calculated_size,
                final_size=Decimal("0"),
                estimated_cost=Decimal("0"),
                approved=False,
                rejection_reason=SizeRejectionReason.INSUFFICIENT_BALANCE,
                ratio_applied=self.target.position_ratio,
            )

        # Approved!
        return SizingResult(
            target_size=target_size,
            calculated_size=calculated_size,
            final_size=final_size,
            estimated_cost=estimated_cost,
            approved=True,
            rejection_reason=SizeRejectionReason.NONE,
            ratio_applied=self.target.position_ratio,
            max_position_capped=max_position_capped,
            balance_capped=balance_capped,
        )

    def calculate_exit_size(
        self,
        target_exit_size: Decimal,
        our_position_size: Decimal,
        target_total_position: Decimal,
        current_price: Decimal,
    ) -> SizingResult:
        """Calculate size for an exit (sell) trade.

        When the target reduces their position, we reduce ours proportionally.

        Args:
            target_exit_size: How much the target is selling
            our_position_size: Our current position size
            target_total_position: Target's total position before exit
            current_price: Current market price

        Returns:
            SizingResult for the exit trade
        """
        if our_position_size <= 0:
            return SizingResult(
                target_size=target_exit_size,
                calculated_size=Decimal("0"),
                final_size=Decimal("0"),
                estimated_cost=Decimal("0"),
                approved=False,
                rejection_reason=SizeRejectionReason.ZERO_SIZE,
            )

        # Calculate what percentage of their position they're exiting
        if target_total_position > 0:
            exit_ratio = target_exit_size / target_total_position
        else:
            exit_ratio = Decimal("1")  # Full exit

        # Apply same ratio to our position
        calculated_size = our_position_size * exit_ratio

        # Round down
        final_size = calculated_size.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        estimated_value = final_size * current_price

        # Check minimum
        if final_size < MIN_ORDER_SIZE_SHARES:
            return SizingResult(
                target_size=target_exit_size,
                calculated_size=calculated_size,
                final_size=Decimal("0"),
                estimated_cost=Decimal("0"),
                approved=False,
                rejection_reason=SizeRejectionReason.BELOW_MIN_SIZE,
            )

        return SizingResult(
            target_size=target_exit_size,
            calculated_size=calculated_size,
            final_size=final_size,
            estimated_cost=estimated_value,
            approved=True,
        )


def calculate_proportional_size(
    target_size: Decimal,
    ratio: Decimal,
    max_usd: Decimal,
    price: Decimal,
    available_balance: Decimal,
    min_size_usd: Decimal = MIN_ORDER_SIZE_USD,
) -> tuple[Decimal, bool, str]:
    """Simplified size calculation function.

    Args:
        target_size: Target's position size
        ratio: Position ratio to apply
        max_usd: Maximum position in USD
        price: Current price
        available_balance: Available balance
        min_size_usd: Minimum order size in USD

    Returns:
        Tuple of (size, approved, reason)
    """
    # Apply ratio
    size = target_size * ratio

    # Cap at max USD
    size_usd = size * price
    if size_usd > max_usd:
        size = max_usd / price
        size_usd = max_usd

    # Cap at balance
    if size_usd > available_balance * Decimal("0.99"):
        size = (available_balance * Decimal("0.99")) / price
        size_usd = size * price

    # Round
    size = size.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    size_usd = size * price

    # Check minimum
    if size_usd < min_size_usd:
        return Decimal("0"), False, "below_minimum"

    return size, True, "approved"
