"""Slippage calculation and protection.

This module provides:
- Slippage calculation between target price and current price
- Order book depth analysis
- Slippage protection with configurable thresholds
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..api.clob import OrderBook, OrderSide
from ..config.models import SlippageAction
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SlippageStatus(str, Enum):
    """Status of slippage check."""

    OK = "ok"
    EXCEEDED = "exceeded"
    NO_LIQUIDITY = "no_liquidity"
    PRICE_UNAVAILABLE = "price_unavailable"


@dataclass
class SlippageCheckResult:
    """Result of a slippage check."""

    status: SlippageStatus
    target_price: Decimal
    current_price: Optional[Decimal]
    slippage_percent: Optional[Decimal]
    max_allowed: Decimal
    recommended_action: SlippageAction

    # For order execution
    execution_price: Optional[Decimal] = None
    available_liquidity: Optional[Decimal] = None

    @property
    def is_ok(self) -> bool:
        """Check if slippage is acceptable."""
        return self.status == SlippageStatus.OK

    @property
    def slippage_bps(self) -> Optional[int]:
        """Get slippage in basis points."""
        if self.slippage_percent is not None:
            return int(self.slippage_percent * 10000)
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "status": self.status.value,
            "target_price": str(self.target_price),
            "current_price": str(self.current_price) if self.current_price else None,
            "slippage_percent": str(self.slippage_percent) if self.slippage_percent else None,
            "slippage_bps": self.slippage_bps,
            "max_allowed_percent": str(self.max_allowed),
            "recommended_action": self.recommended_action.value,
        }


class SlippageCalculator:
    """Calculates and checks slippage for trades.

    Compares the target's execution price with the current market price
    to determine if slippage is acceptable.

    Example:
        calculator = SlippageCalculator(max_slippage=Decimal("0.05"))
        result = calculator.check_slippage(
            target_price=Decimal("0.50"),
            order_book=order_book,
            side=OrderSide.BUY,
            size=Decimal("100"),
        )
        if result.is_ok:
            execute_at(result.execution_price)
    """

    def __init__(
        self,
        max_slippage: Decimal = Decimal("0.05"),
        slippage_action: SlippageAction = SlippageAction.SKIP,
    ):
        """Initialize the slippage calculator.

        Args:
            max_slippage: Maximum allowed slippage (0.05 = 5%)
            slippage_action: Action to take when slippage exceeded
        """
        self.max_slippage = max_slippage
        self.slippage_action = slippage_action

    def check_slippage(
        self,
        target_price: Decimal,
        order_book: Optional[OrderBook],
        side: OrderSide,
        size: Optional[Decimal] = None,
    ) -> SlippageCheckResult:
        """Check slippage against order book.

        Args:
            target_price: Price the target executed at
            order_book: Current order book
            side: Trade side (BUY or SELL)
            size: Order size (for depth check)

        Returns:
            SlippageCheckResult with analysis
        """
        # Handle missing order book
        if order_book is None:
            return SlippageCheckResult(
                status=SlippageStatus.PRICE_UNAVAILABLE,
                target_price=target_price,
                current_price=None,
                slippage_percent=None,
                max_allowed=self.max_slippage,
                recommended_action=SlippageAction.SKIP,
            )

        # Get current market price
        if side == OrderSide.BUY:
            current_price = order_book.best_ask
        else:
            current_price = order_book.best_bid

        # Handle no liquidity
        if current_price is None:
            return SlippageCheckResult(
                status=SlippageStatus.NO_LIQUIDITY,
                target_price=target_price,
                current_price=None,
                slippage_percent=None,
                max_allowed=self.max_slippage,
                recommended_action=SlippageAction.SKIP,
            )

        # Calculate slippage
        if side == OrderSide.BUY:
            # For buys, slippage is how much more we pay
            slippage = (current_price - target_price) / target_price
        else:
            # For sells, slippage is how much less we receive
            slippage = (target_price - current_price) / target_price

        # Calculate available liquidity at best price
        available_liquidity = None
        if size:
            depth = order_book.get_depth(side, price_levels=5)
            available_liquidity = depth

        # Determine status
        if slippage <= self.max_slippage:
            status = SlippageStatus.OK
            recommended_action = SlippageAction.SKIP  # Doesn't matter, we'll execute
            execution_price = current_price
        else:
            status = SlippageStatus.EXCEEDED
            recommended_action = self.slippage_action

            if self.slippage_action == SlippageAction.LIMIT_ORDER:
                # Use target's price for limit order
                execution_price = target_price
            elif self.slippage_action == SlippageAction.EXECUTE_ANYWAY:
                execution_price = current_price
            else:
                execution_price = None

        return SlippageCheckResult(
            status=status,
            target_price=target_price,
            current_price=current_price,
            slippage_percent=slippage,
            max_allowed=self.max_slippage,
            recommended_action=recommended_action,
            execution_price=execution_price,
            available_liquidity=available_liquidity,
        )

    def check_simple_slippage(
        self,
        target_price: Decimal,
        current_price: Decimal,
        side: OrderSide,
    ) -> SlippageCheckResult:
        """Simple slippage check with direct prices.

        Args:
            target_price: Price the target executed at
            current_price: Current market price
            side: Trade side (BUY or SELL)

        Returns:
            SlippageCheckResult with analysis
        """
        # Calculate slippage
        if side == OrderSide.BUY:
            slippage = (current_price - target_price) / target_price
        else:
            slippage = (target_price - current_price) / target_price

        # Determine status
        if slippage <= self.max_slippage:
            status = SlippageStatus.OK
            execution_price = current_price
        else:
            status = SlippageStatus.EXCEEDED
            if self.slippage_action == SlippageAction.LIMIT_ORDER:
                execution_price = target_price
            elif self.slippage_action == SlippageAction.EXECUTE_ANYWAY:
                execution_price = current_price
            else:
                execution_price = None

        return SlippageCheckResult(
            status=status,
            target_price=target_price,
            current_price=current_price,
            slippage_percent=slippage,
            max_allowed=self.max_slippage,
            recommended_action=self.slippage_action if status == SlippageStatus.EXCEEDED else SlippageAction.SKIP,
            execution_price=execution_price,
        )


class LiquidityChecker:
    """Checks order book liquidity for safe execution."""

    def __init__(
        self,
        min_depth_usd: Decimal = Decimal("100"),
    ):
        """Initialize liquidity checker.

        Args:
            min_depth_usd: Minimum required depth in USD
        """
        self.min_depth_usd = min_depth_usd

    def check_liquidity(
        self,
        order_book: Optional[OrderBook],
        side: OrderSide,
        order_size: Decimal,
        price: Decimal,
    ) -> tuple[bool, str]:
        """Check if there's sufficient liquidity.

        Args:
            order_book: Current order book
            side: Trade side
            order_size: Planned order size
            price: Expected price

        Returns:
            Tuple of (is_sufficient, reason)
        """
        if order_book is None:
            return False, "no_order_book"

        # Get depth at top levels
        depth = order_book.get_depth(side, price_levels=5)
        depth_usd = depth * price

        if depth_usd < self.min_depth_usd:
            return False, f"insufficient_depth: ${depth_usd:.2f} < ${self.min_depth_usd:.2f}"

        # Check if our order would consume too much of the book
        order_usd = order_size * price
        if order_usd > depth_usd * Decimal("0.5"):
            return False, f"order_too_large: ${order_usd:.2f} > 50% of depth"

        return True, "sufficient"

    def estimate_price_impact(
        self,
        order_book: Optional[OrderBook],
        side: OrderSide,
        order_size: Decimal,
    ) -> Optional[Decimal]:
        """Estimate price impact of an order.

        Args:
            order_book: Current order book
            side: Trade side
            order_size: Order size in shares

        Returns:
            Estimated price impact as decimal, or None if can't estimate
        """
        if order_book is None:
            return None

        levels = order_book.asks if side == OrderSide.BUY else order_book.bids
        if not levels:
            return None

        best_price = levels[0].price
        remaining = order_size
        weighted_price = Decimal("0")
        total_filled = Decimal("0")

        for level in levels:
            fill_at_level = min(remaining, level.size)
            weighted_price += level.price * fill_at_level
            total_filled += fill_at_level
            remaining -= fill_at_level

            if remaining <= 0:
                break

        if total_filled == 0:
            return None

        avg_price = weighted_price / total_filled
        impact = abs(avg_price - best_price) / best_price

        return impact


def calculate_slippage(
    target_price: Decimal,
    current_price: Decimal,
    side: str,
) -> Decimal:
    """Simple slippage calculation.

    Args:
        target_price: Price target executed at
        current_price: Current market price
        side: "BUY" or "SELL"

    Returns:
        Slippage as decimal (0.05 = 5%)
    """
    if target_price == 0:
        return Decimal("1")  # 100% slippage if no target price

    if side.upper() == "BUY":
        return (current_price - target_price) / target_price
    else:
        return (target_price - current_price) / target_price


def is_slippage_acceptable(
    slippage: Decimal,
    max_slippage: Decimal,
) -> bool:
    """Check if slippage is acceptable.

    Args:
        slippage: Calculated slippage
        max_slippage: Maximum allowed slippage

    Returns:
        True if slippage is acceptable
    """
    return slippage <= max_slippage
