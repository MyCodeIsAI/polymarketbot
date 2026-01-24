"""
Core copy trade account configuration.

This module contains the CopyTradeAccount dataclass and slippage tier
configuration used across all trading modes (ghost, live, etc.).
"""

from decimal import Decimal
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


# Default tiered slippage thresholds for prediction markets
# Format: (max_price, max_slippage_percent)
# Logic: Lower probability bets have more edge, so more slippage is acceptable
# Maximum 300% at lowest tier, scaling down for higher-priced positions
DEFAULT_SLIPPAGE_TIERS: List[Tuple[Decimal, Decimal]] = [
    (Decimal("0.05"), Decimal("3.00")),    # 0-5¢: Allow up to 300% slippage (1¢→4¢ ok)
    (Decimal("0.10"), Decimal("2.00")),    # 5-10¢: Allow up to 200% slippage (8¢→24¢ ok)
    (Decimal("0.20"), Decimal("1.00")),    # 10-20¢: Allow up to 100% slippage (15¢→30¢ ok)
    (Decimal("0.35"), Decimal("0.50")),    # 20-35¢: Allow up to 50% slippage (30¢→45¢ ok)
    (Decimal("0.50"), Decimal("0.30")),    # 35-50¢: Allow up to 30% slippage (45¢→58.5¢ ok)
    (Decimal("0.70"), Decimal("0.20")),    # 50-70¢: Allow up to 20% slippage (60¢→72¢ ok)
    (Decimal("0.85"), Decimal("0.12")),    # 70-85¢: Allow up to 12% slippage (80¢→89.6¢ ok)
    (Decimal("1.00"), Decimal("0.06")),    # 85-100¢: Allow up to 6% slippage (90¢→95.4¢ ok)
]


@dataclass
class CopyTradeAccount:
    """Configuration for an account to copy trade."""

    id: int
    name: str
    wallet: str
    enabled: bool = True
    position_ratio: Decimal = Decimal("0.01")
    max_position_usd: Decimal = Decimal("500")

    # Tiered slippage (replaces flat slippage_tolerance)
    use_tiered_slippage: bool = True
    slippage_tiers: List[tuple] = field(default_factory=lambda: DEFAULT_SLIPPAGE_TIERS.copy())
    flat_slippage_tolerance: Decimal = Decimal("0.05")  # Fallback if tiered disabled

    # Keywords filter (empty = copy all)
    keywords: List[str] = field(default_factory=list)

    # Stoploss settings
    max_drawdown_percent: Decimal = Decimal("15")  # Stop copying if account drops 15%
    stoploss_triggered: bool = False

    # Advanced risk settings
    take_profit_pct: Decimal = Decimal("0")  # 0 = disabled, auto-close at this profit %
    stop_loss_pct: Decimal = Decimal("0")  # 0 = disabled, auto-close at this loss %
    max_concurrent: int = 0  # 0 = unlimited, max open positions at once
    max_holding_hours: int = 0  # 0 = disabled, auto-close after N hours
    min_liquidity: Decimal = Decimal("0")  # 0 = no minimum, skip low liquidity markets
    cooldown_seconds: int = 10  # Wait between retry attempts

    # Order execution type
    order_type: str = "market"  # "market" = instant execution, "limit" = only at same price or better

    # Tracking
    last_seen_trade_id: Optional[str] = None
    baseline_value: Optional[Decimal] = None  # P/L when we started tracking (for reference)
    current_value: Optional[Decimal] = None  # Current total P/L from copying this account
    peak_value: Optional[Decimal] = None  # High watermark - highest P/L achieved (for drawdown calc)
    open_positions_count: int = 0  # Track current open positions for max_concurrent

    def get_max_slippage_for_price(self, target_price: Decimal) -> Decimal:
        """
        Get the maximum allowed slippage percentage for a given entry price.

        Uses tiered slippage based on the principle that:
        - Low probability bets (1-10¢) have massive potential edge, so more slippage ok
        - High probability bets (80-99¢) have thin margins, so less slippage allowed
        """
        if not self.use_tiered_slippage:
            return self.flat_slippage_tolerance

        for max_price, max_slippage in self.slippage_tiers:
            if target_price <= max_price:
                return max_slippage

        # Default to tightest slippage for very high prices
        return Decimal("0.05")

    def is_slippage_acceptable(self, target_price: Decimal, actual_price: Decimal) -> tuple[bool, Decimal, Decimal]:
        """
        Check if the slippage between target and actual price is acceptable.

        Returns: (is_acceptable, actual_slippage_pct, max_allowed_slippage_pct)
        """
        if target_price <= 0:
            return (True, Decimal("0"), Decimal("1"))

        # Calculate actual slippage
        actual_slippage = (actual_price - target_price) / target_price

        # Get max allowed for this price level
        max_allowed = self.get_max_slippage_for_price(target_price)

        return (actual_slippage <= max_allowed, actual_slippage, max_allowed)

    def matches_keywords(self, market_title: str) -> bool:
        """Check if market title matches any of the keywords."""
        if not self.keywords:
            return True  # No filter = match all

        title_lower = market_title.lower()
        for kw in self.keywords:
            if kw.strip().lower() in title_lower:
                return True
        return False

    def update_pnl(self, new_pnl: Decimal) -> None:
        """
        Update the current P/L and track the high watermark.

        Args:
            new_pnl: Current total P/L (realized + unrealized) from copying this account
        """
        self.current_value = new_pnl

        # Update peak if this is a new high
        if self.peak_value is None or new_pnl > self.peak_value:
            self.peak_value = new_pnl

        # Set baseline on first update (for reference)
        if self.baseline_value is None:
            self.baseline_value = new_pnl

    def get_drawdown_percent(self) -> Optional[Decimal]:
        """
        Calculate current drawdown from peak as a percentage.

        Returns None if not enough data, otherwise returns drawdown %.
        Positive values mean we're below the peak (losing).
        """
        if self.peak_value is None or self.current_value is None:
            return None

        if self.peak_value <= 0:
            # If peak is 0 or negative, use absolute difference
            if self.current_value < self.peak_value:
                return abs(self.peak_value - self.current_value)
            return Decimal("0")

        # Drawdown = (peak - current) / peak * 100
        # Positive = below peak, Negative = above peak (shouldn't happen if peak is updated)
        drawdown = (self.peak_value - self.current_value) / self.peak_value * Decimal("100")
        return max(Decimal("0"), drawdown)

    def check_drawdown(self) -> bool:
        """
        Check if account has exceeded max drawdown from peak (high watermark).

        Uses the peak P/L (highest value achieved) as the reference point,
        NOT the starting/baseline value. This is the standard way to calculate
        drawdown in trading systems.

        Returns True if stoploss should trigger.
        """
        if self.peak_value is None or self.current_value is None:
            return False

        # Calculate drawdown from peak
        drawdown = self.get_drawdown_percent()
        if drawdown is None:
            return False

        if drawdown >= self.max_drawdown_percent:
            self.stoploss_triggered = True
            return True

        return False

    def reset_stoploss(self) -> None:
        """Reset stoploss state (use after adding more funds or manual override)."""
        self.stoploss_triggered = False
        # Don't reset peak_value - that stays as the historical high
        # User can manually reset peak by setting peak_value = current_value

    def to_dict(self) -> dict:
        """Convert account to dictionary for API/serialization."""
        # Convert slippage tiers to a serializable format
        tiers_dict = {
            f"{int(float(max_price) * 100)}": float(max_slippage) * 100  # Store as percentage
            for max_price, max_slippage in self.slippage_tiers
        }
        return {
            "id": self.id,
            "name": self.name,
            "target_wallet": self.wallet,
            "enabled": self.enabled,
            "position_ratio": str(self.position_ratio),
            "max_position_usd": str(self.max_position_usd),
            "use_tiered_slippage": self.use_tiered_slippage,
            "flat_slippage_tolerance": str(self.flat_slippage_tolerance),
            "slippage_tiers": tiers_dict,  # e.g., {"5": 300, "10": 200, ...}
            "keywords": self.keywords,
            "max_drawdown_percent": str(self.max_drawdown_percent),
            "stoploss_triggered": self.stoploss_triggered,
            "baseline_value": str(self.baseline_value) if self.baseline_value else None,
            "current_value": str(self.current_value) if self.current_value else None,
            "peak_value": str(self.peak_value) if self.peak_value else None,
            "drawdown_percent": float(self.get_drawdown_percent()) if self.get_drawdown_percent() is not None else None,
            # Advanced risk settings
            "take_profit_pct": float(self.take_profit_pct),
            "stop_loss_pct": float(self.stop_loss_pct),
            "max_concurrent": self.max_concurrent,
            "max_holding_hours": self.max_holding_hours,
            "min_liquidity": float(self.min_liquidity),
            "cooldown_seconds": self.cooldown_seconds,
            "open_positions_count": self.open_positions_count,
            "order_type": self.order_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CopyTradeAccount":
        """Create account from dictionary."""
        account = cls(
            id=data['id'],
            name=data['name'],
            wallet=data.get('wallet', data.get('target_wallet', '')),
            enabled=data.get('enabled', True),
            position_ratio=Decimal(str(data.get('position_ratio', '0.01'))),
            max_position_usd=Decimal(str(data.get('max_position_usd', '500'))),
            use_tiered_slippage=data.get('use_tiered_slippage', True),
            flat_slippage_tolerance=Decimal(str(data.get('flat_slippage_tolerance', '0.05'))),
            keywords=data.get('keywords', []),
            max_drawdown_percent=Decimal(str(data.get('max_drawdown_percent', '15'))),
            stoploss_triggered=data.get('stoploss_triggered', False),
            take_profit_pct=Decimal(str(data.get('take_profit_pct', '0'))),
            stop_loss_pct=Decimal(str(data.get('stop_loss_pct', '0'))),
            max_concurrent=int(data.get('max_concurrent', 0)),
            max_holding_hours=int(data.get('max_holding_hours', 0)),
            min_liquidity=Decimal(str(data.get('min_liquidity', '0'))),
            cooldown_seconds=int(data.get('cooldown_seconds', 10)),
            order_type=data.get('order_type', 'market'),
        )
        # Load P/L tracking values if present
        if data.get('baseline_value'):
            account.baseline_value = Decimal(str(data['baseline_value']))
        if data.get('current_value'):
            account.current_value = Decimal(str(data['current_value']))
        if data.get('peak_value'):
            account.peak_value = Decimal(str(data['peak_value']))
        return account
