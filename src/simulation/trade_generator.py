"""Synthetic trade generator for simulation testing.

Generates realistic mock trades with configurable patterns
for testing copy trading logic without real market exposure.
"""

import asyncio
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Any, AsyncIterator, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TradePattern(str, Enum):
    """Trade generation patterns."""

    RANDOM = "random"  # Completely random trades
    TRENDING = "trending"  # Tends to follow recent direction
    MEAN_REVERT = "mean_revert"  # Reverses after moves
    BURST = "burst"  # Clusters of rapid trades
    WHALE = "whale"  # Large infrequent trades
    SCALPER = "scalper"  # Small frequent trades


@dataclass
class MockMarket:
    """A simulated market for generating trades."""

    market_id: str
    name: str
    outcomes: List[str]
    current_prices: Dict[str, float]  # outcome -> price
    volume_24h: float
    liquidity: float

    def update_price(self, outcome: str, delta: float) -> None:
        """Update price with bounds checking."""
        current = self.current_prices.get(outcome, 0.5)
        new_price = max(0.01, min(0.99, current + delta))
        self.current_prices[outcome] = new_price

        # Update complementary outcome for binary markets
        if len(self.outcomes) == 2:
            other = [o for o in self.outcomes if o != outcome][0]
            self.current_prices[other] = 1 - new_price


# Sample markets for simulation
SAMPLE_MARKETS = [
    MockMarket(
        market_id="0xelection2024",
        name="Will Trump win the 2024 election?",
        outcomes=["Yes", "No"],
        current_prices={"Yes": 0.52, "No": 0.48},
        volume_24h=5000000,
        liquidity=1000000,
    ),
    MockMarket(
        market_id="0xfedrate",
        name="Will the Fed cut rates in Q1 2025?",
        outcomes=["Yes", "No"],
        current_prices={"Yes": 0.65, "No": 0.35},
        volume_24h=500000,
        liquidity=200000,
    ),
    MockMarket(
        market_id="0xbtc100k",
        name="Will Bitcoin reach $100K in 2024?",
        outcomes=["Yes", "No"],
        current_prices={"Yes": 0.40, "No": 0.60},
        volume_24h=2000000,
        liquidity=500000,
    ),
    MockMarket(
        market_id="0xsuperbowl",
        name="Who will win Super Bowl LIX?",
        outcomes=["Chiefs", "Eagles"],
        current_prices={"Chiefs": 0.55, "Eagles": 0.45},
        volume_24h=3000000,
        liquidity=800000,
    ),
    MockMarket(
        market_id="0xoscar",
        name="Best Picture Oscar 2025",
        outcomes=["Oppenheimer", "Other"],
        current_prices={"Oppenheimer": 0.70, "Other": 0.30},
        volume_24h=200000,
        liquidity=50000,
    ),
]


class TradeGenerator:
    """Generates synthetic trades for simulation."""

    def __init__(
        self,
        wallets: List[str],
        trades_per_hour: int = 10,
        min_size: float = 10.0,
        max_size: float = 1000.0,
        buy_probability: float = 0.6,
        pattern: TradePattern = TradePattern.RANDOM,
        markets: Optional[List[MockMarket]] = None,
    ):
        self.wallets = wallets or ["0xSIMULATED"]
        self.trades_per_hour = trades_per_hour
        self.min_size = min_size
        self.max_size = max_size
        self.buy_probability = buy_probability
        self.pattern = pattern
        self.markets = markets or SAMPLE_MARKETS.copy()

        # State for pattern generation
        self._last_trade_side: Dict[str, str] = {}
        self._trade_count = 0
        self._burst_mode = False
        self._burst_remaining = 0

    async def trades(self) -> AsyncIterator[Dict[str, Any]]:
        """Generate trades as an async iterator."""
        interval_s = 3600 / self.trades_per_hour

        while True:
            # Adjust interval based on pattern
            actual_interval = self._calculate_interval(interval_s)
            await asyncio.sleep(actual_interval)

            trade = self._generate_trade()
            yield trade

    def _calculate_interval(self, base_interval: float) -> float:
        """Calculate interval based on pattern."""
        if self.pattern == TradePattern.BURST:
            if self._burst_mode:
                self._burst_remaining -= 1
                if self._burst_remaining <= 0:
                    self._burst_mode = False
                return random.uniform(0.5, 2.0)  # Rapid trades in burst
            elif random.random() < 0.1:  # 10% chance to start burst
                self._burst_mode = True
                self._burst_remaining = random.randint(3, 8)
                return random.uniform(0.5, 2.0)

        elif self.pattern == TradePattern.SCALPER:
            return base_interval * random.uniform(0.3, 0.7)

        elif self.pattern == TradePattern.WHALE:
            return base_interval * random.uniform(2.0, 5.0)

        # Add some randomness to base interval
        return base_interval * random.uniform(0.5, 1.5)

    def _generate_trade(self) -> Dict[str, Any]:
        """Generate a single trade."""
        self._trade_count += 1

        # Select wallet
        wallet = random.choice(self.wallets)

        # Select market (weight by volume)
        market = self._select_market()

        # Select outcome
        outcome = random.choice(market.outcomes)

        # Determine side based on pattern
        side = self._determine_side(wallet, market.market_id)

        # Calculate size based on pattern
        size = self._calculate_size()

        # Get current price
        price = market.current_prices.get(outcome, 0.5)

        # Update market price (simulate market impact)
        price_impact = (size / market.liquidity) * 0.1
        if side == "BUY":
            market.update_price(outcome, price_impact)
        else:
            market.update_price(outcome, -price_impact)

        trade = {
            "trade_id": f"sim_{uuid.uuid4().hex[:12]}",
            "wallet": wallet,
            "market_id": market.market_id,
            "market_name": market.name,
            "outcome": outcome,
            "side": side,
            "size": size,
            "price": price,
            "timestamp": datetime.utcnow(),
        }

        logger.debug(
            "generated_trade",
            trade_id=trade["trade_id"],
            market=market.name[:30],
            side=side,
            size=size,
        )

        return trade

    def _select_market(self) -> MockMarket:
        """Select a market weighted by volume."""
        total_volume = sum(m.volume_24h for m in self.markets)
        r = random.uniform(0, total_volume)

        cumulative = 0
        for market in self.markets:
            cumulative += market.volume_24h
            if r <= cumulative:
                return market

        return self.markets[-1]

    def _determine_side(self, wallet: str, market_id: str) -> str:
        """Determine trade side based on pattern."""
        key = f"{wallet}_{market_id}"

        if self.pattern == TradePattern.TRENDING:
            # 70% chance to continue last direction
            last_side = self._last_trade_side.get(key)
            if last_side and random.random() < 0.7:
                side = last_side
            else:
                side = "BUY" if random.random() < self.buy_probability else "SELL"

        elif self.pattern == TradePattern.MEAN_REVERT:
            # 70% chance to reverse
            last_side = self._last_trade_side.get(key)
            if last_side and random.random() < 0.7:
                side = "SELL" if last_side == "BUY" else "BUY"
            else:
                side = "BUY" if random.random() < self.buy_probability else "SELL"

        else:
            side = "BUY" if random.random() < self.buy_probability else "SELL"

        self._last_trade_side[key] = side
        return side

    def _calculate_size(self) -> float:
        """Calculate trade size based on pattern."""
        if self.pattern == TradePattern.WHALE:
            # Large trades, skewed toward max
            base = random.uniform(0.5, 1.0)
            return self.min_size + (self.max_size - self.min_size) * (base ** 0.5)

        elif self.pattern == TradePattern.SCALPER:
            # Small trades, skewed toward min
            base = random.uniform(0, 0.5)
            return self.min_size + (self.max_size - self.min_size) * (base ** 2)

        elif self.pattern == TradePattern.BURST:
            # Consistent size during bursts
            if self._burst_mode:
                return random.uniform(self.min_size, self.min_size + (self.max_size - self.min_size) * 0.3)

        # Default: log-normal distribution
        import math
        mu = math.log((self.min_size + self.max_size) / 2)
        sigma = 0.5
        size = random.lognormvariate(mu, sigma)
        return max(self.min_size, min(self.max_size, size))


class RealisticTradeGenerator(TradeGenerator):
    """More realistic trade generator based on actual trading patterns."""

    def __init__(
        self,
        wallets: List[str],
        trader_profile: str = "average",
        **kwargs,
    ):
        # Configure based on trader profile
        profiles = {
            "whale": {
                "trades_per_hour": 2,
                "min_size": 1000,
                "max_size": 50000,
                "pattern": TradePattern.WHALE,
            },
            "scalper": {
                "trades_per_hour": 30,
                "min_size": 10,
                "max_size": 100,
                "pattern": TradePattern.SCALPER,
            },
            "average": {
                "trades_per_hour": 5,
                "min_size": 50,
                "max_size": 500,
                "pattern": TradePattern.RANDOM,
            },
            "momentum": {
                "trades_per_hour": 8,
                "min_size": 100,
                "max_size": 1000,
                "pattern": TradePattern.TRENDING,
            },
        }

        profile_config = profiles.get(trader_profile, profiles["average"])
        merged_config = {**profile_config, **kwargs}

        super().__init__(wallets=wallets, **merged_config)

        self.trader_profile = trader_profile

        logger.info(
            "realistic_generator_init",
            profile=trader_profile,
            trades_per_hour=self.trades_per_hour,
        )
