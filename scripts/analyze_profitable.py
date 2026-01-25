#!/usr/bin/env python3
"""Analyze and filter the collected profitable accounts.

Goal: Find SYSTEMATICALLY profitable traders, not lucky streaks.
Key signals:
- High trade count (more data = less luck)
- Profit not concentrated in few big wins
- Consistent activity over time
- Good risk-adjusted metrics (Sharpe, profit factor)
"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data import DataAPIClient, ActivityType

# Load accounts from file (relative to project root)
ACCOUNTS_FILE = PROJECT_ROOT / "data" / "profitable_accounts.txt"
ANALYSIS_OUTPUT = PROJECT_ROOT / "data" / "analyzed_accounts.json"


@dataclass
class AccountMetrics:
    wallet: str
    total_pnl: float
    num_trades: int = 0
    avg_position: float = 0
    median_position: float = 0
    max_position: float = 0
    position_consistency: float = 0  # Lower std dev = more consistent sizing
    account_age_days: int = 0
    active_days: int = 0
    unique_markets: int = 0
    trades_per_week: float = 0
    pnl_per_trade: float = 0  # Average profit per trade
    pnl_per_market: float = 0  # Average profit per market
    activity_recency_days: int = 0  # Days since last trade
    buy_sell_ratio: float = 0.5  # Ratio of buys to total trades

    # Win rate (estimated from profitable vs unprofitable trades)
    win_rate: float = 0.5  # Ratio of winning trades (price improved)
    profitable_trades: int = 0
    losing_trades: int = 0

    # Recent activity metrics (for detecting dead/inactive accounts)
    trades_last_7d: int = 0
    trades_last_30d: int = 0
    recent_activity_ratio: float = 0  # % of trades in last 30 days
    is_currently_active: bool = False  # Traded in last 7 days

    # Performance trajectory (for detecting accounts in drawdown)
    pnl_last_30d: float = 0  # Estimated recent PnL (volume-weighted)
    recent_vs_historical: float = 0  # Ratio: recent performance vs historical

    # =========================================================================
    # ENHANCED DRAWDOWN METRICS (proper P/L curve analysis)
    # =========================================================================
    # Max drawdown: worst peak-to-trough decline (most important for risk)
    max_drawdown_pct: float = 0  # Worst single drawdown (0-100%)
    max_drawdown_usd: float = 0  # Max drawdown in USD terms

    # Average drawdown: typical behavior during losing streaks
    avg_drawdown_pct: float = 0  # Average of all drawdowns
    avg_drawdown_usd: float = 0  # Average drawdown in USD

    # Drawdown frequency: how often they experience significant drawdowns
    drawdown_count: int = 0  # Number of drawdowns > 5%
    severe_drawdown_count: int = 0  # Number of drawdowns > 15%
    drawdown_frequency: float = 0  # Drawdowns per 100 trades

    # Recovery metrics
    avg_recovery_trades: int = 0  # Average trades to recover from drawdown
    current_drawdown_pct: float = 0  # Are they IN a drawdown right now?

    # P/L curve smoothness (for "appealing P/L curve")
    pl_curve_smoothness: float = 0  # 0-1, higher = smoother curve
    profit_factor: float = 0  # Gross profit / gross loss

    # Additional P/L metrics (for detailed analysis)
    avg_win_size: float = 0  # Average winning trade USD
    avg_loss_size: float = 0  # Average losing trade USD
    gross_profit: float = 0  # Total gross profit USD
    gross_loss: float = 0  # Total gross loss USD
    sharpe_ratio: float = 0  # Risk-adjusted return (simplified)

    # Legacy field for backward compatibility
    estimated_drawdown_pct: float = 0  # Deprecated, use max_drawdown_pct

    # Derived
    systematic_score: float = 0  # Our custom score for "systematic" trading
    score_breakdown: dict = None  # Detailed breakdown of score components

    # Metadata
    trades_fetched: int = 0  # How many trades were actually fetched (for "load more")

    def to_dict(self):
        return {
            "wallet_address": self.wallet,  # Consistent with frontend
            "wallet": self.wallet,
            "total_pnl": self.total_pnl,
            "num_trades": self.num_trades,
            "avg_position": self.avg_position,
            "avg_position_size": self.avg_position,  # Alias for frontend
            "median_position": self.median_position,
            "max_position": self.max_position,
            "position_consistency": self.position_consistency,
            "account_age_days": self.account_age_days,
            "active_days": self.active_days,
            "unique_markets": self.unique_markets,
            "trades_per_week": self.trades_per_week,
            "pnl_per_trade": self.pnl_per_trade,
            "pnl_per_market": self.pnl_per_market,
            "activity_recency_days": self.activity_recency_days,
            "buy_sell_ratio": self.buy_sell_ratio,
            # Win rate
            "win_rate": self.win_rate,
            "profitable_trades": self.profitable_trades,
            "losing_trades": self.losing_trades,
            # Activity metrics
            "trades_last_7d": self.trades_last_7d,
            "trades_last_30d": self.trades_last_30d,
            "recent_activity_ratio": self.recent_activity_ratio,
            "is_currently_active": self.is_currently_active,
            # Performance trajectory
            "pnl_last_30d": self.pnl_last_30d,
            "recent_vs_historical": self.recent_vs_historical,
            # Enhanced drawdown metrics
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_usd": self.max_drawdown_usd,
            "avg_drawdown_pct": self.avg_drawdown_pct,
            "avg_drawdown_usd": self.avg_drawdown_usd,
            "drawdown_count": self.drawdown_count,
            "severe_drawdown_count": self.severe_drawdown_count,
            "drawdown_frequency": self.drawdown_frequency,
            "avg_recovery_trades": self.avg_recovery_trades,
            "current_drawdown_pct": self.current_drawdown_pct,
            "pl_curve_smoothness": self.pl_curve_smoothness,
            "profit_factor": self.profit_factor,
            # Additional P/L metrics
            "avg_win_size": self.avg_win_size,
            "avg_loss_size": self.avg_loss_size,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "sharpe_ratio": self.sharpe_ratio,
            # Legacy (deprecated)
            "estimated_drawdown_pct": self.max_drawdown_pct,  # Map to new field
            # Scoring
            "systematic_score": self.systematic_score,
            "score_breakdown": self.score_breakdown or {},
            # Metadata
            "trades_fetched": self.trades_fetched,
        }


def detect_market_category(event_slug: str = None, market_title: str = None) -> str:
    """
    Detect the broad category of a market based on event_slug and title.

    Categories:
        sports, politics, crypto, finance, weather, entertainment, tech, other
    """
    # Combine all text for matching
    text = ""
    if event_slug:
        text += event_slug.lower() + " "
    if market_title:
        text += market_title.lower()

    if not text:
        return "other"

    # Sports patterns (most specific first)
    sports_patterns = [
        # Soccer/Football leagues
        "fl1", "fr2", "epl", "laliga", "serie", "bundesliga", "mls", "ucl", "uel",
        "premier league", "champions league", "europa league", "copa", "fifa",
        # US Sports
        "nfl", "nba", "mlb", "nhl", "ncaa", "college football", "college basketball",
        "super bowl", "world series", "stanley cup",
        # Other sports
        "tennis", "atp", "wta", "wimbledon", "us open", "australian open",
        "golf", "pga", "masters", "british open",
        "f1", "formula 1", "nascar", "indy",
        "ufc", "mma", "boxing",
        "olympics", "world cup",
        # Generic sports terms
        "win on 20", "-win-", "-draw", "-home-", "-away-",
        "vs.", "match", "game", "playoffs", "finals",
        "team", "player", "score", "points",
    ]

    for pattern in sports_patterns:
        if pattern in text:
            return "sports"

    # Politics patterns
    politics_patterns = [
        "president", "election", "congress", "senate", "house", "governor",
        "democrat", "republican", "trump", "biden", "vote", "poll",
        "primary", "caucus", "nominee", "candidate",
        "government", "legislation", "bill", "law",
        "political", "parliament", "minister", "mayor",
    ]

    for pattern in politics_patterns:
        if pattern in text:
            return "politics"

    # Crypto patterns
    crypto_patterns = [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
        "solana", "sol", "cardano", "ada", "dogecoin", "doge",
        "token", "defi", "nft", "web3", "halving",
        "$100k", "$150k", "$200k",  # BTC price predictions
    ]

    for pattern in crypto_patterns:
        if pattern in text:
            return "crypto"

    # Finance/Economics patterns
    finance_patterns = [
        "fed", "federal reserve", "interest rate", "inflation", "gdp",
        "stock", "market", "s&p", "nasdaq", "dow", "index",
        "earnings", "revenue", "ipo", "merger", "acquisition",
        "dollar", "euro", "currency", "forex",
        "recession", "economic", "unemployment", "jobs report",
    ]

    for pattern in finance_patterns:
        if pattern in text:
            return "finance"

    # Weather patterns
    weather_patterns = [
        "weather", "temperature", "hurricane", "storm", "rain",
        "snow", "flood", "drought", "climate", "celsius", "fahrenheit",
        "hottest", "coldest", "forecast",
    ]

    for pattern in weather_patterns:
        if pattern in text:
            return "weather"

    # Entertainment/Culture patterns
    entertainment_patterns = [
        "movie", "film", "oscar", "emmy", "grammy", "award",
        "music", "album", "song", "artist", "celebrity",
        "tv show", "series", "netflix", "streaming",
        "gaming", "esports", "twitch",
    ]

    for pattern in entertainment_patterns:
        if pattern in text:
            return "entertainment"

    # Tech patterns
    tech_patterns = [
        "apple", "google", "microsoft", "meta", "amazon", "tesla",
        "ai", "artificial intelligence", "openai", "chatgpt",
        "launch", "release", "iphone", "android",
        "spacex", "nasa", "rocket", "satellite",
    ]

    for pattern in tech_patterns:
        if pattern in text:
            return "tech"

    return "other"


def load_accounts() -> list[tuple[str, float]]:
    """Load wallet,pnl pairs from file."""
    accounts = []
    with open(ACCOUNTS_FILE) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                wallet = parts[0]
                pnl = float(parts[1])
                accounts.append((wallet, pnl))
    return accounts


async def fetch_all_activities(client: DataAPIClient, wallet: str, max_pages: int = 20) -> tuple[list, list]:
    """Fetch ALL trading AND redemption activities by paginating through the API.

    Args:
        client: DataAPIClient instance
        wallet: Wallet address
        max_pages: Max pages to fetch (each page = 500 activities)

    Returns:
        Tuple of (trade_activities, redeem_activities)
    """
    all_trades = []
    all_redeems = []

    # Fetch TRADE activities
    offset = 0
    limit = 500
    for page in range(max_pages):
        try:
            activities = await client.get_activity(
                user=wallet,
                activity_type=ActivityType.TRADE,
                limit=limit,
                offset=offset,
            )
            if not activities:
                break
            all_trades.extend(activities)
            if len(activities) < limit:
                break
            offset += limit
            await asyncio.sleep(0.02)
        except Exception:
            break

    # Fetch REDEEM activities (resolved positions)
    offset = 0
    for page in range(max_pages // 2):  # Fewer pages for redeems
        try:
            activities = await client.get_activity(
                user=wallet,
                activity_type=ActivityType.REDEEM,
                limit=limit,
                offset=offset,
            )
            if not activities:
                break
            all_redeems.extend(activities)
            if len(activities) < limit:
                break
            offset += limit
            await asyncio.sleep(0.02)
        except Exception:
            break

    return all_trades, all_redeems


def calculate_drawdown_metrics(trades: list, total_pnl: float, redeems: list = None) -> dict:
    """
    Calculate proper drawdown metrics by constructing a REAL P/L curve from trades AND redemptions.

    This uses proper position tracking to compute realized P/L:
    - BUY: Accumulate position with cost basis
    - SELL: Compute realized P/L = proceeds - (avg_cost * shares_sold)
    - REDEEM: Compute resolution P/L = redeem_amount - cost_basis (winning resolution)

    Args:
        trades: List of TRADE activities
        total_pnl: Total P/L from leaderboard (for reference)
        redeems: List of REDEEM activities (resolved positions)

    Returns dict with:
        - max_drawdown_pct: Worst peak-to-trough decline as percentage
        - max_drawdown_usd: Worst decline in USD
        - avg_drawdown_pct: Average of all drawdowns
        - avg_drawdown_usd: Average drawdown in USD
        - drawdown_count: Number of drawdowns > 5%
        - severe_drawdown_count: Number of drawdowns > 15%
        - drawdown_frequency: Drawdowns per 100 trades
        - avg_recovery_trades: Average trades to recover from drawdown
        - current_drawdown_pct: Current drawdown (if in one)
        - pl_curve_smoothness: 0-1 score for P/L curve smoothness
        - profit_factor: Gross profit / gross loss
    """
    redeems = redeems or []

    if not trades or len(trades) < 10:
        return {
            "max_drawdown_pct": 0,
            "max_drawdown_usd": 0,
            "avg_drawdown_pct": 0,
            "avg_drawdown_usd": 0,
            "drawdown_count": 0,
            "severe_drawdown_count": 0,
            "drawdown_frequency": 0,
            "avg_recovery_trades": 0,
            "current_drawdown_pct": 0,
            "pl_curve_smoothness": 0.5,
            "profit_factor": 1.0,
        }

    # Combine trades and redeems, sort chronologically
    all_activities = list(trades) + list(redeems)
    sorted_activities = sorted(all_activities, key=lambda a: a.timestamp)

    # =========================================================================
    # PROPER P/L TRACKING: Track positions by condition_id and compute realized P/L
    # We use condition_id because REDEEM activities don't have token_id
    # =========================================================================
    from decimal import Decimal

    # Track positions by condition_id for resolution matching
    positions = {}  # condition_id -> {"size": Decimal, "cost": Decimal, "outcome": str}
    cumulative_pnl = []
    running_pnl = Decimal("0")
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    win_count = 0
    loss_count = 0

    for activity in sorted_activities:
        activity_type = getattr(activity.type, 'value', str(activity.type)) if hasattr(activity, 'type') else "TRADE"
        condition_id = activity.condition_id or "unknown"

        if activity_type == "REDEEM":
            # =====================================================================
            # REDEEM: Market resolved - user claiming winnings
            # Redeem amount = $1.00 per winning share
            # P/L = redeem_amount - cost_basis
            # =====================================================================
            redeem_amount = Decimal(str(activity.usd_value)) if activity.usd_value else Decimal("0")

            if condition_id in positions and redeem_amount > 0:
                pos = positions[condition_id]
                cost_basis = pos["cost"]

                # Realized P/L from resolution = what they got - what they paid
                realized_pnl = redeem_amount - cost_basis

                running_pnl += realized_pnl

                if realized_pnl > 0:
                    gross_profit += realized_pnl
                    win_count += 1
                elif realized_pnl < 0:
                    gross_loss += abs(realized_pnl)
                    loss_count += 1

                # Position is now closed
                del positions[condition_id]

            elif redeem_amount > 0:
                # Redeem without tracked position (position was from before our data window)
                # Conservatively estimate they bought at ~50% avg price
                estimated_cost = redeem_amount * Decimal("0.5")
                estimated_profit = redeem_amount - estimated_cost
                running_pnl += estimated_profit
                gross_profit += estimated_profit
                win_count += 1

        else:
            # =====================================================================
            # TRADE: Buy or Sell
            # =====================================================================
            trade_size = Decimal(str(activity.usd_value)) if activity.usd_value else Decimal("0")
            if trade_size <= 0:
                continue

            side = getattr(activity.side, 'value', str(activity.side)) if activity.side else "BUY"
            price = Decimal(str(activity.price)) if activity.price else Decimal("0.5")
            outcome = getattr(activity, 'outcome', '') or ''

            # Calculate share quantity from USD value and price
            if price > 0:
                shares = trade_size / price
            else:
                shares = trade_size

            if side.upper() == "BUY":
                # Accumulate position
                if condition_id not in positions:
                    positions[condition_id] = {"size": Decimal("0"), "cost": Decimal("0"), "outcome": outcome}
                positions[condition_id]["size"] += shares
                positions[condition_id]["cost"] += trade_size
            else:  # SELL
                if condition_id in positions:
                    pos = positions[condition_id]
                    if pos["size"] > 0:
                        # Compute average cost basis
                        avg_cost = pos["cost"] / pos["size"]
                        shares_to_sell = min(shares, pos["size"])

                        # Realized P/L = sale proceeds - cost basis
                        cost_basis = avg_cost * shares_to_sell
                        sale_proceeds = shares_to_sell * price
                        realized_pnl = sale_proceeds - cost_basis

                        running_pnl += realized_pnl

                        if realized_pnl > 0:
                            gross_profit += realized_pnl
                            win_count += 1
                        elif realized_pnl < 0:
                            gross_loss += abs(realized_pnl)
                            loss_count += 1

                        # Update position
                        pos["size"] -= shares_to_sell
                        pos["cost"] -= cost_basis
                        if pos["size"] <= 0:
                            del positions[condition_id]
                else:
                    # Selling without a tracked position (short or data gap)
                    # Treat as a realized loss at 10% of trade value (conservative)
                    estimated_loss = trade_size * Decimal("0.1")
                running_pnl -= estimated_loss
                gross_loss += estimated_loss
                loss_count += 1

        cumulative_pnl.append(float(running_pnl))

    if not cumulative_pnl or len(cumulative_pnl) < 5:
        return {
            "max_drawdown_pct": 0,
            "max_drawdown_usd": 0,
            "avg_drawdown_pct": 0,
            "avg_drawdown_usd": 0,
            "drawdown_count": 0,
            "severe_drawdown_count": 0,
            "drawdown_frequency": 0,
            "avg_recovery_trades": 0,
            "current_drawdown_pct": 0,
            "pl_curve_smoothness": 0.5,
            "profit_factor": 1.0,
        }

    # =========================================================================
    # Calculate drawdowns from the REAL P/L curve
    # =========================================================================
    peak = 0.0
    peak_idx = 0
    drawdowns = []
    current_drawdown_start = None
    min_since_peak = 0.0

    for i, pnl_point in enumerate(cumulative_pnl):
        if pnl_point > peak:
            # New peak - if we were in a drawdown, record it
            if current_drawdown_start is not None and peak > 0:
                drawdown_usd = peak - min_since_peak
                drawdown_pct = (drawdown_usd / peak) * 100
                recovery_trades = i - current_drawdown_start
                if drawdown_pct > 1:  # Only track drawdowns > 1%
                    drawdowns.append({
                        "pct": drawdown_pct,
                        "usd": drawdown_usd,
                        "recovery_trades": recovery_trades,
                    })
                current_drawdown_start = None
            peak = pnl_point
            peak_idx = i
            min_since_peak = pnl_point
        elif pnl_point < peak:
            if current_drawdown_start is None:
                current_drawdown_start = i
                min_since_peak = pnl_point
            else:
                min_since_peak = min(min_since_peak, pnl_point)

    # Check if currently in a drawdown
    current_drawdown_pct = 0.0
    if current_drawdown_start is not None and peak > 0:
        current_drawdown_usd = peak - min_since_peak
        current_drawdown_pct = (current_drawdown_usd / peak) * 100
        if current_drawdown_pct > 1:
            drawdowns.append({
                "pct": current_drawdown_pct,
                "usd": current_drawdown_usd,
                "recovery_trades": 0,
            })

    # Filter to meaningful drawdowns (> 2%)
    significant_drawdowns = [d for d in drawdowns if d["pct"] > 2]

    # Calculate metrics
    if significant_drawdowns:
        max_dd = max(significant_drawdowns, key=lambda d: d["pct"])
        max_drawdown_pct = max_dd["pct"]
        max_drawdown_usd = max_dd["usd"]
        avg_drawdown_pct = sum(d["pct"] for d in significant_drawdowns) / len(significant_drawdowns)
        avg_drawdown_usd = sum(d["usd"] for d in significant_drawdowns) / len(significant_drawdowns)

        drawdown_count = len([d for d in significant_drawdowns if d["pct"] > 5])
        severe_drawdown_count = len([d for d in significant_drawdowns if d["pct"] > 15])

        recovered_drawdowns = [d for d in significant_drawdowns if d["recovery_trades"] > 0]
        avg_recovery_trades = (
            sum(d["recovery_trades"] for d in recovered_drawdowns) / len(recovered_drawdowns)
            if recovered_drawdowns else 0
        )
    else:
        max_drawdown_pct = 0
        max_drawdown_usd = 0
        avg_drawdown_pct = 0
        avg_drawdown_usd = 0
        drawdown_count = 0
        severe_drawdown_count = 0
        avg_recovery_trades = 0

    # Drawdown frequency (per 100 trades)
    num_trades = len(cumulative_pnl)
    drawdown_frequency = (len(significant_drawdowns) / num_trades) * 100 if num_trades > 0 else 0

    # P/L curve smoothness
    final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
    if len(cumulative_pnl) > 10 and final_pnl > 0:
        expected_pnl = [i * (final_pnl / len(cumulative_pnl)) for i in range(len(cumulative_pnl))]
        deviations = [abs(actual - expected) for actual, expected in zip(cumulative_pnl, expected_pnl)]
        avg_deviation = sum(deviations) / len(deviations)
        deviation_ratio = avg_deviation / max(1, final_pnl)
        pl_curve_smoothness = max(0, min(1, 1 - deviation_ratio))
    else:
        pl_curve_smoothness = 0.5

    # Profit factor and win rate
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)
    profit_factor = min(profit_factor, 10.0)  # Cap at 10x

    actual_win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0.5

    # Average win/loss sizes
    avg_win_size = float(gross_profit / win_count) if win_count > 0 else 0
    avg_loss_size = float(gross_loss / loss_count) if loss_count > 0 else 0

    # Sharpe ratio (simplified: mean P/L / std dev of P/L per trade)
    if len(cumulative_pnl) > 10:
        trade_returns = [cumulative_pnl[i] - cumulative_pnl[i-1] for i in range(1, len(cumulative_pnl))]
        if trade_returns:
            import statistics
            mean_return = statistics.mean(trade_returns)
            std_return = statistics.stdev(trade_returns) if len(trade_returns) > 1 else 1
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0

    return {
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "max_drawdown_usd": round(max_drawdown_usd, 2),
        "avg_drawdown_pct": round(avg_drawdown_pct, 2),
        "avg_drawdown_usd": round(avg_drawdown_usd, 2),
        "drawdown_count": drawdown_count,
        "severe_drawdown_count": severe_drawdown_count,
        "drawdown_frequency": round(drawdown_frequency, 2),
        "avg_recovery_trades": int(avg_recovery_trades),
        "current_drawdown_pct": round(current_drawdown_pct, 2),
        "pl_curve_smoothness": round(pl_curve_smoothness, 3),
        "profit_factor": round(profit_factor, 2),
        "actual_win_rate": round(actual_win_rate, 3),
        "avg_win_size": round(avg_win_size, 2),
        "avg_loss_size": round(avg_loss_size, 2),
        "gross_profit": round(float(gross_profit), 2),
        "gross_loss": round(float(gross_loss), 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "win_count": win_count,
        "loss_count": loss_count,
    }


async def fetch_account_metrics(client: DataAPIClient, wallet: str, pnl: float) -> Optional[AccountMetrics]:
    """Fetch detailed metrics for an account using activity data."""
    try:
        # Get ALL trading AND redemption activity by paginating
        trades, redeems = await fetch_all_activities(client, wallet, max_pages=20)

        # Use trades for most metrics (redeems passed separately to P/L calculation)
        activities = trades  # For backward compatibility with metrics below

        if not activities:
            return None

        num_trades = len(activities)
        num_redeems = len(redeems)  # Track resolved positions
        if num_trades < 10:  # Need minimum trades for meaningful analysis
            return None

        # Position sizes from activities - ACTUAL DATA
        sizes = [float(a.usd_value) for a in activities if a.usd_value > 0]
        if not sizes:
            return None

        avg_position = sum(sizes) / len(sizes)
        sorted_sizes = sorted(sizes)
        median_position = sorted_sizes[len(sorted_sizes) // 2]
        max_position = max(sizes)

        # Position consistency (coefficient of variation - lower = more consistent)
        if avg_position > 0:
            variance = sum((s - avg_position) ** 2 for s in sizes) / len(sizes)
            std_dev = variance ** 0.5
            position_consistency = 1 - min(1, std_dev / avg_position)  # 0-1, higher = more consistent
        else:
            position_consistency = 0

        # Activity metrics - dates (include BOTH trades AND redeems for activity tracking)
        now = datetime.now()
        dates = set()
        timestamps_dt = []

        # Process trade timestamps
        for a in activities:
            try:
                dates.add(a.timestamp.strftime("%Y-%m-%d"))
                timestamps_dt.append(a.timestamp)
            except:
                pass

        # Also include redeem timestamps for activity recency
        # (if someone redeemed recently, they're still active on the platform)
        for r in redeems:
            try:
                timestamps_dt.append(r.timestamp)
                # Don't add to dates - we count active TRADING days separately
            except:
                pass

        active_days = len(dates)  # Active TRADING days only

        # Account age and activity recency - based on ALL activities (trades + redeems)
        if len(timestamps_dt) >= 2:
            sorted_ts = sorted(timestamps_dt)
            account_age_days = max(1, (sorted_ts[-1] - sorted_ts[0]).days)
            # Activity recency - days since last trade OR redeem
            activity_recency_days = (now - sorted_ts[-1]).days
        else:
            account_age_days = max(30, active_days * 2)
            activity_recency_days = 999

        trades_per_week = (num_trades / max(1, account_age_days)) * 7

        # Unique market CATEGORIES (sports, politics, crypto, etc.) - not individual markets
        categories = set()
        for a in activities:
            category = detect_market_category(
                event_slug=getattr(a, 'event_slug', None),
                market_title=getattr(a, 'market_title', None)
            )
            categories.add(category)
        unique_markets = len(categories)  # Now counts categories, not individual condition IDs

        # Buy/sell ratio - ACTUAL DATA
        buys = sum(1 for a in activities if a.side and a.side.value == "BUY")
        buy_sell_ratio = buys / num_trades if num_trades > 0 else 0.5

        # Derived metrics based on ACTUAL data
        pnl_per_trade = pnl / num_trades if num_trades > 0 else 0
        pnl_per_market = pnl / unique_markets if unique_markets > 0 else 0

        # === NEW: RECENT ACTIVITY METRICS ===
        # Count trades in last 7 and 30 days
        trades_last_7d = 0
        trades_last_30d = 0
        volume_last_30d = 0
        volume_older = 0

        for a in activities:
            try:
                days_ago = (now - a.timestamp).days
                trade_value = float(a.usd_value) if a.usd_value else 0

                if days_ago <= 7:
                    trades_last_7d += 1
                    trades_last_30d += 1
                    volume_last_30d += trade_value
                elif days_ago <= 30:
                    trades_last_30d += 1
                    volume_last_30d += trade_value
                else:
                    volume_older += trade_value
            except:
                pass

        # Also count recent redeems for activity status
        # (redeeming = still active on platform, even if not trading)
        redeems_last_7d = 0
        for r in redeems:
            try:
                days_ago = (now - r.timestamp).days
                if days_ago <= 7:
                    redeems_last_7d += 1
            except:
                pass

        # Account is active if they traded OR redeemed in last 7 days
        is_currently_active = (trades_last_7d > 0) or (redeems_last_7d > 0)
        recent_activity_ratio = trades_last_30d / num_trades if num_trades > 0 else 0

        # === NEW: PERFORMANCE TRAJECTORY (drawdown estimation) ===
        # We estimate recent P/L based on volume-weighted assumption
        # If they're profitable overall and trading recently, assume similar rate
        # If not trading recently, they might be in drawdown or abandoned

        # Rough estimate: allocate PnL proportionally to volume
        total_volume = volume_last_30d + volume_older
        if total_volume > 0 and volume_older > 0:
            # Estimate what portion of PnL came from last 30 days
            volume_ratio = volume_last_30d / total_volume
            pnl_last_30d = pnl * volume_ratio

            # Compare recent PnL rate to historical
            # (pnl per $ traded recently vs historically)
            recent_pnl_rate = pnl_last_30d / volume_last_30d if volume_last_30d > 0 else 0
            historical_pnl_rate = (pnl - pnl_last_30d) / volume_older if volume_older > 0 else 0

            if historical_pnl_rate > 0:
                recent_vs_historical = recent_pnl_rate / historical_pnl_rate
            else:
                recent_vs_historical = 1.0 if recent_pnl_rate >= 0 else 0.5
        else:
            pnl_last_30d = pnl if volume_last_30d > 0 else 0
            recent_vs_historical = 1.0

        # === CALCULATE PROPER DRAWDOWN METRICS ===
        # Analyze the full P/L curve including both trades AND redemptions
        # This captures both active trading P/L and resolution P/L
        drawdown_metrics = calculate_drawdown_metrics(trades, pnl, redeems)

        # === ESTIMATE WIN RATE ===
        # Since we don't have per-trade P/L, estimate based on overall metrics
        # A profitable trader with good PnL efficiency likely has > 50% win rate
        total_volume = sum(sizes)
        if total_volume > 0 and pnl > 0:
            # Return on volume - higher return = better win rate
            return_on_volume = pnl / total_volume
            # Map: 0% ROV -> 45% WR, 10% ROV -> 55% WR, 50% ROV -> 70% WR
            estimated_win_rate = min(0.80, 0.45 + return_on_volume * 0.5)
        elif pnl > 0:
            # Profitable but can't calculate ROV, assume decent
            estimated_win_rate = 0.52
        else:
            # Not profitable
            estimated_win_rate = 0.45

        # Use ACTUAL win rate from P/L calculation if available, else fall back to estimate
        actual_win_rate = drawdown_metrics.get("actual_win_rate")
        if actual_win_rate and actual_win_rate > 0:
            # Use real win rate from position tracking
            final_win_rate = actual_win_rate
        else:
            # Fall back to estimate, adjusted by trade frequency
            if num_trades >= 200 and pnl > 0:
                estimated_win_rate = min(0.80, estimated_win_rate + 0.03)
            elif num_trades >= 100 and pnl > 0:
                estimated_win_rate = min(0.80, estimated_win_rate + 0.02)
            final_win_rate = estimated_win_rate

        metrics = AccountMetrics(
            wallet=wallet,
            total_pnl=pnl,
            num_trades=num_trades,
            avg_position=avg_position,
            median_position=median_position,
            max_position=max_position,
            position_consistency=position_consistency,
            account_age_days=account_age_days,
            active_days=active_days,
            unique_markets=unique_markets,
            trades_per_week=trades_per_week,
            pnl_per_trade=pnl_per_trade,
            pnl_per_market=pnl_per_market,
            activity_recency_days=activity_recency_days,
            buy_sell_ratio=buy_sell_ratio,
            # Win rate (actual from P/L tracking)
            win_rate=final_win_rate,
            # Activity metrics
            trades_last_7d=trades_last_7d,
            trades_last_30d=trades_last_30d,
            recent_activity_ratio=recent_activity_ratio,
            is_currently_active=is_currently_active,
            # Performance trajectory
            pnl_last_30d=pnl_last_30d,
            recent_vs_historical=recent_vs_historical,
            # Enhanced drawdown metrics
            max_drawdown_pct=drawdown_metrics["max_drawdown_pct"],
            max_drawdown_usd=drawdown_metrics["max_drawdown_usd"],
            avg_drawdown_pct=drawdown_metrics["avg_drawdown_pct"],
            avg_drawdown_usd=drawdown_metrics["avg_drawdown_usd"],
            drawdown_count=drawdown_metrics["drawdown_count"],
            severe_drawdown_count=drawdown_metrics["severe_drawdown_count"],
            drawdown_frequency=drawdown_metrics["drawdown_frequency"],
            avg_recovery_trades=drawdown_metrics["avg_recovery_trades"],
            current_drawdown_pct=drawdown_metrics["current_drawdown_pct"],
            pl_curve_smoothness=drawdown_metrics["pl_curve_smoothness"],
            profit_factor=drawdown_metrics["profit_factor"],
            # Additional P/L metrics
            avg_win_size=drawdown_metrics.get("avg_win_size", 0),
            avg_loss_size=drawdown_metrics.get("avg_loss_size", 0),
            gross_profit=drawdown_metrics.get("gross_profit", 0),
            gross_loss=drawdown_metrics.get("gross_loss", 0),
            sharpe_ratio=drawdown_metrics.get("sharpe_ratio", 0),
            # Win/loss counts
            profitable_trades=drawdown_metrics.get("win_count", 0),
            losing_trades=drawdown_metrics.get("loss_count", 0),
            # Metadata
            trades_fetched=num_trades,
        )

        # Calculate systematic score with breakdown
        score, breakdown = calculate_systematic_score(metrics)
        metrics.systematic_score = score
        metrics.score_breakdown = breakdown

        return metrics

    except Exception as e:
        print(f"    Error fetching {wallet[:10]}...: {e}")
        return None


def calculate_systematic_score(m: AccountMetrics) -> tuple[float, dict]:
    """
    Calculate how "systematic" a trader appears to be.
    Higher = more likely to be consistently profitable.

    PHILOSOPHY: If they're profitable, don't penalize minor things.
    Focus on: PROFITABILITY, ACTIVITY, and RISK MANAGEMENT.

    Returns:
        (score, breakdown) - Score 0-100 and dict of component scores
    """
    breakdown = {}
    score = 50  # Start neutral

    # ==========================================================================
    # 1. ACTIVITY RECENCY - Must be actively trading
    # ==========================================================================
    activity_score = 0
    if m.is_currently_active:  # Traded in last 7 days
        activity_score = 20
    elif m.activity_recency_days <= 14:
        activity_score = 15
    elif m.activity_recency_days <= 30:
        activity_score = 8
    elif m.activity_recency_days <= 60:
        activity_score = 0  # Neutral, not penalized
    elif m.activity_recency_days <= 90:
        activity_score = -5  # Slight concern
    else:
        activity_score = -15  # Dead account

    # Recent trading volume bonus (no penalty for low volume)
    if m.trades_last_30d >= 50:
        activity_score += 10
    elif m.trades_last_30d >= 20:
        activity_score += 6
    elif m.trades_last_30d >= 10:
        activity_score += 3

    breakdown["activity"] = activity_score
    score += activity_score

    # ==========================================================================
    # 2. WIN RATE - Key profitability indicator
    # ==========================================================================
    win_score = 0
    if m.win_rate >= 0.65:
        win_score = 15
    elif m.win_rate >= 0.55:
        win_score = 10
    elif m.win_rate >= 0.50:
        win_score = 5
    elif m.win_rate >= 0.45:
        win_score = 2  # Still OK if profitable overall
    else:
        win_score = 0  # Don't penalize - they might have big wins

    breakdown["win_rate"] = win_score
    score += win_score

    # ==========================================================================
    # 3. PNL EFFICIENCY - Profit per trade (bonuses only, no penalties)
    # ==========================================================================
    efficiency_score = 0
    if m.pnl_per_trade >= 500:
        efficiency_score = 12
    elif m.pnl_per_trade >= 200:
        efficiency_score = 9
    elif m.pnl_per_trade >= 100:
        efficiency_score = 6
    elif m.pnl_per_trade >= 50:
        efficiency_score = 3
    elif m.pnl_per_trade >= 20:
        efficiency_score = 1
    # No penalty for low efficiency - profit is profit

    # Per-market efficiency bonus
    if m.pnl_per_market >= 5000:
        efficiency_score += 5
    elif m.pnl_per_market >= 2000:
        efficiency_score += 3

    breakdown["efficiency"] = efficiency_score
    score += efficiency_score

    # ==========================================================================
    # 4. POSITION SIZING - Bonus for consistency, minimal penalties
    # ==========================================================================
    sizing_score = 0
    if m.position_consistency >= 0.7:
        sizing_score = 8
    elif m.position_consistency >= 0.5:
        sizing_score = 5
    elif m.position_consistency >= 0.3:
        sizing_score = 2
    # No penalty for variable sizing - some strategies require it

    # Bonus for controlled max position (not penalty for large)
    if m.avg_position > 0:
        max_to_avg_ratio = m.max_position / m.avg_position
        if max_to_avg_ratio < 3:
            sizing_score += 4  # Very controlled sizing

    breakdown["sizing"] = sizing_score
    score += sizing_score

    # ==========================================================================
    # 5. MARKET FOCUS - Fewer markets = more focused strategy (GOOD)
    # ==========================================================================
    focus_score = 0
    if m.unique_markets <= 5:
        focus_score = 10  # Highly focused specialist
    elif m.unique_markets <= 10:
        focus_score = 8  # Focused strategy
    elif m.unique_markets <= 20:
        focus_score = 5  # Moderate focus
    elif m.unique_markets <= 50:
        focus_score = 2  # Some diversification
    else:
        focus_score = 0  # Very diversified (neutral)

    breakdown["focus"] = focus_score
    score += focus_score

    # ==========================================================================
    # 6. TRADE VOLUME - More data = more reliable (bonuses only)
    # ==========================================================================
    volume_score = 0
    if m.num_trades >= 500:
        volume_score = 8
    elif m.num_trades >= 200:
        volume_score = 5
    elif m.num_trades >= 100:
        volume_score = 3
    elif m.num_trades >= 50:
        volume_score = 1
    # No penalty for fewer trades - quality over quantity

    breakdown["volume"] = volume_score
    score += volume_score

    # ==========================================================================
    # 7. RISK METRICS - Drawdown and P/L curve quality
    # ==========================================================================
    risk_score = 0

    # Max drawdown - the primary risk indicator
    if m.max_drawdown_pct <= 5:
        risk_score = 12
    elif m.max_drawdown_pct <= 10:
        risk_score = 8
    elif m.max_drawdown_pct <= 15:
        risk_score = 5
    elif m.max_drawdown_pct <= 25:
        risk_score = 2
    elif m.max_drawdown_pct <= 35:
        risk_score = 0  # Neutral
    else:
        risk_score = -5  # Only penalize severe drawdowns

    # P/L curve smoothness bonus
    if m.pl_curve_smoothness >= 0.8:
        risk_score += 6
    elif m.pl_curve_smoothness >= 0.6:
        risk_score += 3
    elif m.pl_curve_smoothness >= 0.4:
        risk_score += 1

    # Profit factor bonus (no penalty)
    if m.profit_factor >= 2.0:
        risk_score += 5
    elif m.profit_factor >= 1.5:
        risk_score += 3
    elif m.profit_factor >= 1.2:
        risk_score += 1

    breakdown["risk"] = risk_score
    score += risk_score

    # ==========================================================================
    # 8. TRAJECTORY - Recent performance vs historical
    # ==========================================================================
    trajectory_score = 0
    if m.recent_vs_historical >= 1.2:
        trajectory_score = 8  # Getting better
    elif m.recent_vs_historical >= 0.9:
        trajectory_score = 4  # Consistent
    elif m.recent_vs_historical >= 0.7:
        trajectory_score = 1  # Slight decline (minimal impact)
    else:
        trajectory_score = 0  # Don't penalize - could be temporary

    breakdown["trajectory"] = trajectory_score
    score += trajectory_score

    # ==========================================================================
    # 9. ACCOUNT AGE - Bonus for longevity (no penalty for new)
    # ==========================================================================
    longevity_score = 0
    if m.account_age_days >= 365:
        longevity_score = 5
    elif m.account_age_days >= 180:
        longevity_score = 3
    elif m.account_age_days >= 90:
        longevity_score = 1
    # No penalty for new accounts - they might be great

    breakdown["longevity"] = longevity_score
    score += longevity_score

    return max(0, min(100, score)), breakdown


async def analyze_batch(accounts: list[tuple[str, float]], start_idx: int, batch_size: int = 50, concurrency: int = 10):
    """Analyze a batch of accounts with concurrent requests."""
    batch = accounts[start_idx:start_idx + batch_size]
    results = []

    async with DataAPIClient() as client:
        # Process in concurrent chunks
        for chunk_start in range(0, len(batch), concurrency):
            chunk = batch[chunk_start:chunk_start + concurrency]

            # Fetch all in parallel
            tasks = [fetch_account_metrics(client, wallet, pnl) for wallet, pnl in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in chunk_results:
                if isinstance(result, AccountMetrics):
                    results.append(result)

            progress = start_idx + chunk_start + len(chunk)
            print(f"  Analyzed {progress}/{len(accounts)}...")

            await asyncio.sleep(0.05)  # Small rate limit between chunks

    return results


async def main():
    print("Loading accounts...")
    accounts = load_accounts()
    print(f"Loaded {len(accounts):,} accounts")

    # Sort by PnL descending
    accounts.sort(key=lambda x: -x[1])

    # Show PnL distribution
    print(f"\nPnL Distribution:")
    print(f"  $1M+:    {sum(1 for _, p in accounts if p >= 1000000):,}")
    print(f"  $100k+:  {sum(1 for _, p in accounts if p >= 100000):,}")
    print(f"  $50k+:   {sum(1 for _, p in accounts if p >= 50000):,}")
    print(f"  $20k+:   {sum(1 for _, p in accounts if p >= 20000):,}")
    print(f"  $10k+:   {sum(1 for _, p in accounts if p >= 10000):,}")
    print(f"  $5k-10k: {sum(1 for _, p in accounts if 5000 <= p < 10000):,}")

    # Analyze top 500 by PnL - good balance of coverage and speed
    print(f"\nAnalyzing top 500 accounts by PnL...")

    analyzed = await analyze_batch(accounts, 0, 500, concurrency=15)

    print(f"\nAnalyzed {len(analyzed)} accounts successfully")

    # Score distribution
    print(f"\nScore Distribution:")
    print(f"  90-100: {sum(1 for a in analyzed if a.systematic_score >= 90)}")
    print(f"  80-89:  {sum(1 for a in analyzed if 80 <= a.systematic_score < 90)}")
    print(f"  70-79:  {sum(1 for a in analyzed if 70 <= a.systematic_score < 80)}")
    print(f"  60-69:  {sum(1 for a in analyzed if 60 <= a.systematic_score < 70)}")
    print(f"  50-59:  {sum(1 for a in analyzed if 50 <= a.systematic_score < 60)}")
    print(f"  <50:    {sum(1 for a in analyzed if a.systematic_score < 50)}")

    # Activity distribution
    print(f"\nActivity Distribution:")
    print(f"  Active (7d):   {sum(1 for a in analyzed if a.is_currently_active)}")
    print(f"  Active (14d):  {sum(1 for a in analyzed if a.activity_recency_days <= 14)}")
    print(f"  Active (30d):  {sum(1 for a in analyzed if a.activity_recency_days <= 30)}")
    print(f"  Stale (30-90d): {sum(1 for a in analyzed if 30 < a.activity_recency_days <= 90)}")
    print(f"  Dead (90d+):   {sum(1 for a in analyzed if a.activity_recency_days > 90)}")

    # HARD FILTER: Must be active (traded in last 14 days)
    # We don't want to copy-trade dead accounts
    active_accounts = [a for a in analyzed if a.activity_recency_days <= 14]
    print(f"\nAfter activity filter (last 14 days): {len(active_accounts)} accounts")

    # Filter by systematic score (now stricter with activity weighting)
    systematic = [a for a in active_accounts if a.systematic_score >= 65]
    print(f"After score filter (>= 65): {len(systematic)} accounts")

    # Additional quality filter: no major drawdowns
    quality = [a for a in systematic if a.max_drawdown_pct < 25]
    print(f"After max drawdown filter (<25%): {len(quality)} accounts")

    # Sort by score descending
    quality.sort(key=lambda x: (-x.systematic_score, -x.total_pnl))

    print(f"\n{'='*150}")
    print(f"TOP COPY-TRADING CANDIDATES (Active + High Score + Low Drawdown)")
    print(f"{'='*150}")
    print(f"{'Wallet':<44} {'PnL':>12} {'Trades':>6} {'7d':>4} {'30d':>4} {'Mkts':>4} {'$/Trd':>7} {'Recency':>8} {'MaxDD':>6} {'AvgDD':>6} {'PF':>5} {'Score':>6}")
    print("-" * 150)

    for m in quality[:30]:
        recency_str = f"{m.activity_recency_days}d ago" if m.activity_recency_days > 0 else "today"
        print(f"{m.wallet} {m.total_pnl:>12,.0f} {m.num_trades:>6} {m.trades_last_7d:>4} {m.trades_last_30d:>4} {m.unique_markets:>4} {m.pnl_per_trade:>7,.0f} {recency_str:>8} {m.max_drawdown_pct:>5.1f}% {m.avg_drawdown_pct:>5.1f}% {m.profit_factor:>5.2f} {m.systematic_score:>6.0f}")

    # Save results - now saving quality filtered accounts
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_collected": len(accounts),
        "total_analyzed": len(analyzed),
        "active_count": len(active_accounts),
        "systematic_count": len(systematic),
        "quality_count": len(quality),
        "filters_applied": {
            "activity_max_days": 14,
            "min_score": 65,
            "max_drawdown_pct": 25,
        },
        "activity_distribution": {
            "active_7d": sum(1 for a in analyzed if a.is_currently_active),
            "active_14d": sum(1 for a in analyzed if a.activity_recency_days <= 14),
            "active_30d": sum(1 for a in analyzed if a.activity_recency_days <= 30),
            "stale_30_90d": sum(1 for a in analyzed if 30 < a.activity_recency_days <= 90),
            "dead_90d_plus": sum(1 for a in analyzed if a.activity_recency_days > 90),
        },
        "pnl_distribution": {
            "1m_plus": sum(1 for _, p in accounts if p >= 1000000),
            "100k_plus": sum(1 for _, p in accounts if p >= 100000),
            "50k_plus": sum(1 for _, p in accounts if p >= 50000),
            "20k_plus": sum(1 for _, p in accounts if p >= 20000),
            "10k_plus": sum(1 for _, p in accounts if p >= 10000),
        },
        "score_distribution": {
            "90-100": sum(1 for a in quality if a.systematic_score >= 90),
            "80-89": sum(1 for a in quality if 80 <= a.systematic_score < 90),
            "70-79": sum(1 for a in quality if 70 <= a.systematic_score < 80),
            "65-69": sum(1 for a in quality if 65 <= a.systematic_score < 70),
        },
        "accounts": [a.to_dict() for a in quality],  # Only quality accounts!
    }

    with open(ANALYSIS_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(quality)} quality accounts to {ANALYSIS_OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
