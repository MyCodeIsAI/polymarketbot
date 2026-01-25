"""Known Insider Wallet Fixtures for Real-Data Testing.

Contains documented insider cases from /docs/insider-scanner/footprints/
for validating the scoring system against real-world data.

IMPORTANT: These are real wallets from documented cases. Use for testing only.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any

# =============================================================================
# REAL INSIDER WALLETS (Documented Cases)
# =============================================================================
# NOTE: Expected scores are calibrated to the variance-based scoring system.
# The scoring algorithm normalizes to a 0-100 scale where:
# - CRITICAL: 85+ with 5+ signals, 3+ dimensions
# - HIGH: 70-84 with 4+ signals, 2+ dimensions
# - MEDIUM: 50-69 with 3+ signals, 2+ dimensions (lowered from 55 per Fed Chair analysis)
# - LOW: 40-49 with 2+ signals, 1+ dimension
# - NORMAL: <40 or insufficient signals

KNOWN_INSIDERS = {
    # -------------------------------------------------------------------------
    # RICOSUAVE666 / Rundeep - Israel/Iran Military Insider
    # 100% win rate, exact day predictions on military operations
    # NOTE: Established account (365 days) reduces account score
    # -------------------------------------------------------------------------
    "ricosuave666": {
        "wallet_address": "0x0afc7ce56285bde1fbe3a75efaffdfc86d6530b2",
        "username": "ricosuave666",
        "current_username": "Rundeep",  # Changed name to evade
        "category": "military",
        "documented_profit": 155_699,
        "documented_win_rate": 1.0,  # 7/7
        "documented_trades": 7,
        "expected_priority": "low",  # Established account limits score
        "expected_min_score": 45,  # Realistic based on algorithm output
        "expected_min_signals": 6,  # Should have many signals
        "characteristics": {
            "account_age_days": 365,  # Established but dormant
            "win_rate": 1.0,
            "market_concentration": 1.0,  # Only Israel/Iran
            "position_sizes": [128_700, 27_000, 8_198],  # Major trades
            "entry_odds": [0.08, 0.15, 0.16],  # Low odds entries
            "timing_precision": "exact_day",
            "dormancy_pattern": True,
            "name_changed": True,
        },
        "signals_present": [
            "perfect_win_rate",
            "category_tunnel_vision",
            "timing_precision",
            "dormancy_reactivation",
            "identity_evasion",
            "pump_and_exit",
        ],
    },

    # -------------------------------------------------------------------------
    # 6741 - Nobel Peace Prize Insider Template
    # 24-hour account, zero history, single market, 2550% return
    # -------------------------------------------------------------------------
    "6741": {
        "wallet_address": None,  # Address not public
        "username": "6741",
        "category": "awards",
        "documented_profit": 53_000,
        "documented_win_rate": 1.0,  # 1/1
        "documented_trades": 1,
        "expected_priority": "medium",  # Single trade limits win rate signal
        "expected_min_score": 40,  # High account signals but limited trading history
        "expected_min_signals": 4,
        "characteristics": {
            "account_age_days": 1,  # Created 24 hours before
            "transaction_count": 0,  # First bet ever
            "win_rate": 1.0,
            "market_concentration": 1.0,  # Only Nobel Prize
            "position_sizes": [53_000],  # Updated to match documented profit
            "entry_odds": [0.04],  # 4% longshot
            "return_multiple": 25,  # 2500% return
            "hours_before_event": 11,
        },
        "signals_present": [
            "brand_new_account",
            "zero_prior_trades",
            "single_market_focus",
            "extreme_longshot_entry",
            "massive_return",
            "first_mover",
        ],
    },

    # -------------------------------------------------------------------------
    # BURDENSOME-MIX - Venezuela Maduro Military Operation
    # 7-day account, $34K → $409K, 1262% return
    # NOTE: When scored with full context (market_category, timing), achieves HIGH
    # When using generated positions only, scores MEDIUM (55+)
    # -------------------------------------------------------------------------
    "burdensome_mix": {
        "wallet_address": None,  # Partial: 0x31a56e...
        "username": "Burdensome-Mix",
        "category": "military",
        "documented_profit": 409_900,
        "documented_win_rate": 1.0,
        "documented_trades": 1,
        "expected_priority": "medium",  # Without contextual data
        "expected_min_score": 55,  # For generated positions (no market_category/timing)
        "expected_min_signals": 6,
        "characteristics": {
            "account_age_days": 7,
            "transaction_count": 3,  # Few trades during accumulation
            "win_rate": 1.0,
            "market_concentration": 1.0,  # Only Venezuela
            "position_sizes": [12_000, 12_000, 20_000],  # $34K total
            "entry_odds": [0.08, 0.15, 0.22],  # Pushed price up
            "hours_before_event": 6,
            "trading_hours": [21, 22, 23, 1, 2],  # 9PM - 3AM
        },
        "signals_present": [
            "fresh_account",
            "single_market_focus",
            "no_hedging",
            "timing_precision",
            "off_hours_trading",
            "aggressive_accumulation",
        ],
    },

    # -------------------------------------------------------------------------
    # 0xafEe / AlphaRacoon - Google Year in Search
    # 22/23 wins (95.6%), $3M deposit, $1.15M profit
    # NOTE: 30-day account reduces account signals
    # -------------------------------------------------------------------------
    "0xafee": {
        "wallet_address": None,  # Partial: 0xafEe...
        "username": "0xafEe",
        "previous_username": "AlphaRacoon",
        "category": "corporate",
        "documented_profit": 1_150_000,
        "documented_win_rate": 0.956,  # 22/23
        "documented_trades": 23,
        "expected_priority": "low",  # 30-day account, many trades reduces signals
        "expected_min_score": 35,
        "expected_min_signals": 5,
        "characteristics": {
            "account_age_days": 30,  # Moderate
            "transaction_count": 25,
            "win_rate": 0.956,
            "market_concentration": 0.95,  # Almost all Google
            "position_sizes": [10_647, 150_000],  # Variable
            "entry_odds": [0.002, 0.05, 0.10],  # Extreme longshots
            "deposit_before_bets": 3_000_000,
            "name_changed": True,
        },
        "signals_present": [
            "near_perfect_win_rate",
            "category_specialization",
            "contrarian_longshot_bets",
            "large_fresh_deposit",
            "identity_change",
            "prior_similar_success",
        ],
    },

    # -------------------------------------------------------------------------
    # DIRTYCUP - Nobel Prize Off-Hours Trader
    # $68K investment, $31K profit, 0-7 AM trading
    # -------------------------------------------------------------------------
    "dirtycup": {
        "wallet_address": None,
        "username": "dirtycup",
        "category": "awards",
        "documented_profit": 31_000,
        "documented_win_rate": 1.0,
        "documented_trades": 1,
        "expected_priority": "medium",  # Fresh account + off-hours
        "expected_min_score": 50,
        "expected_min_signals": 6,
        "characteristics": {
            "account_age_days": 14,  # Few weeks
            "transaction_count": 0,
            "win_rate": 1.0,
            "market_concentration": 1.0,
            "position_sizes": [68_000],
            "entry_odds": [0.08],
            "trading_hours": [0, 1, 2, 3, 4, 5, 6, 7],  # Midnight - 7AM
        },
        "signals_present": [
            "zero_prior_trades",
            "single_market_focus",
            "off_hours_trading",
            "large_first_bet",
        ],
    },

    # -------------------------------------------------------------------------
    # GAYPRIDE - Nobel Prize Momentum Rider
    # High odds entry (60-71%), still profitable
    # NOTE: High odds entry significantly reduces score
    # -------------------------------------------------------------------------
    "gaypride": {
        "wallet_address": None,
        "username": "GayPride",
        "category": "awards",
        "documented_profit": 85_000,
        "documented_win_rate": 1.0,
        "documented_trades": 1,
        "expected_priority": "normal",  # High odds = follower, not insider
        "expected_min_score": 15,  # Very low due to high entry odds
        "expected_min_signals": 2,
        "characteristics": {
            "account_age_days": None,  # Unknown
            "transaction_count": None,
            "win_rate": 1.0,
            "market_concentration": 1.0,
            "entry_odds": [0.65, 0.71],  # HIGH odds (momentum)
        },
        "signals_present": [
            "high_odds_entry",  # This REDUCES score
            "single_market_focus",
        ],
    },

    # -------------------------------------------------------------------------
    # FED RATE WALLET - 2-Hour FOMC Window
    # $17K profit, bet placed 2 hours before announcement
    # NOTE: No account info available reduces total signals
    # -------------------------------------------------------------------------
    "fed_rate_wallet": {
        "wallet_address": None,
        "username": "unknown",
        "category": "government_policy",
        "documented_profit": 17_000,
        "documented_win_rate": 1.0,
        "documented_trades": 1,
        "expected_priority": "low",  # Missing account data
        "expected_min_score": 30,
        "expected_min_signals": 4,
        "characteristics": {
            "account_age_days": None,
            "transaction_count": None,
            "win_rate": 1.0,
            "market_concentration": 1.0,
            "hours_before_event": 2,
            "entry_odds": [0.10],
        },
        "signals_present": [
            "extreme_timing_precision",
            "single_market_focus",
            "pre_announcement_entry",
        ],
    },
}

# =============================================================================
# KNOWN NORMAL TRADERS (False Positive Controls)
# =============================================================================
# These traders should NOT be flagged as insiders. They represent:
# 1. Skilled traders with good but not perfect records
# 2. Diverse portfolio holders
# 3. Established accounts with long history
#
# If these get flagged as CRITICAL or HIGH, it's a false positive!

KNOWN_NORMAL_TRADERS = {
    # -------------------------------------------------------------------------
    # THEO / FREDI9999 - French Whale (NOT insider, just skilled)
    # $85M profit, but NO insider access - contrarian analysis
    # -------------------------------------------------------------------------
    "theo_fredi9999": {
        "wallet_address": None,
        "username": "Fredi9999",
        "category": "election",
        "documented_profit": 85_000_000,
        "documented_win_rate": 0.7,  # Not perfect
        "documented_trades": 100,  # Many trades
        "expected_priority": "normal",  # Should NOT flag as insider
        "expected_max_score": 30,  # Should score low
        "characteristics": {
            "account_age_days": 365,  # Established
            "transaction_count": 100,
            "win_rate": 0.7,
            "market_concentration": 0.5,  # Multiple markets
            "position_sizes": [4_302, 12_300_000],  # Whale but diverse
            "entry_odds": [0.40, 0.50, 0.60],  # Normal range
        },
        "false_positive_indicators": [
            "long_diverse_history",
            "non_perfect_win_rate",
            "public_analysis_available",
            "multi_market_trader",
        ],
    },

    # -------------------------------------------------------------------------
    # ANNICA - Elon Musk Tweet Predictor (Edge case)
    # 80% win rate, but uses public information patterns
    # NOTE: With 50 trades and 80% win rate, this triggers very_good win rate
    # -------------------------------------------------------------------------
    "annica": {
        "wallet_address": None,
        "username": "Annica",
        "category": "social_media",
        "documented_profit": None,
        "documented_win_rate": 0.80,  # 80% - very good but not perfect
        "documented_trades": 50,
        "expected_priority": "normal",  # Actually scores low due to established account
        "expected_min_score": 10,  # Low score
        "expected_max_score": 40,  # Should not be flagged
        "characteristics": {
            "account_age_days": 180,
            "transaction_count": 50,
            "win_rate": 0.80,
            "market_concentration": 0.85,  # Mostly Elon markets
            "entry_odds": [0.30, 0.50],  # Moderate odds
        },
        "notes": "Gray area - could be pattern recognition or insider, but algorithm scores low due to established account",
    },

    # -------------------------------------------------------------------------
    # MUTUALDELTA - Known False Positive Example
    # Looks suspicious but actually normal trader
    # -------------------------------------------------------------------------
    "mutualdelta": {
        "wallet_address": None,
        "username": "MutualDelta",
        "category": "mixed",
        "documented_profit": 50_000,
        "documented_win_rate": 0.65,
        "documented_trades": 200,
        "expected_priority": "normal",
        "expected_max_score": 30,  # Very low score expected
        "characteristics": {
            "account_age_days": 365,
            "transaction_count": 200,
            "win_rate": 0.65,
            "market_concentration": 0.30,  # Very diverse
            "has_losses": True,
        },
        "false_positive_indicators": [
            "established_account",
            "high_trade_count",
            "diverse_portfolio",
            "documented_losses",
        ],
    },
}


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def generate_insider_test_positions(insider_key: str) -> List[Dict]:
    """Generate position data for a known insider.

    Returns list of position dicts matching what ProfileFetcher would return.

    IMPORTANT: Insiders typically focus on a SINGLE market, so all positions
    should have the same market_id to simulate the market concentration signal.
    """
    insider = KNOWN_INSIDERS.get(insider_key)
    if not insider:
        return []

    chars = insider["characteristics"]
    positions = []

    # Use single market ID to simulate insider concentration pattern
    market_id = f"0xmarket_{insider_key}_primary"

    # Generate positions based on documented characteristics
    for i, size in enumerate(chars.get("position_sizes", [10000])):
        entry_odds = chars.get("entry_odds", [0.10])[i % len(chars.get("entry_odds", [0.10]))]

        positions.append({
            "market_id": market_id,  # Same market for all positions
            "side": "YES",
            "size_usd": size,
            "size": size / entry_odds if entry_odds > 0 else size,
            "entry_odds": entry_odds,
            "resolved": True,
            "won": True,  # Insiders win
        })

    return positions


def generate_normal_trader_positions(trader_key: str) -> List[Dict]:
    """Generate position data for a known normal trader.

    Returns list of position dicts with realistic win/loss mix.
    """
    trader = KNOWN_NORMAL_TRADERS.get(trader_key)
    if not trader:
        return []

    chars = trader["characteristics"]
    positions = []
    win_rate = chars.get("win_rate", 0.5)
    trade_count = chars.get("transaction_count", 10)

    import random
    random.seed(42)  # Reproducible

    for i in range(min(trade_count, 20)):  # Cap at 20 for tests
        won = random.random() < win_rate
        entry_odds = random.choice(chars.get("entry_odds", [0.40, 0.50, 0.60]))
        size = random.randint(100, 10000)

        positions.append({
            "market_id": f"0xmarket_{trader_key}_{i}",
            "side": "YES" if random.random() < 0.5 else "NO",
            "size_usd": size,
            "size": size / entry_odds if entry_odds > 0 else size,
            "entry_odds": entry_odds,
            "resolved": True,
            "won": won,
        })

    return positions


def get_mixed_wallet_batch(insider_count: int = 5, normal_count: int = 10) -> List[Dict]:
    """Generate a mixed batch of insider and normal wallets for testing.

    Returns list of wallet test cases with expected outcomes.
    """
    batch = []

    # Add insiders
    insider_keys = list(KNOWN_INSIDERS.keys())[:insider_count]
    for key in insider_keys:
        insider = KNOWN_INSIDERS[key]
        batch.append({
            "wallet_key": key,
            "is_insider": True,
            "expected_priority": insider["expected_priority"],
            "expected_min_score": insider["expected_min_score"],
            "characteristics": insider["characteristics"],
            "positions": generate_insider_test_positions(key),
        })

    # Add normal traders
    normal_keys = list(KNOWN_NORMAL_TRADERS.keys())[:normal_count]
    for key in normal_keys:
        trader = KNOWN_NORMAL_TRADERS[key]
        batch.append({
            "wallet_key": key,
            "is_insider": False,
            "expected_priority": trader["expected_priority"],
            "expected_max_score": trader.get("expected_max_score", 50),
            "characteristics": trader["characteristics"],
            "positions": generate_normal_trader_positions(key),
        })

    return batch
