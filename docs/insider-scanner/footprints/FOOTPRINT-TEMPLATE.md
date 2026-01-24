# Insider Detection Footprint Template

## Overview

This template consolidates detection signals from all documented insider trading cases:
- **Burdensome-Mix** (Maduro/Venezuela)
- **0xafEe** (Google Year in Search)
- **Théo Cluster** (Trump Election)
- **SBet365** (Maduro + Iran)
- **dirtycup** (Nobel Prize)
- **OpenAI Wallets** (Tech launches)

---

## Tier 1: Critical Signals (Must Match 2+ to Flag)

These signals have the highest correlation with confirmed insider activity.

| Signal | Threshold | Weight | Found In |
|--------|-----------|--------|----------|
| Fresh wallet (low tx count) | < 5 prior transactions | 20 | All cases |
| Single market focus | 100% of bets in 1 market | 25 | Burdensome-Mix, dirtycup, OpenAI |
| Near-perfect win rate | > 90% in specific category | 25 | 0xafEe, SBet365 |
| Large longshot bet | > $10K on < 10% odds | 20 | All cases |
| Pre-event timing | Position built < 24h before | 20 | All cases |

---

## Tier 2: Supporting Signals (Increases Confidence)

| Signal | Threshold | Weight | Found In |
|--------|-----------|--------|----------|
| Account age at first bet | < 14 days | 15 | Burdensome-Mix, dirtycup, 6741 |
| Category specialization | > 80% in one category | 15 | All cases |
| Off-hours trading | Trades between 12 AM - 6 AM local | 10 | dirtycup, Burdensome-Mix |
| No hedging | 0 opposite positions | 10 | All cases |
| Large initial deposit | > $50K immediately before betting | 15 | 0xafEe, OpenAI |
| Centralized exchange funding | Coinbase, Kraken, Binance | 10 | Théo, Burdensome-Mix |

---

## Tier 3: Cluster/Sybil Signals

| Signal | Threshold | Weight | Found In |
|--------|-----------|--------|----------|
| Same funding source | Multiple wallets from same address | 30 | Théo cluster, Maduro cohort |
| Temporal trading correlation | Trades within 5 min of each other | 20 | Théo cluster |
| Market overlap > 90% | Same markets traded | 20 | Théo cluster |
| Account creation proximity | Created within 7 days of each other | 15 | Maduro cohort |
| Proportional position sizing | Similar % allocations | 15 | Théo cluster |

---

## Master Detection Configuration

```json
{
  "version": "1.0.0",
  "thresholds": {
    "alert_score": 70,
    "watchlist_score": 50,
    "clear_score": 30
  },

  "tier1_signals": {
    "fresh_wallet": {
      "max_prior_transactions": 5,
      "weight": 20,
      "required": false
    },
    "single_market_focus": {
      "concentration_min": 0.95,
      "weight": 25,
      "required": false
    },
    "high_win_rate": {
      "min_rate": 0.90,
      "min_trades": 3,
      "weight": 25,
      "required": false
    },
    "large_longshot_bet": {
      "min_usd": 10000,
      "max_probability": 0.10,
      "weight": 20,
      "required": false
    },
    "pre_event_timing": {
      "max_hours_before": 24,
      "weight": 20,
      "required": false
    }
  },

  "tier2_signals": {
    "account_age": {
      "max_days": 14,
      "weight": 15
    },
    "category_specialization": {
      "concentration_min": 0.80,
      "weight": 15
    },
    "off_hours_trading": {
      "suspicious_hours_utc": [0, 1, 2, 3, 4, 5, 6],
      "weight": 10
    },
    "no_hedging": {
      "opposite_positions": 0,
      "weight": 10
    },
    "large_initial_deposit": {
      "min_usd": 50000,
      "max_days_before_bet": 7,
      "weight": 15
    },
    "cex_funding": {
      "exchanges": ["coinbase", "kraken", "binance", "gemini"],
      "weight": 10
    }
  },

  "tier3_cluster_signals": {
    "same_funding_source": {
      "min_wallets": 2,
      "weight": 30
    },
    "temporal_correlation": {
      "max_seconds_apart": 300,
      "min_occurrences": 3,
      "weight": 20
    },
    "market_overlap": {
      "min_overlap": 0.90,
      "weight": 20
    },
    "creation_proximity": {
      "max_days_apart": 7,
      "weight": 15
    },
    "proportional_sizing": {
      "size_ratio_tolerance": 0.20,
      "weight": 15
    }
  },

  "high_risk_categories": [
    "politics",
    "elections",
    "geopolitical",
    "regime_change",
    "military_operations",
    "corporate_announcements",
    "product_launches",
    "ai_releases",
    "awards",
    "sports_outcomes",
    "legal_rulings"
  ],

  "high_risk_entities": [
    "openai",
    "google",
    "anthropic",
    "meta",
    "apple",
    "microsoft",
    "us_government",
    "venezuela",
    "iran",
    "nobel_committee"
  ]
}
```

---

## Scoring Algorithm

### Step 1: Calculate Individual Scores

```python
def calculate_insider_score(wallet_data):
    score = 0
    matched_signals = []

    # Tier 1 Signals
    if wallet_data['prior_tx_count'] < 5:
        score += 20
        matched_signals.append('fresh_wallet')

    if wallet_data['market_concentration'] > 0.95:
        score += 25
        matched_signals.append('single_market_focus')

    if wallet_data['win_rate'] > 0.90 and wallet_data['total_trades'] >= 3:
        score += 25
        matched_signals.append('high_win_rate')

    if wallet_data['largest_bet_usd'] > 10000 and wallet_data['largest_bet_odds'] < 0.10:
        score += 20
        matched_signals.append('large_longshot_bet')

    if wallet_data['hours_before_event'] < 24:
        score += 20
        matched_signals.append('pre_event_timing')

    # Tier 2 Signals (add if any Tier 1 matched)
    if len(matched_signals) > 0:
        if wallet_data['account_age_days'] < 14:
            score += 15
            matched_signals.append('new_account')

        if wallet_data['category_concentration'] > 0.80:
            score += 15
            matched_signals.append('category_specialist')

        if wallet_data['trading_hour_utc'] in range(0, 7):
            score += 10
            matched_signals.append('off_hours')

        if wallet_data['hedge_positions'] == 0:
            score += 10
            matched_signals.append('no_hedge')

    return min(score, 100), matched_signals
```

### Step 2: Apply Category Multiplier

```python
HIGH_RISK_MULTIPLIER = 1.25

if market_category in HIGH_RISK_CATEGORIES:
    score = min(score * HIGH_RISK_MULTIPLIER, 100)
```

### Step 3: Check for Cluster Membership

```python
def check_cluster(wallet, known_clusters):
    for cluster in known_clusters:
        if shares_funding_source(wallet, cluster):
            return cluster, 30  # bonus points
        if temporal_correlation(wallet, cluster):
            return cluster, 20
    return None, 0
```

---

## Flag File Schema

When an insider is detected, their funding sources are logged:

```json
{
  "flagged_addresses": [
    {
      "address": "0xCoinbaseHotWallet...",
      "type": "exchange_hot_wallet",
      "exchange": "coinbase",
      "first_seen": "2025-12-27T00:00:00Z",
      "associated_wallets": [
        {
          "wallet": "0x31a5...",
          "username": "Burdensome-Mix",
          "insider_score": 95,
          "event": "maduro_removal"
        }
      ],
      "alert_priority": "high"
    }
  ]
}
```

---

## Real-Time Detection Flow

```
New Trade Event
     ↓
Parse wallet address
     ↓
Check FLAG file → MATCH? → IMMEDIATE ALERT
     ↓ (no match)
Calculate insider score
     ↓
Score >= 70? → ALERT + Add to watchlist + Extract funding source
     ↓
Score >= 50? → Add to watchlist
     ↓
Score < 50? → Log and continue
```

---

## New Account Detection Flow

```
ProxyCreation Event
     ↓
Extract new wallet address
     ↓
Monitor for USDC deposit
     ↓
Deposit > $20K? → Add to watchlist (reason: large_deposit)
     ↓
Extract funding source
     ↓
Check FLAG file → MATCH? → IMMEDIATE ALERT (reason: flagged_funder)
     ↓
Continue monitoring for first trade
```

---

## Category-Specific Patterns

### Politics/Geopolitical
- **Timing:** 1-72 hours before event
- **Pattern:** New wallet, single market, large bet
- **Examples:** Maduro, elections

### Tech/AI Launches
- **Timing:** Days to 1 week before
- **Pattern:** Exact date bets, category specialist
- **Examples:** OpenAI, Google

### Awards/Announcements
- **Timing:** 6-24 hours before
- **Pattern:** Zero history, overnight trading
- **Examples:** Nobel Prize

### Corporate/Earnings
- **Timing:** Hours to days before
- **Pattern:** Options-style betting on outcomes
- **Examples:** Not yet documented

---

## Calibration Data

Based on documented cases:

| Metric | Insider Average | Normal Trader Average |
|--------|-----------------|----------------------|
| Account age at first bet | 5.2 days | 45+ days |
| Markets traded | 1.3 | 8+ |
| Win rate (large bets) | 94% | 48% |
| Largest bet / total capital | 78% | 12% |
| Category concentration | 97% | 35% |
| Hours before event | 8.5 | N/A |

---

## Sources

All patterns derived from documented cases in:
- `/footprints/BURDENSOME-MIX.md`
- `/footprints/0XAFEE.md`
- `/footprints/THEO-CLUSTER.md`
- `/footprints/SBET365.md`
- `/footprints/DIRTYCUP.md`
- `/footprints/OPENAI-WALLETS.md`

External references:
- https://github.com/pselamy/polymarket-insider-tracker
- https://app.polysights.xyz/insider-finder
- https://www.polywhaler.com/
