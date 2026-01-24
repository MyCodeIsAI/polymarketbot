# Polymarket Insider Detection: Comprehensive Scoring System

## Overview

This scoring system is derived from **12+ documented insider trading cases** across multiple categories. It uses a multi-dimensional approach combining:

1. **Account Signals** - Wallet age, transaction history, funding
2. **Trading Signals** - Position sizing, timing, win rates
3. **Behavioral Signals** - Category focus, patterns, evasion
4. **Contextual Signals** - Market type, event timing, news correlation
5. **Cluster Signals** - Multi-wallet coordination, funding source overlap

---

## Sample Size & Confidence

### Documented Cases (12+)

| Case | Category | Profit | Win Rate | Key Signal |
|------|----------|--------|----------|------------|
| Burdensome-Mix | Geopolitical | $409K | 100% | Fresh wallet, timing |
| 0xa72D | Geopolitical | $75K | 100% | Same cohort |
| SBet365 | Geopolitical | $145K+ | 100% | Repeat offender |
| 0xafEe | Tech/Corporate | $1.15M | 95.6% | Category specialist |
| Théo Cluster (11 wallets) | Elections | $85M | 100% | Multi-account |
| dirtycup | Awards | ~$31K | 100% | Zero history |
| 6741 | Awards | $53K | 100% | 24hr old account |
| ricosuave666 | Military | $155K | 100% (7/7) | Exact day timing |
| bigwinner01 | Policy | $250M+ | High | 30-min pre-timing |
| Portugal cluster | Elections | €5M+ vol | 100% | Exit poll timing |
| Iran wallets (4) | Military | Mixed | TBD | Synchronized |
| OpenAI wallets | Tech | $20K+ | 100% | Product launches |
| Annica | Social Media | $267K | 80% | Pattern specialist |

---

## Scoring Dimensions

### Dimension 1: Account Signals (Max 25 points)

```
ACCOUNT_AGE_SCORE:
  < 1 day:    15 points
  1-7 days:   12 points
  7-14 days:  8 points
  14-30 days: 4 points
  > 30 days:  0 points

TRANSACTION_HISTORY_SCORE:
  0 prior transactions:  10 points
  1-2 transactions:      8 points
  3-5 transactions:      5 points
  6-10 transactions:     2 points
  > 10 transactions:     0 points

TOTAL DIMENSION 1: min(age + history, 25)
```

**Rationale:** 10/12 cases involved accounts < 14 days old with < 5 prior transactions.

---

### Dimension 2: Trading Signals (Max 35 points)

```
POSITION_SIZE_SCORE (with cumulative tracking):

  Step 1: Calculate CUMULATIVE position (sum all entries same side)
  Step 2: Apply liquidity adjustment (position / market_liquidity)
  Step 3: Score based on cumulative total:

  > $100K cumulative:    12 points
  $50K-$100K:            10 points
  $20K-$50K:             7 points
  $10K-$20K:             4 points
  $5K-$10K:              2 points (variance allowance)
  < $5K:                 0 points

  BONUS: Split entry pattern detected (+2 points)
  (When avg entry < 50% of cumulative = evasion behavior)

WIN_RATE_SCORE (category-specific, min 3 trades):
  100%:                  15 points
  90-99%:                12 points
  80-89%:                8 points (Annica case)
  70-79%:                4 points
  < 70%:                 0 points

ODDS_AT_ENTRY_SCORE (with variance for momentum riders):
  < 5% (extreme longshot):   8 points
  5-10%:                     6 points
  10-20%:                    4 points
  20-35%:                    2 points
  35-60%:                    1 point (variance: GayPride case)
  > 60%:                     0 points

TOTAL DIMENSION 2: min(size + winrate + odds, 35)
```

**Position Size Observed Ranges:**
- Minimum: $1,167 (Iran wallet)
- Median: $25,000 - $40,000
- Maximum: $340M (bigwinner01)
- Most insiders: $17K - $68K range

**Rationale:** Insiders bet large on longshots and win consistently. BUT allow variance - some enter at higher odds (GayPride at 60-71%), some have smaller positions split across entries.

---

### Dimension 3: Behavioral Signals (Max 25 points)

```
MARKET_CONCENTRATION_SCORE:
  100% in 1 market:       10 points
  > 90% in 1 category:    8 points
  > 80% in 1 category:    5 points
  > 50% in 1 category:    2 points
  < 50%:                  0 points

TRADING_TIME_SCORE:
  0-6 AM UTC:             5 points  (off-hours)
  Weekend trading:        3 points
  Normal hours:           0 points

EVASION_BEHAVIOR_SCORE:
  Username change:        5 points
  Immediate withdrawal:   5 points
  Account went dormant:   3 points
  None detected:          0 points

HEDGE_POSITION_SCORE:
  No hedging at all:      5 points
  Minimal hedging:        2 points
  Normal hedging:         0 points

TOTAL DIMENSION 3: min(concentration + time + evasion + hedge, 25)
```

**Rationale:** Insiders show tunnel vision (single market), trade at odd hours, don't hedge, and often try to hide after winning.

---

### Dimension 4: Contextual Signals (Max 20 points)

```
MARKET_CATEGORY_SCORE:
  Military operations:     8 points
  Government policy:       7 points
  Elections:               6 points
  Corporate announcements: 5 points
  Awards/committees:       5 points
  Sports (injury/trades):  4 points
  Tech product launches:   4 points
  Social media activity:   2 points
  Other:                   0 points

EVENT_TIMING_SCORE:
  < 6 hours before event:   8 points
  6-24 hours:               6 points
  24-72 hours:              4 points
  > 72 hours:               2 points
  No event correlation:     0 points

NEWS_CORRELATION_SCORE:
  Bet placed, news broke, bet won:  4 points
  Pattern repeated 2+ times:        4 points (bonus)
  No clear correlation:             0 points

TOTAL DIMENSION 4: min(category + timing + news, 20)
```

**Rationale:** Certain categories have higher insider risk due to information asymmetry. Timing correlation with news is strong indicator.

---

### Dimension 5: Cluster Signals (Max 20 points, if applicable)

```
SAME_FUNDING_SOURCE_SCORE:
  Same deposit address as flagged wallet:   15 points
  Same exchange origin:                     8 points
  No match:                                 0 points

SYNCHRONIZED_TRADING_SCORE:
  Trades within 5 min of flagged wallet:    10 points
  Trades within 1 hour:                     6 points
  Same day:                                 3 points
  No correlation:                           0 points

MARKET_OVERLAP_SCORE:
  > 90% same markets as flagged cluster:    10 points
  > 70%:                                    6 points
  > 50%:                                    3 points
  < 50%:                                    0 points

TOTAL DIMENSION 5: min(funding + sync + overlap, 20)
```

**Rationale:** Théo cluster (11 wallets), Maduro cohort (3 wallets), Iran wallets (4 synchronized) all showed coordination.

---

## Variance & Overfitting Prevention

**CRITICAL: See `VARIANCE-CALIBRATION.md` for full details.**

### Key Principles

1. **Soft Thresholds**: Use gradients, not hard cutoffs
2. **Signal Correlation**: Require multiple signals across dimensions
3. **Confidence Intervals**: Report score ranges, not point estimates
4. **Cumulative Positions**: Sum all entries, don't just look at singles

### Minimum Signal Requirements

| Priority | Min Score | Min Signals | Min Dimensions |
|----------|-----------|-------------|----------------|
| CRITICAL | 85+ | 5+ signals | 3+ dimensions |
| HIGH | 70-84 | 4+ signals | 2+ dimensions |
| MEDIUM | 55-69 | 3+ signals | 2+ dimensions |
| LOW | 40-54 | 2+ signals | Any |

**Single-dimension flags are automatically downgraded.**

---

## Composite Score Calculation

```python
def calculate_insider_score(wallet, market=None):
    scores = {
        'account': calculate_account_score(wallet),        # max 25
        'trading': calculate_trading_score(wallet),        # max 35
        'behavioral': calculate_behavioral_score(wallet),   # max 25
        'contextual': calculate_contextual_score(wallet, market),  # max 20
        'cluster': calculate_cluster_score(wallet)         # max 20 (bonus)
    }

    # Count triggered signals for validation
    signal_count = count_triggered_signals(scores)
    active_dimensions = count_active_dimensions(scores)

    # Base score (max 105 before normalization)
    base_score = sum([
        scores['account'],
        scores['trading'],
        scores['behavioral'],
        scores['contextual']
    ])

    # Cluster bonus (additive, can push over 100)
    cluster_bonus = scores['cluster']

    # Normalize to 0-100 scale
    normalized = min((base_score / 105) * 100, 100)

    # Add cluster factor (can exceed 100 for high-priority alerts)
    final_score = normalized + (cluster_bonus * 0.5)

    # VARIANCE: Calculate confidence interval
    confidence_width = 10 if signal_count < 3 else 7 if signal_count < 5 else 5

    # VARIANCE: Single-dimension downgrade
    if final_score >= 70 and active_dimensions < 2:
        final_score = min(final_score, 69)
        downgraded = True
    else:
        downgraded = False

    return {
        'score': min(final_score, 100),
        'confidence_low': max(final_score - confidence_width, 0),
        'confidence_high': min(final_score + confidence_width, 100),
        'breakdown': scores,
        'signal_count': signal_count,
        'active_dimensions': active_dimensions,
        'downgraded': downgraded,
        'priority': get_priority_level(final_score)
    }
```

---

## Confidence Modifiers

### Positive Modifiers (Increase Confidence)

| Factor | Modifier |
|--------|----------|
| Win resolved (bet won) | +15% confidence |
| Pattern matches known insider | +20% confidence |
| Multiple signals in same dimension | +10% confidence |
| Cluster membership confirmed | +25% confidence |
| Repeat behavior (same wallet) | +20% confidence |

### Negative Modifiers (Decrease Confidence)

| Factor | Modifier |
|--------|----------|
| Bet lost (like mutualdelta) | -30% confidence |
| Controllable outcome market | -15% confidence |
| High trade volume (whale, not insider) | -10% confidence |
| Long account history with varied bets | -20% confidence |
| Public explanation provided | -5% confidence |

---

## Priority Levels

```python
def get_priority_level(score):
    if score >= 85:
        return "CRITICAL"      # Immediate alert, likely insider
    elif score >= 70:
        return "HIGH"          # Strong indicators, add to watchlist + alert
    elif score >= 55:
        return "MEDIUM"        # Notable signals, add to watchlist
    elif score >= 40:
        return "LOW"           # Some signals, monitor passively
    else:
        return "NORMAL"        # No significant signals
```

### Action Matrix

| Priority | Score | Actions |
|----------|-------|---------|
| CRITICAL | 85+ | Immediate alert (all channels), add to FLAG file, extract funding source, detailed logging |
| HIGH | 70-84 | Alert (Discord + browser), add to watchlist, monitor trades |
| MEDIUM | 55-69 | Add to watchlist, daily digest alert |
| LOW | 40-54 | Log for analysis, weekly review |
| NORMAL | < 40 | No action |

---

## Category-Specific Scoring Adjustments

### Military/Geopolitical Markets

```python
MILITARY_BOOST = 1.3  # 30% score multiplier

if market.category in ['military', 'geopolitical', 'regime_change']:
    if wallet.is_new and wallet.single_market_focus:
        score *= MILITARY_BOOST
```

**Rationale:** Military operations require classified knowledge. Fresh wallets betting only on these markets are highly suspicious.

### Election Markets

```python
def election_adjustment(wallet, market, current_time):
    hours_to_close = market.close_time - current_time

    # Increase suspicion in final hours
    if hours_to_close < 2:
        return 1.25  # 25% boost
    elif hours_to_close < 6:
        return 1.15  # 15% boost
    elif hours_to_close < 24:
        return 1.05  # 5% boost
    return 1.0
```

**Rationale:** Exit polls circulate hours before results. Final-hour trading is highest risk.

### Tech/Corporate Markets

```python
TECH_EMPLOYEE_INDICATORS = {
    'exact_date_prediction': 20,  # OpenAI, Google cases
    'product_name_precision': 15,
    'repeated_company_wins': 25
}
```

**Rationale:** Corporate employees know launch timelines. Exact date predictions are strong signals.

---

## Real-Time Scoring Pipeline

```
New Trade Event
       ↓
┌──────────────────┐
│ Extract Features │
├──────────────────┤
│ - Wallet age     │
│ - Position size  │
│ - Market type    │
│ - Entry odds     │
│ - Trading time   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Calculate Scores │
├──────────────────┤
│ D1: Account      │
│ D2: Trading      │
│ D3: Behavioral   │
│ D4: Contextual   │
│ D5: Cluster      │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Apply Modifiers  │
├──────────────────┤
│ - Category boost │
│ - Time factors   │
│ - Confidence adj │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Determine Action │
├──────────────────┤
│ CRITICAL → Alert │
│ HIGH → Watchlist │
│ MEDIUM → Log     │
│ LOW → Monitor    │
└──────────────────┘
```

---

## Backtesting Against Known Insiders

### Expected Scores for Documented Cases

| Case | Expected Score | Key Scoring Dimensions |
|------|---------------|------------------------|
| Burdensome-Mix | 92-98 | Account (23), Trading (30), Behavioral (20), Contextual (18) |
| ricosuave666 | 88-95 | Trading (35), Behavioral (22), Contextual (16) |
| 0xafEe | 85-92 | Trading (33), Account (18), Behavioral (18) |
| dirtycup | 80-88 | Account (25), Behavioral (20), Contextual (15) |
| 6741 | 82-90 | Account (25), Trading (25), Contextual (15) |
| Théo cluster | 75-85 | Cluster (20), Trading (28), Behavioral (15) |
| bigwinner01 | 70-82 | Trading (30), Contextual (18) (less account signal) |
| Annica | 55-68 | Trading (25), Behavioral (20) (lower due to no event timing) |

### Validation Criteria

```python
def validate_scoring_system():
    known_insiders = load_known_insiders()
    normal_traders = load_sample_normal_traders()

    # All known insiders should score > 70
    for insider in known_insiders:
        score = calculate_insider_score(insider)
        assert score > 70, f"Insider {insider.name} scored {score}, expected > 70"

    # < 5% of normal traders should score > 70
    high_scoring_normal = [t for t in normal_traders if calculate_insider_score(t) > 70]
    false_positive_rate = len(high_scoring_normal) / len(normal_traders)
    assert false_positive_rate < 0.05, f"False positive rate {false_positive_rate} too high"
```

---

## Special Detection Rules

### Rule 1: Funding Source Match (Instant Flag)

```python
if wallet.funding_source in FLAG_FILE:
    return {
        'score': 95,
        'priority': 'CRITICAL',
        'reason': 'FLAGGED_FUNDER',
        'linked_insider': FLAG_FILE[wallet.funding_source]
    }
```

### Rule 2: Perfect Category Win Rate (Instant High)

```python
if category_win_rate == 1.0 and category_trades >= 3:
    score = max(score, 75)
    add_flag('PERFECT_WIN_RATE', category)
```

### Rule 3: Pre-Event Cluster Formation

```python
if market.has_upcoming_resolution(hours=24):
    new_wallets = get_new_wallets_entering(market, hours=6)
    if len(new_wallets) >= 3:
        for wallet in new_wallets:
            score = max(score, 70)
            add_flag('PRE_EVENT_CLUSTER', len(new_wallets))
```

### Rule 4: Name Change Detection

```python
if wallet.username_changed_recently(days=7):
    if wallet.recent_large_win:
        score += 10
        add_flag('EVASION_BEHAVIOR', 'name_change_after_win')
```

---

## Implementation Notes

### Database Schema Addition

```sql
CREATE TABLE insider_scores (
    id INTEGER PRIMARY KEY,
    wallet_address TEXT NOT NULL,
    score REAL NOT NULL,
    priority TEXT NOT NULL,
    dimension_account REAL,
    dimension_trading REAL,
    dimension_behavioral REAL,
    dimension_contextual REAL,
    dimension_cluster REAL,
    confidence REAL,
    flags TEXT,  -- JSON array of triggered rules
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    market_id TEXT,  -- optional, if calculated for specific market
    UNIQUE(wallet_address, market_id)
);

CREATE INDEX idx_scores_priority ON insider_scores(priority);
CREATE INDEX idx_scores_score ON insider_scores(score DESC);
```

### Caching Strategy

- Cache wallet base scores for 1 hour
- Recalculate on new trade
- Cache market category risks indefinitely
- Update cluster associations daily

---

## Sources & Validation Data

All scoring weights derived from documented cases in:
- `/footprints/BURDENSOME-MIX.md`
- `/footprints/0XAFEE.md`
- `/footprints/THEO-CLUSTER.md`
- `/footprints/SBET365.md`
- `/footprints/DIRTYCUP.md`
- `/footprints/OPENAI-WALLETS.md`
- `/footprints/RICOSUAVE666.md`
- `/footprints/BIGWINNER01.md`
- `/footprints/PORTUGAL-INSIDERS.md`
- `/footprints/IRAN-STRIKE-WALLETS.md`
- `/footprints/ANNICA.md`

External references:
- Polysights 85% accuracy rate
- Lookonchain wallet analysis methodology
- Chainalysis clustering heuristics

---

## Cumulative Position Tracking

**CRITICAL: Insiders often split large bets into multiple smaller entries.**

### Position Aggregation Logic

```python
def get_cumulative_position(wallet, market_id):
    """
    Sum all positions on same side for same market.
    Example: 5x $10K YES bets = $50K cumulative YES position
    """
    trades = get_wallet_trades(wallet, market_id)

    yes_positions = [t for t in trades if t.side == 'YES']
    no_positions = [t for t in trades if t.side == 'NO']

    return {
        'yes_total': sum(t.size for t in yes_positions),
        'no_total': sum(t.size for t in no_positions),
        'yes_count': len(yes_positions),
        'no_count': len(no_positions),
        'dominant_side': 'YES' if sum(t.size for t in yes_positions) > sum(t.size for t in no_positions) else 'NO',
        'dominant_size': max(sum(t.size for t in yes_positions), sum(t.size for t in no_positions)),
        'is_split_entry': len(yes_positions) > 1 or len(no_positions) > 1,
        'avg_entry_size': (sum(t.size for t in trades) / len(trades)) if trades else 0
    }

def detect_split_entry_pattern(position_data):
    """
    Split entry = multiple smaller bets summing to large position.
    This is evasion behavior - insiders trying to avoid detection.
    """
    if not position_data['is_split_entry']:
        return False

    # If average entry is less than 50% of total = split pattern
    dominant_size = position_data['dominant_size']
    avg_size = position_data['avg_entry_size']

    if avg_size < dominant_size * 0.5:
        return True  # Split entry detected

    return False
```

### Position Size Observed Ranges (17 Cases)

| Percentile | Single Entry | Cumulative Total |
|------------|--------------|------------------|
| Minimum | $1,167 | $2,000 |
| 25th | $8,000 | $17,000 |
| Median | $25,000 | $40,000 |
| 75th | $68,000 | $155,000 |
| Maximum | $340M | $85M (cluster) |

### Liquidity-Adjusted Scoring

```python
def liquidity_adjusted_score(position_size, market):
    """
    $50K on a $100K liquidity market = market mover
    $50K on a $10M liquidity market = small fish
    """
    liquidity_ratio = position_size / market.current_liquidity

    if liquidity_ratio > 0.10:  # >10% of liquidity
        return 12  # Maximum score
    elif liquidity_ratio > 0.05:
        return 10
    elif liquidity_ratio > 0.02:
        return 7
    elif liquidity_ratio > 0.01:
        return 4
    else:
        return 0  # Small relative to market
```
