# Variance Calibration: Preventing Overfitting in Insider Detection

## The Overfitting Problem

With 17 documented insider cases, we have enough patterns to identify signals BUT not enough to create a perfect profile. The next insider:
- **WILL NOT** match all signals perfectly
- **MAY** have signals we haven't seen yet
- **COULD** fail some signals that most insiders triggered

**Key Principle: Build for variance, not perfection.**

---

## Position Size Analysis (From 17 Cases)

### Raw Data Extracted

| Case | Min Position | Max Position | Cumulative Total | Entry Pattern |
|------|--------------|--------------|------------------|---------------|
| 6741 | $2K | $50K | $2K-$50K | Single entry |
| dirtycup | $68K | $68K | $68K | Single entry |
| Burdensome-Mix | $12K | $20K | $34K | Multiple entries |
| 0xafEe | $10K | Unknown | $3M+ cumulative | Multiple markets |
| ricosuave666 | $8K | Unknown | $155K+ | Multiple markets |
| bigwinner01 | $100M | $340M | $250M+ | Split platforms |
| Annica | Unknown | Unknown | $267K total | Many small bets |
| GayPride | $50K+ | Unknown | ~$85K | Accumulation |
| FED-RATE | $17K | $17K | $17K | Single entry |
| Théo Cluster | $3.5M | $12.3M | $24.7M (4 accts) | Multi-wallet |
| Iran Wallets | $1.2K | $9.9K | $17.8K (cluster) | Synchronized |
| SBet365 | $25K | Unknown | $25K+ | Multiple events |
| WLFI | $34K | $34K | $34K | Single entry |
| OpenAI | $20K | $40K | $20K-$40K | Multiple accounts |
| Portugal | N/A | N/A | €5M+ (window) | Market-wide surge |

### Position Size Thresholds (With Variance Bands)

**Single Position Sizing:**

| Threshold | Observed Range | Recommended Detection | Variance Band |
|-----------|----------------|----------------------|---------------|
| Micro | $1K - $5K | Low signal alone | ±50% |
| Small | $5K - $20K | Moderate signal | ±40% |
| Medium | $20K - $100K | Elevated signal | ±30% |
| Large | $100K - $1M | High signal | ±25% |
| Whale | $1M+ | Very high signal | ±20% |

**Why Variance Bands:**
- $17K position (FED-RATE) and $68K position (dirtycup) are both single-bet insiders
- Don't penalize for being "outside" observed range
- Allow 30-50% variance on thresholds

### Cumulative Position Logic

```python
def calculate_cumulative_position(wallet, market_id):
    """
    Sum all positions on same side for same market.
    Insiders often enter multiple times to avoid detection.
    """
    trades = get_trades(wallet, market_id)

    # Group by side (YES/NO)
    yes_total = sum(t.size for t in trades if t.side == 'YES')
    no_total = sum(t.size for t in trades if t.side == 'NO')

    return {
        'yes_cumulative': yes_total,
        'no_cumulative': no_total,
        'dominant_side': 'YES' if yes_total > no_total else 'NO',
        'dominant_size': max(yes_total, no_total),
        'entry_count': len(trades),
        'avg_entry_size': (yes_total + no_total) / len(trades) if trades else 0
    }

def position_size_score(cumulative_data, variance_factor=1.3):
    """
    Score with variance-adjusted thresholds.
    variance_factor: Multiplier to widen detection bands.
    """
    size = cumulative_data['dominant_size']

    # Base thresholds (adjusted by variance)
    thresholds = {
        'micro': 5000 * variance_factor,
        'small': 20000 * variance_factor,
        'medium': 100000 * variance_factor,
        'large': 1000000 * variance_factor
    }

    if size < thresholds['micro']:
        return 0  # Likely noise
    elif size < thresholds['small']:
        return 4  # Small signal
    elif size < thresholds['medium']:
        return 7  # Medium signal
    elif size < thresholds['large']:
        return 10  # Large signal
    else:
        return 12  # Whale signal

    # Bonus for split entry pattern (evasion behavior)
    if cumulative_data['entry_count'] > 1:
        if cumulative_data['avg_entry_size'] < size * 0.5:
            return score + 2  # Split entry bonus
```

---

## Signal Variance Analysis

### Signals That ALWAYS Applied (100% of cases)

These are safe to weight heavily:

| Signal | Cases | Confidence |
|--------|-------|------------|
| Bet on correct outcome (won) | 15/17 | 88% (2 pending) |
| Single category focus (>70%) | 16/17 | 94% |
| Bet placed before event resolution | 17/17 | 100% |
| No hedging positions | 16/17 | 94% |

### Signals That MOSTLY Applied (70-90% of cases)

Weight moderately, allow for exceptions:

| Signal | Cases | Notes |
|--------|-------|-------|
| Account age < 30 days | 12/17 (70%) | bigwinner01, Annica had older accounts |
| Entry odds < 30% | 12/17 (70%) | GayPride entered at 60-71% |
| Zero or < 5 prior trades | 11/17 (65%) | Some had established histories |
| Single market only | 12/17 (70%) | Some spread across related markets |

### Signals That SOMETIMES Applied (40-70% of cases)

Lower weight, supporting signals only:

| Signal | Cases | Notes |
|--------|-------|-------|
| Account age < 7 days | 9/17 (53%) | Highly variable |
| Off-hours trading | 6/17 (35%) | Time zone dependent |
| Username change | 2/17 (12%) | Rare evasion behavior |
| Direct CEX funding | 8/17 (47%) | Some used DEX/bridges |

### Signals That RARELY Applied (< 40% of cases)

Treat as bonus signals, not requirements:

| Signal | Cases | Notes |
|--------|-------|-------|
| Multiple synchronized wallets | 4/17 (24%) | Cluster-only pattern |
| Known political ENS domain | 1/17 (6%) | WLFI case only |
| Public denial | 1/17 (6%) | bigwinner01 only |
| Account went dormant after | 3/17 (18%) | Most stayed active |

---

## Variance-Adjusted Scoring

### Original Scoring (Overfit Risk)

```python
# BAD: Too rigid thresholds
if account_age_days < 7:
    score += 15
elif account_age_days < 30:
    score += 10
# Account at day 8 gets 10 points, day 6 gets 15 - cliff effect
```

### Variance-Adjusted Scoring (Better)

```python
# GOOD: Smooth gradients with soft thresholds
def account_age_score(days, base_threshold=7, variance=3):
    """
    Score with soft threshold and variance band.
    - Peak score at very fresh accounts
    - Gradual decline rather than hard cutoff
    - variance parameter widens the transition zone
    """
    if days < 1:
        return 15  # Maximum freshness
    elif days < base_threshold:
        # Linear decay from 15 to 10
        return 15 - (5 * (days / base_threshold))
    elif days < base_threshold + variance:
        # Transition zone
        return 10 - (5 * ((days - base_threshold) / variance))
    elif days < 30:
        return 5  # Still noteworthy
    else:
        return 0  # Not a signal
```

### Composite Variance Strategy

```python
def calculate_variance_adjusted_score(wallet, market=None):
    """
    Apply variance at multiple levels:
    1. Individual signal thresholds have soft edges
    2. Minimum signal count required (not all signals needed)
    3. Dimension caps prevent single-dimension dominance
    4. Confidence intervals on final score
    """

    signals = collect_all_signals(wallet, market)
    scores = {}

    # Score each dimension with variance
    scores['account'] = score_account_signals(signals, variance=1.3)
    scores['trading'] = score_trading_signals(signals, variance=1.2)
    scores['behavioral'] = score_behavioral_signals(signals, variance=1.4)
    scores['contextual'] = score_contextual_signals(signals, variance=1.2)
    scores['cluster'] = score_cluster_signals(signals, variance=1.1)

    # Dimension caps (prevent single dimension from dominating)
    capped_scores = {
        'account': min(scores['account'], 25),
        'trading': min(scores['trading'], 35),
        'behavioral': min(scores['behavioral'], 25),
        'contextual': min(scores['contextual'], 20),
        'cluster': min(scores['cluster'], 20)
    }

    raw_score = sum(capped_scores.values())

    # Normalize to 0-100
    # Max possible: 125 (if all dimensions maxed with cluster)
    # Base max (no cluster): 105
    normalized = (raw_score / 105) * 100

    # Calculate confidence interval based on signal count
    signal_count = count_triggered_signals(signals)
    confidence_width = calculate_confidence_interval(signal_count)

    return {
        'score': min(normalized, 100),
        'confidence_low': max(normalized - confidence_width, 0),
        'confidence_high': min(normalized + confidence_width, 100),
        'signal_count': signal_count,
        'dimensions': capped_scores
    }
```

---

## Minimum Signal Count Requirements

**Don't flag based on single strong signal. Require correlation.**

### Tier Thresholds

| Priority | Min Score | Min Signals | Signal Distribution |
|----------|-----------|-------------|---------------------|
| CRITICAL | 85+ | 5+ signals | 3+ dimensions |
| HIGH | 70-84 | 4+ signals | 2+ dimensions |
| MEDIUM | 55-69 | 3+ signals | 2+ dimensions |
| LOW | 40-54 | 2+ signals | Any |

### Why This Prevents Overfitting

**Example: Single Strong Signal**
- Wallet has $100K position (max trading signal)
- But: 2-year-old account, 50+ prior trades, diverse markets
- Single signal alone should NOT flag as insider
- Require: Position size + at least ONE account/behavioral signal

```python
def validate_flag_criteria(result):
    """
    Ensure we're not flagging on single dimension.
    """
    if result['score'] >= 70:  # HIGH or CRITICAL
        # Must have signals in multiple dimensions
        active_dimensions = sum(
            1 for d, s in result['dimensions'].items()
            if s > 0
        )
        if active_dimensions < 2:
            result['downgraded'] = True
            result['score'] = min(result['score'], 69)
            result['reason'] = 'Single-dimension flag, downgraded to MEDIUM'

    return result
```

---

## False Positive Mitigations

### Known False Positive Patterns

From documented cases:

1. **mutualdelta**: Perfect insider pattern → Lost $40K
   - Lesson: BET OUTCOME matters. Losses dramatically reduce confidence.

2. **Théo Cluster**: Multi-wallet coordination → NOT insider trading
   - Lesson: Public analysts can have insider-like patterns.

3. **GayPride**: Momentum riding → Secondary/copying behavior
   - Lesson: Late entries at high odds may be followers, not sources.

### False Positive Reduction Rules

```python
FALSE_POSITIVE_CHECKS = {
    'bet_lost': {
        'modifier': -40,
        'reason': 'Position resolved as loss'
    },
    'long_diverse_history': {
        'condition': 'trades > 100 and unique_categories > 5',
        'modifier': -20,
        'reason': 'Established diverse trader'
    },
    'public_analyst': {
        'condition': 'has_public_track_record or verified_identity',
        'modifier': -25,
        'reason': 'Known public figure with analysis history'
    },
    'high_odds_entry': {
        'condition': 'entry_odds > 0.60',
        'modifier': -15,
        'reason': 'Entered after odds already favorable (follower pattern)'
    },
    'split_positions': {
        'condition': 'has_both_yes_and_no_positions',
        'modifier': -10,
        'reason': 'Hedging behavior (insiders don\'t hedge)'
    }
}
```

---

## Position Size Comparative Analysis

### Observed Ranges (For Context)

| Metric | Minimum | 25th Percentile | Median | 75th Percentile | Maximum |
|--------|---------|-----------------|--------|-----------------|---------|
| Single Entry | $1,167 | $8,000 | $25,000 | $68,000 | $340M |
| Cumulative Total | $2,000 | $17,000 | $40,000 | $155,000 | $85M |
| Entry Count | 1 | 1 | 2 | 5 | 50+ |
| % of Capital | 50% | 80% | 95% | 100% | 100% |

### Scaling by Market Liquidity

```python
def liquidity_adjusted_position_score(position_size, market):
    """
    $50K on a $100K liquidity market = whale
    $50K on a $10M liquidity market = small fish
    """
    market_volume = market.total_volume
    market_liquidity = market.current_liquidity

    # Position as % of market
    volume_ratio = position_size / market_volume if market_volume > 0 else 1
    liquidity_ratio = position_size / market_liquidity if market_liquidity > 0 else 1

    if liquidity_ratio > 0.10:  # >10% of liquidity
        return 12  # Market mover
    elif liquidity_ratio > 0.05:
        return 10
    elif liquidity_ratio > 0.02:
        return 7
    elif liquidity_ratio > 0.01:
        return 4
    else:
        return 0  # Small relative to market
```

---

## Final Recommendations

### 1. Use Soft Thresholds Everywhere
No hard cutoffs. Use gradients and transition zones.

### 2. Require Signal Correlation
Single strong signals should elevate monitoring, not trigger alerts.

### 3. Weight By Observation Frequency
Signals seen in 90%+ of cases: Full weight
Signals seen in 50-70%: 70% weight
Signals seen in < 50%: 50% weight (supporting only)

### 4. Always Calculate Confidence Intervals
Report `score: 75 (CI: 68-82)` rather than just `score: 75`

### 5. Track False Positive Rate
If > 5% of flagged accounts are false positives, widen thresholds.

### 6. Cumulative Position Is King
Individual entry size matters less than total exposure.
A trader with 10x $5K entries = same as 1x $50K entry.

---

## Implementation Checklist

- [ ] Replace hard thresholds with soft gradient functions
- [ ] Add variance parameters to all scoring functions
- [ ] Implement minimum signal count validation
- [ ] Add confidence interval calculation
- [ ] Implement false positive reduction rules
- [ ] Add liquidity-adjusted position scoring
- [ ] Implement cumulative position tracking
- [ ] Test against known cases with variance applied
- [ ] Validate false positive rate < 5%
