# Polymarket Arbitrage Bot Strategy Extraction Report

**Date**: 2026-01-26
**Wallets Analyzed**: 3
**Total Trades Analyzed**: 75,000+

---

## Executive Summary

Three distinct strategies were identified across the analyzed wallets:

| Wallet | Strategy | Profitability | Key Characteristics |
|--------|----------|---------------|---------------------|
| 0x93c22116 | **PAIR_ACCUMULATION** | ~51% profitable markets | Buy cheap sides (<30¢), avg pair $0.967 |
| 0x6031b6ee | **MARKET_MAKING** | Break-even (~$1.00) | 70% balanced, 8s timing, centered at 48¢ |
| 0xe00740bc | **LATENCY_ARBITRAGE** | 21% profitable markets | 41% buys >70¢, directional betting |

**Recommended Strategy for Replication**: PAIR_ACCUMULATION (0x93c22116)

---

## Strategy 1: PAIR_ACCUMULATION (Primary Target)

### Wallet: 0x93c22116e4402c9332ee6db578050e688934c072

### Core Philosophy
> "Buy whichever side is cheap at any moment. Accumulate until pair cost is profitable."

### Key Metrics

| Metric | Value |
|--------|-------|
| Total Crypto 15m Trades | 25,000 |
| Markets Traded | 61 |
| Markets with Both Sides | 86.9% |
| Profitable Markets (<$0.98) | 50.9% |
| Avg Pair Cost | $0.967 |
| Min Pair Cost Achieved | $0.47 |

### Entry Price Thresholds

```
CRITICAL FINDING: 44% of buys are under $0.30

Entry Price Distribution:
  < $0.10: 7.2%   ← Extreme discount buys
  < $0.20: 19.3%  ← Heavy accumulation zone
  < $0.30: 44.3%  ← PRIMARY ENTRY ZONE
  < $0.40: 58.1%
  < $0.50: 64.3%
  $0.40-$0.60: 20.8%
  > $0.70: 11.9%  ← Occasional high-confidence buys
```

### Inferred Entry Rules

```python
# Primary entry threshold
CHEAP_SIDE_THRESHOLD = 0.30  # Buy when price < 30¢

# Secondary entry threshold
MODERATE_THRESHOLD = 0.40    # Buy when price < 40¢ if pair cost still good

# Maximum entry price
MAX_ENTRY_PRICE = 0.70       # Rarely buy above this

# Pair cost validation
def should_buy(current_price, current_pair_cost):
    if current_price < 0.30:
        return True  # Always buy cheap sides
    if current_price < 0.40 and current_pair_cost < 0.95:
        return True  # Buy moderate if pair still profitable
    return False
```

### Timing Patterns

```
Time Between First Up/Down Trades:
  Average: 294 seconds (4.9 minutes)
  Median:  86 seconds (1.4 minutes)
  Range:   4 seconds to 43 minutes

Total Accumulation Window:
  Average: 27.3 minutes
  Median:  13.0 minutes

INSIGHT: Not simultaneous execution - opportunistic over time
```

### Trade Sequence Example (Solana 7:00PM-7:15PM)

```
19:00:39  BUY Up   @ $0.40  (initial position)
19:00:53  BUY Up   @ $0.37  (price improved)
19:01:27  BUY Down @ $0.57  (48 sec later, start other side)
19:02:05  BUY Down @ $0.51  (price improving)
19:02:15  BUY Down @ $0.46  (aggressive accumulation)
...
19:05:55  BUY Up   @ $0.22  (price crashed - heavy buy)
19:06:29  BUY Up   @ $0.13  (extreme discount)
19:07:09  BUY Up   @ $0.11  (maximum opportunity)

Result: Pair cost $0.697, hedge ratio 95.9%
```

### Position Sizing

```
Trade Size Distribution:
  Avg:    $18.98
  Median: $3.64   ← Many small trades
  Max:    $688

Pattern: Many small opportunistic trades, occasional larger positions
```

### Outcome Bias

```
Up Buys:   15,426 trades ($233,564)
Down Buys:  9,574 trades ($241,000)

INSIGHT: Slightly more Up trades but similar USD exposure
         Market direction doesn't matter - both sides accumulated
```

---

## Strategy 2: MARKET_MAKING (Secondary Pattern)

### Wallet: 0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d

### Key Characteristics

| Metric | Value |
|--------|-------|
| Both Sides | 100% of markets |
| Balanced (>90% hedge) | 70% |
| Avg Pair Cost | $1.003 |
| Avg Entry Price | $0.48 |

### Inferred Strategy

```
- Provides liquidity near 50/50 odds
- Rapid execution (8 second median gap between sides)
- Highly balanced positions (not seeking arbitrage profit)
- Likely earning from spread + rebates, not pair cost
```

### Entry Pattern

```
Buy Price Distribution:
  $0.40-$0.60: 71.6%  ← Concentrated near fair value
```

---

## Strategy 3: LATENCY_ARBITRAGE (Directional)

### Wallet: 0xe00740bce98a594e26861838885ab310ec3b548c

### Key Characteristics

| Metric | Value |
|--------|-------|
| Avg Pair Cost | $1.059 (unprofitable as pure arb) |
| Buys > 70¢ | 41.4% |
| Balanced Markets | 5% |

### Inferred Strategy

```
- Monitors CEX (Binance) for directional signals
- Buys high-probability side before Polymarket catches up
- NOT pair accumulation - directional betting
- Only 21% of markets profitable by pair cost
- Must be winning on directional bets to be profitable
```

---

## Recommended Bot Parameters (PAIR_ACCUMULATION)

### Entry Rules

```python
class ArbitrageConfig:
    # Entry thresholds
    AGGRESSIVE_BUY_THRESHOLD = 0.25   # Always buy
    STANDARD_BUY_THRESHOLD = 0.35     # Buy if pair cost OK
    MAX_ENTRY_PRICE = 0.50            # Never buy above

    # Pair cost targets
    TARGET_PAIR_COST = 0.95           # Ideal target
    MAX_PAIR_COST = 0.98              # Absolute maximum

    # Position management
    MIN_HEDGE_RATIO = 0.70            # Accept imbalance up to 30%
    MAX_POSITION_PER_MARKET = 1000    # USD

    # Timing
    CHECK_INTERVAL_MS = 500           # Price check frequency
    MAX_ACCUMULATION_MINUTES = 12     # Stop accumulating after

    # Risk
    MAX_CONCURRENT_MARKETS = 4        # Limit exposure
```

### Entry Logic Pseudocode

```python
def evaluate_opportunity(market, current_holdings):
    up_price = market.get_best_ask("Up")
    down_price = market.get_best_ask("Down")

    # Calculate current pair cost if we bought now
    up_avg = holdings.up_avg_price if holdings.up_shares > 0 else up_price
    down_avg = holdings.down_avg_price if holdings.down_shares > 0 else down_price

    # Check Up side
    if up_price < AGGRESSIVE_BUY_THRESHOLD:
        buy("Up", calculate_size(up_price))
    elif up_price < STANDARD_BUY_THRESHOLD:
        projected_pair = up_avg + down_avg
        if projected_pair < TARGET_PAIR_COST:
            buy("Up", calculate_size(up_price))

    # Check Down side (same logic)
    if down_price < AGGRESSIVE_BUY_THRESHOLD:
        buy("Down", calculate_size(down_price))
    elif down_price < STANDARD_BUY_THRESHOLD:
        projected_pair = up_avg + down_avg
        if projected_pair < TARGET_PAIR_COST:
            buy("Down", calculate_size(down_price))

def calculate_size(price):
    # Smaller sizes at higher prices
    if price < 0.15:
        return MAX_TRADE_SIZE
    elif price < 0.25:
        return MAX_TRADE_SIZE * 0.7
    elif price < 0.35:
        return MAX_TRADE_SIZE * 0.4
    else:
        return MAX_TRADE_SIZE * 0.2
```

### Position Management

```python
def should_continue_accumulating(holdings, market):
    # Stop if pair cost exceeded
    if holdings.pair_cost > MAX_PAIR_COST:
        return False

    # Stop if max position reached
    if holdings.total_usd > MAX_POSITION_PER_MARKET:
        return False

    # Stop if accumulation time exceeded
    if holdings.accumulation_time > MAX_ACCUMULATION_MINUTES * 60:
        return False

    # Stop if market resolution imminent
    if market.time_to_resolution < 60:  # 1 minute
        return False

    return True
```

---

## Risk Considerations

### Fee Impact

```
Current fee structure:
- 2% winner fee (deducted from winning side)
- Dynamic taker fees (up to 3.15% at 50% odds)

Profitability threshold:
- At 50% odds: pair cost must be < $0.95 for profit
- At 30%/70% odds: pair cost must be < $0.97 for profit
```

### Execution Risks

```
1. Non-atomic execution - legs can fail independently
2. Price slippage during accumulation window
3. Market resolution timing uncertainty
4. Liquidity constraints for large positions
```

### Mitigation Strategies

```
1. Use Fill-or-Kill (FOK) orders
2. Small position sizes (median $3.64)
3. Multiple trades rather than large single trades
4. Stop accumulating before resolution
```

---

## Next Steps

1. **Implement monitoring**: Real-time price feeds for BTC/ETH/SOL/XRP 15m markets
2. **Build entry detection**: Identify when prices drop below thresholds
3. **Create position tracker**: Track accumulated positions per market
4. **Add execution layer**: FOK orders via CLOB API
5. **Backtest**: Validate parameters against historical data
6. **Paper trade**: Test in live market without real capital

---

## Data Files Generated

```
arbitrage_analysis/data/
├── 0x93c22116_analysis.json   # Full market-by-market breakdown
├── 0x93c22116_raw_trades.json # 25,000 individual trades
├── 0x6031b6ee_analysis.json
├── 0x6031b6ee_raw_trades.json
├── 0xe00740bc_analysis.json
└── 0xe00740bc_raw_trades.json
```
