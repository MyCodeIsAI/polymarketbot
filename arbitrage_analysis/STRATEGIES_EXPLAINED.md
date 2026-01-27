# Polymarket 15-Minute Market Strategies Explained

## Overview

Three distinct strategies were identified from analyzing 75,000+ trades across 3 profitable wallets. Each works differently and has different risk/reward profiles.

---

## Strategy 1: PAIR_ACCUMULATION (Recommended)

**Wallet**: 0x93c22116
**Profitability**: 51% of markets profitable
**Our Implementation**: ✅ YES - This is what our bot uses

### How It Works (Plain English)

1. **Wait for a cheap side**: When one side of a market (Up or Down) drops below $0.30, buy it. This happens when the market temporarily misprices one outcome.

2. **Keep buying cheap**: If the price stays cheap or gets cheaper, keep accumulating that side.

3. **Complete the pair**: Once you own one side, watch for the OTHER side to become affordable. Buy the other side even if it's more expensive ($0.60-$0.70) as long as your total pair cost stays under $0.97.

4. **Profit at resolution**: When the market resolves, you get $1.00 payout (one side wins). If your pair cost was $0.95, you profit $0.05 per pair regardless of which side wins.

### Example Trade Sequence

```
Time        Action              Price    Running Pair Cost
19:00:39    BUY Up              $0.40    $0.40 (need Down)
19:00:53    BUY Up              $0.37    $0.38 avg (still need Down)
19:01:27    BUY Down            $0.57    $0.95 pair cost ← NOW HEDGED
19:02:05    BUY Down            $0.51    $0.89 pair cost (improving!)
19:05:55    BUY Up              $0.22    $0.75 pair cost (price crashed)
19:06:29    BUY Up              $0.13    $0.70 pair cost (great entry)

RESULT: Pair cost $0.70, guaranteed $0.30 profit per share
```

### Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| AGGRESSIVE_BUY | < $0.30 | Always buy - great price |
| STANDARD_BUY | < $0.40 | Buy if pair cost still good |
| PAIR_COMPLETE | < $0.75 | Max price to complete a pair |
| TARGET_PAIR_COST | $0.97 | Aim for this pair cost |
| MAX_PAIR_COST | $0.98 | Never exceed this |

### Risk Profile

- **Hedged**: ~87% of markets end up with both sides
- **Unhedged Risk**: ~13% may not complete pairs before resolution
- **Profit Source**: Pair cost below $1.00

---

## Strategy 2: MARKET_MAKING

**Wallet**: 0x6031b6ee
**Profitability**: Break-even on pair cost (~$1.00)
**Our Implementation**: ❌ NO - Requires market maker rebates

### How It Works (Plain English)

1. **Buy both sides immediately**: When entering a market, buy BOTH Up and Down within seconds of each other.

2. **Stay balanced**: Keep equal amounts of Up and Down shares (98%+ hedge ratio).

3. **Trade near 50/50 odds**: Most buys are around $0.48-$0.52 (near fair value).

4. **Profit from rebates**: Since pair cost is ~$1.00, there's no arbitrage profit. Instead, they earn maker rebates from the exchange for providing liquidity.

### Example

```
Buy Up   @ $0.48 → $0.48
Buy Down @ $0.52 → $1.00 pair cost (break-even)

Profit comes from exchange rebates, NOT from pair cost.
```

### Why We Don't Use This

- Requires market maker status for rebates
- No profit from pair cost itself
- Need high volume to make rebates worthwhile
- More capital intensive

---

## Strategy 3: LATENCY_ARBITRAGE (Directional)

**Wallet**: 0xe00740bc
**Profitability**: Only 21% profitable by pair cost
**Our Implementation**: ❌ NO - Requires CEX price feeds

### How It Works (Plain English)

1. **Monitor centralized exchanges**: Watch Binance/Coinbase for BTC/ETH/SOL price movements.

2. **Bet on direction**: When crypto price moves on CEX, quickly buy the winning side on Polymarket before odds adjust.

3. **Accept unbalanced positions**: 41% of buys are at expensive prices (>$0.70) because they're confident in direction.

4. **Profit from being faster**: Make money by predicting market resolution, not from pair arbitrage.

### Example

```
CEX: Bitcoin pumps 0.5% in 30 seconds
Action: Quickly buy "Up" on Polymarket at $0.65 before price adjusts
Result: If Up wins, profit. If Down wins, loss.

This is DIRECTIONAL BETTING, not arbitrage.
```

### Why We Don't Use This

- Requires real-time CEX price feeds
- Requires sub-second latency
- High risk - losses if direction is wrong
- Avg pair cost $1.06 means LOSING money on pairs

---

## Strategy Comparison

| Aspect | PAIR_ACCUMULATION | MARKET_MAKING | LATENCY_ARB |
|--------|-------------------|---------------|-------------|
| Pair Cost | $0.967 (profit) | $1.00 (break-even) | $1.06 (loss) |
| Both Sides | 87% | 100% | 95% |
| Entry Prices | Cheap (<$0.40) | Fair (~$0.48) | Often expensive (>$0.70) |
| Profit Source | Pair arbitrage | Maker rebates | Directional bets |
| Risk | Moderate | Low | High |
| Requirements | Price monitoring | Maker status | CEX feeds + speed |
| **Our Choice** | ✅ YES | ❌ NO | ❌ NO |

---

## Why We Chose PAIR_ACCUMULATION

1. **Proven profitable**: 51% of markets under $0.98 pair cost
2. **No special access needed**: Just monitor Polymarket prices
3. **Hedged most of the time**: 87% of positions have both sides
4. **Clear edge**: Buy cheap, complete pairs, profit on resolution
5. **Replicable**: We can copy exactly what they do

---

## Current Bot Configuration

```python
# Exactly matching profitable account 0x93c22116
AGGRESSIVE_BUY_THRESHOLD = 0.30   # 44% of their buys were here
STANDARD_BUY_THRESHOLD = 0.40     # 58% cumulative
MAX_COMPLETION_PRICE = 0.75       # They bought at $0.60-0.70
TARGET_PAIR_COST = 0.97           # Their avg was $0.967
MAX_PAIR_COST = 0.98              # 51% of markets were under this
DYNAMIC_THRESHOLDS = False        # Fixed thresholds like they used
```
