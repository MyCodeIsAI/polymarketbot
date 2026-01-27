# Polymarket 15-Minute Crypto Arbitrage: Research & Reverse Engineering Plan

**Date**: 2026-01-26
**Purpose**: Reverse engineer profitable arbitrage bot strategies on Polymarket's 15-minute crypto markets

---

## Executive Summary

Based on comprehensive research including the IMDEA Networks academic paper (arXiv:2508.03474), three distinct arbitrage mechanisms exist on Polymarket:

1. **Single-Condition Market Rebalancing**: Exploits YES+NO price deviations from $1.00
2. **Multiple-Condition Market Rebalancing**: Exploits pricing across all outcomes in multi-outcome markets
3. **Combinatorial Arbitrage**: Exploits logical dependencies between related markets

**Scale**: $40 million in arbitrage profits extracted April 2024 - April 2025 across 86 million bets.

**Top Performers**:
- Leading account: $2,009,632 profit across 4,049 transactions
- 2nd place: $1,273,059 across 2,215 transactions
- Top 3 wallets combined: $4.2 million

---

## Part 1: Academic Foundation (arXiv:2508.03474)

### Authors & Affiliations
- Oriol Saguillo, Vahid Ghafouri, Lucianna Kiffer, Guillermo Suarez-Tangil
- IMDEA Networks Institute, Oxford Internet Institute
- Presented at AFT 2025 (Advances in Financial Technologies)
- Funded by Flashbots Research Proposal FRP-51

### Core Findings

#### Mathematical Formulas for Arbitrage

**Single-Condition Arbitrage (Binary Markets)**:
```
Theoretical: Price(YES) + Price(NO) = 1.00

Long Arbitrage (buy both):
  Condition: ∑ᵢ val(Yᵢ,t) < 1
  Profit: 1 - ∑ val(Yᵢ,t)

Short Arbitrage (sell both):
  Condition: ∑ᵢ val(Yᵢ,t) > 1
  Profit: ∑ val(Yᵢ,t) - 1
```

**Multiple-Condition Arbitrage (N outcomes)**:
```
Theoretical constraints:
  - Sum of all YES prices = 1.00
  - Sum of all NO prices = N - 1

Example (3-candidate race):
  Candidate A YES: $0.35
  Candidate B YES: $0.32
  Candidate C YES: $0.30
  Total: $0.97 (should be $1.00)
  Profit per set: $0.03 (3%)
```

**Combinatorial Arbitrage (Cross-Market)**:
```
When markets M1 and M2 depend on identical underlying events:

Condition: ∑ c∈S val(Tc,t) < ∑ c'∈S' val(Tc',t)

Example:
  M1: "Democrats win" @ 0.48
  M2: All Republican margin outcomes total @ 0.40
  Combined cost: $0.88
  Guaranteed payout: $1.00
  Profit: 13.63%
```

### Profit Breakdown by Type

| Arbitrage Type | Long Profits | Short Profits | Subtotal |
|----------------|--------------|---------------|----------|
| Single-Condition | $5,899,287 | $4,682,075 | **$10.58M** |
| Multiple-Condition YES | $11,092,286 | $612,189 | $11.70M |
| Multiple-Condition NO | $17,307,114 | $4,264 | $17.31M |
| **Multiple-Condition Total** | | | **$29.02M** |
| Combinatorial | | | **$95,157** |
| **GRAND TOTAL** | | | **$40M+** |

### Key Statistics

- **Analysis period**: April 1, 2024 to April 1, 2025
- **Bets analyzed**: 86 million
- **Trade grouping window**: 950 blocks (~1 hour) for related trades
- **LLM accuracy for market inference**: 81.45%
- **Median profit margin**: ~60 cents per dollar on opportunities
- **Opportunity duration**: Seconds to minutes typically
- **Price staleness**: Up to 2.5 hours in sparse markets

---

## Part 2: Execution Mechanisms

### CTF (Conditional Token Framework) Operations

**Split Operation** - Minting outcome tokens:
```
1 USDC → 1 YES token + 1 NO token

Use case: Create full position set for arbitrage
```

**Merge Operation** - Burning outcome tokens:
```
1 YES token + 1 NO token → 1 USDC

Use case: Exit arbitrage position, collect guaranteed $1
```

**Redemption** - After market resolution:
```
Winning token (YES or NO) → $1.00
Losing token → $0.00
```

### Order Types and Matching

**Exchange Matching Modes**:
1. **Direct orders**: Standard limit order matching
2. **Complementary crossing**: YES buy matched with NO buy, mint new tokens
3. **Split execution**: Collateral split into both positions
4. **Merge execution**: Positions merged back to collateral

**Fill-or-Kill (FOK) Orders** - Critical for arbitrage:
```
Purpose: Atomic execution of entire order or nothing
Prevents: "Legged" positions with unhedged exposure
Requirement: Both legs execute or neither does
```

### Non-Atomic Execution Risk

Polymarket arbitrage is **NOT atomic** at the order level:
- One leg can succeed while other fails
- Creates unhedged directional exposure
- Execution delays of milliseconds can eliminate profit
- Top traders use 100ms or faster execution

---

## Part 3: 15-Minute Crypto Markets Specifics

### Market Structure
- **Assets**: BTC, ETH, SOL, XRP
- **Duration**: 15-minute windows, continuous
- **Resolution**: Chainlink oracle (Binance USDT ticker)
- **Win Condition**: Green candle = YES wins, Red candle = NO wins

### Fee Structure (Critical Change - Late 2025)

**Previous**: Zero taker fees
**Current**: Dynamic taker fees based on probability

| Probability | Approximate Fee |
|-------------|-----------------|
| 50% | ~3.15% |
| 30% or 70% | ~2.5% |
| 10% or 90% | ~1.5% |
| <5% or >95% | <1% |

**Impact**: Latency arbitrage at 50/50 odds now requires >3.15% edge to profit.

**Winner Fee**: 2% deducted from winning payouts (unchanged)

### Profitability Threshold Calculation

```python
# Minimum spread needed for profit
min_spread_pct = (winner_fee + taker_fee) / (1 - winner_fee)

# At 50% odds:
min_spread = (0.02 + 0.0315) / 0.98 = 5.26%

# Pair cost must be:
max_pair_cost = 1.00 / (1 + min_spread_pct) = $0.95

# At extreme odds (5%/95%):
min_spread = (0.02 + 0.015) / 0.98 = 3.57%
max_pair_cost = $0.965
```

---

## Part 4: Strategy #1 - Asymmetric Pair Accumulation (Gabagool)

### Core Mechanism

**Philosophy**: Never predict direction. Wait for temporary mispricings on each side.

**Execution**:
1. Monitor YES and NO prices independently
2. Buy YES when it becomes "cheap" (below threshold)
3. Buy NO when it becomes "cheap" (below threshold)
4. Accumulate until `avg_yes_price + avg_no_price < threshold`
5. Hold to resolution for guaranteed profit

### Entry Logic

```python
# Individual side buy trigger
def should_buy_side(current_price: float, threshold: float) -> bool:
    return current_price < threshold

# Typical thresholds (to be validated from account data):
YES_BUY_THRESHOLD = 0.45  # Buy YES if price < $0.45
NO_BUY_THRESHOLD = 0.45   # Buy NO if price < $0.45

# Combined pair validation
def is_profitable_pair(avg_yes: float, avg_no: float) -> bool:
    pair_cost = avg_yes + avg_no
    min_profit_margin = 0.03  # 3% after fees
    return pair_cost < (1.00 - min_profit_margin)
```

### Position Balancing

```python
# Goal: qty_yes ≈ qty_no
def calculate_hedge_ratio(yes_shares: float, no_shares: float) -> float:
    return min(yes_shares, no_shares) / max(yes_shares, no_shares)

# Target hedge ratio > 0.90 for minimal directional exposure
# Imbalance = abs(yes_shares - no_shares)
```

### Real Example (Gabagool Bot)

| Side | Shares | Avg Price | Total Cost |
|------|--------|-----------|------------|
| YES | 1,266.72 | $0.517 | $654.89 |
| NO | 1,294.98 | $0.449 | $581.85 |
| **Total** | | | **$1,236.74** |

- Pair cost: $0.966 ($0.517 + $0.449)
- Guaranteed payout: ~$1,266 (limited by smaller position)
- Profit: $58.52 (4.7% gross, ~2.7% net after 2% fee)

---

## Part 5: Strategy #2 - Latency Arbitrage

### Core Mechanism

**Philosophy**: Exploit price lag between CEX (Binance/Coinbase) and Polymarket.

**Execution**:
1. Monitor real-time CEX prices for BTC/ETH/SOL
2. Detect strong directional momentum
3. Compare to Polymarket odds (often lagging 1-2 minutes)
4. Enter when "true" probability diverges from market odds
5. Exit before resolution or hold to settlement

### Example

```
CEX: BTC pumps 2% in 30 seconds
CEX reality: 85% chance of green 15-min candle

Polymarket: Still showing 52% YES / 48% NO
True value: YES should be ~$0.85

Action: Buy YES at $0.52
Outcome: Market catches up, YES goes to $0.80+
Profit: ~50% if sold, or hold to resolution
```

### Post-Fee Viability

With 3.15% taker fee at 50% odds:
- Need odds to be mispriced by >3.5% to profit
- Latency must be sub-100ms to capture window
- Strategy significantly degraded since fee introduction

---

## Part 6: Data Extraction Requirements

### Essential Fields Per Trade

| Field | Purpose | Source |
|-------|---------|--------|
| `timestamp` | Timing analysis | Activity API |
| `condition_id` | Market identification | Activity API |
| `token_id` | YES/NO identification | Activity API |
| `outcome` | "Yes" or "No" | Activity API |
| `side` | BUY or SELL | Activity API |
| `price` | Entry odds | Activity API |
| `size` | Share quantity | Calculated |
| `usd_value` | Dollar amount | Activity API |
| `market_title` | Market filtering | Gamma API (enrichment) |
| `event_slug` | Category detection | Activity API |
| `tx_hash` | Blockchain verification | Activity API |

### Derived Metrics

**Per-Market**:
```python
for condition_id in markets:
    yes_buys = filter(trades, outcome="Yes", side="BUY")
    no_buys = filter(trades, outcome="No", side="BUY")

    total_yes_shares = sum(t.shares for t in yes_buys)
    total_no_shares = sum(t.shares for t in no_buys)

    avg_yes_price = sum(t.usd for t in yes_buys) / total_yes_shares
    avg_no_price = sum(t.usd for t in no_buys) / total_no_shares

    pair_cost = avg_yes_price + avg_no_price
    hedge_ratio = min(yes, no) / max(yes, no)

    time_span = max(timestamps) - min(timestamps)
    time_between_first_trades = abs(yes_buys[0].ts - no_buys[0].ts)
```

**Aggregate**:
```python
# Entry price distribution
pct_under_30c = count(price < 0.30) / total
pct_under_40c = count(price < 0.40) / total
pct_under_50c = count(price < 0.50) / total

# Pair cost distribution
avg_pair_cost = mean(pair_costs)
median_pair_cost = median(pair_costs)

# Strategy classification
pct_both_sides = count(has_yes_and_no) / total_markets
pct_balanced = count(hedge_ratio > 0.90) / total_markets
```

---

## Part 7: API Endpoints

### Data API (`https://data-api.polymarket.com`)

```python
# Fetch trade activity
GET /activity
Params:
  - user: wallet address
  - type: TRADE, SPLIT, MERGE, REDEEM
  - limit: max 500
  - offset: for pagination
  - sortBy: TIMESTAMP
  - sortDirection: DESC
```

### Gamma API (`https://gamma-api.polymarket.com`)

```python
# Market metadata enrichment
GET /markets
Params:
  - condition_ids: comma-separated list
  - active: true/false
```

### CLOB API (`https://clob.polymarket.com`)

```python
# Real-time order book
GET /book
Params:
  - token_id: specific outcome token

# Current market prices
GET /markets/{condition_id}
```

### WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/`)

```python
# Real-time price updates
Channel: market
Payload: orderbook updates, price changes
```

---

## Part 8: Extraction Script Architecture

### Existing Infrastructure (polymarketbot/)

```
src/api/
  ├── data.py      # DataAPIClient - activity, positions, trades
  ├── gamma.py     # GammaAPIClient - market metadata
  ├── clob.py      # CLOBClient - real-time trading
  └── base.py      # BaseAPIClient with rate limiting
```

### New Arbitrage Analysis Module

```
arbitrage_analysis/
  ├── RESEARCH_AND_PLAN.md         # This document
  ├── extract_arbitrage_trades.py  # Trade extraction
  ├── analyze_strategy.py          # Strategy analysis
  ├── compare_wallets.py           # Cross-wallet patterns
  └── data/
      ├── {wallet}_raw_trades.json
      └── {wallet}_analysis.json
```

### Extraction Flow

```
1. INPUT: Wallet addresses of known arbitrage bots

2. FOR EACH wallet:
   a. Paginate through /activity endpoint (500 per page)
   b. Filter to TRADE type
   c. Convert to TradeRecord objects
   d. Filter to 15-min crypto markets (by title/slug)
   e. Group by condition_id

3. FOR EACH market:
   a. Separate YES and NO trades
   b. Calculate side-level metrics (avg price, share count, timing)
   c. Calculate pair-level metrics (pair cost, hedge ratio)

4. AGGREGATE across markets:
   a. Entry price distribution
   b. Pair cost distribution
   c. Strategy type classification
   d. Timing patterns

5. OUTPUT: Structured JSON analysis
```

---

## Part 9: Questions to Answer from Analysis

### Strategy Mechanics
1. What individual price triggers side purchases? (e.g., YES < $0.40)
2. What pair cost threshold triggers accumulation? (e.g., < $0.97)
3. Do they buy both sides every market or selectively?
4. How much position imbalance do they tolerate?
5. Do they ever sell before resolution?

### Timing Analysis
1. How quickly do they accumulate? (seconds vs minutes)
2. Time gap between first YES and first NO purchase?
3. Do they front-run volatility events?
4. Total accumulation window relative to 15-min market?

### Position Sizing
1. Fixed or variable position sizes?
2. Maximum position per market?
3. Maximum total exposure across markets?
4. Position size correlation with pair cost?

### Performance Validation
1. Actual achieved pair costs?
2. Actual win rate?
3. Average profit per market?
4. Profit distribution (few big wins vs many small?)

---

## Part 10: Next Steps

1. **You provide wallet addresses** of known 15-min crypto arbitrage bots
2. **Run extraction script** on each wallet
3. **Analyze patterns** across wallets
4. **Extract strategy parameters**:
   - Entry price thresholds
   - Pair cost targets
   - Position sizing rules
   - Timing constraints
5. **Design replication bot** based on validated parameters
6. **Backtest** against historical 15-min crypto market data
7. **Paper trade** before live deployment

---

## Sources

### Academic Research
- [arXiv:2508.03474 - Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets](https://arxiv.org/abs/2508.03474) - Saguillo et al., IMDEA Networks, AFT 2025
- [Flashbots Research Discussion](https://collective.flashbots.net/t/arbitrage-in-prediction-markets-strategies-impact-and-open-questions/5198)
- [QuantPedia - Systematic Edges in Prediction Markets](https://quantpedia.com/systematic-edges-in-prediction-markets/)

### Technical Documentation
- [Polymarket Documentation](https://docs.polymarket.com/)
- [CTF Merge Operations](https://docs.polymarket.com/developers/CTF/merge)
- [GitHub: CTF Exchange](https://github.com/Polymarket/ctf-exchange)
- [GitHub: Conditional Token Examples](https://github.com/Polymarket/conditional-token-examples-py)

### Industry Coverage
- [AInvest - Algorithmic Arbitrage on Polymarket](https://www.ainvest.com/news/algorithmic-arbitrage-crypto-prediction-markets-exploiting-binary-mispricings-polymarket-2512/)
- [Finance Magnates - Dynamic Fees](https://www.financemagnates.com/cryptocurrency/polymarket-introduces-dynamic-fees-to-curb-latency-arbitrage-in-short-term-crypto-markets/)
- [CoinsBench - Inside the Mind of a Polymarket BOT](https://coinsbench.com/inside-the-mind-of-a-polymarket-bot-3184e9481f0a)
- [BeInCrypto - Arbitrage Bots Dominate](https://beincrypto.com/arbitrage-bots-polymarket-humans/)
- [The Block - 15-Minute Crypto Markets](https://www.theblock.co/post/384461/polymarket-adds-taker-fees-to-15-minute-crypto-markets-to-fund-liquidity-rebates)
