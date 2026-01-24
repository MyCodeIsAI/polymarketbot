# Comprehensive Signal Extraction: All Documented Cases

## Overview

This document extracts EVERY identifiable signal from all 17+ documented insider trading cases. These signals form the foundation of the detection system.

---

## Signal Categories

### Category 1: Account Characteristics (11 Signals)

| Signal | Source Cases | Weight | Detection Method |
|--------|--------------|--------|------------------|
| Account age < 24 hours | 6741, Fed Rate | 25 | `account.createdAt - now` |
| Account age < 7 days | Burdensome-Mix, dirtycup | 20 | `account.createdAt - now` |
| Account age < 30 days | Most cases | 10 | `account.createdAt - now` |
| Zero prior transactions | 6741, dirtycup | 25 | `account.txCount == 0` |
| < 5 prior transactions | Multiple | 15 | `account.txCount < 5` |
| First transaction is large bet | 6741, Fed Rate | 20 | `firstTx.type == 'bet' && firstTx.size > 5000` |
| Single-purpose wallet | All geopolitical | 20 | `uniqueMarkets == 1` |
| Username change after win | ricosuave666→Rundeep | 10 | `username.changed && recentWin` |
| Account dormancy pattern | ricosuave666 (7 months) | 15 | `daysSinceLastActivity > 90 && suddenActivity` |
| Account created specifically for event | 6741, Burdensome-Mix | 20 | `createdBefore(event, 48h) && onlyBetOnEvent` |
| ENS domains resemble known figures | WLFI connection | 15 | `fuzzyMatch(ensDomains, politicalFigures)` |

---

### Category 2: Trading Behavior (14 Signals)

| Signal | Source Cases | Weight | Detection Method |
|--------|--------------|--------|------------------|
| 100% win rate in category | ricosuave666, Maduro cohort | 30 | `categoryWinRate == 1.0 && trades >= 3` |
| 90%+ win rate in category | 0xafEe (95.6%) | 25 | `categoryWinRate >= 0.90` |
| 80%+ win rate overall | Annica (80%) | 15 | `overallWinRate >= 0.80` |
| Single market focus (100%) | Burdensome-Mix, 6741 | 25 | `marketConcentration == 1.0` |
| Category concentration > 90% | 0xafEe (Google), ricosuave666 (Israel) | 20 | `categoryConcentration >= 0.90` |
| Large bet on < 10% odds | All Nobel, Maduro cases | 20 | `betSize > 10000 && entryOdds < 0.10` |
| Large bet on < 5% odds | 6741 (4%), Maduro (8%) | 25 | `betSize > 10000 && entryOdds < 0.05` |
| Exact date prediction | ricosuave666 (June 24) | 30 | `predictedExactDate && resolved` |
| No hedging positions | All cases | 15 | `oppositePositions == 0` |
| Position size > 50% of total capital | Burdensome-Mix | 20 | `positionSize / totalCapital > 0.50` |
| Return multiple > 10x | 6741 (25x), Maduro (12x) | 20 | `profit / investment > 10` |
| Profit > $100K | Multiple | 15 | `profit > 100000` |
| Quick position exit after pump | ricosuave666 (Jan 2026) | 15 | `exitedWithin(48h) && causedOddsSpike` |
| Contrarian longshot wins | 0xafEe (d4vd at 0.2%) | 20 | `entryOdds < 0.05 && won` |

---

### Category 3: Timing Signals (9 Signals)

| Signal | Source Cases | Weight | Detection Method |
|--------|--------------|--------|------------------|
| Bet < 6 hours before event | Burdensome-Mix (3h), dirtycup | 25 | `eventTime - betTime < 6h` |
| Bet < 24 hours before event | Most cases | 20 | `eventTime - betTime < 24h` |
| Bet during private info window | Portugal (exit polls), Nobel | 25 | `betTime.overlaps(privateInfoWindow)` |
| Off-hours trading (0-6 AM UTC) | dirtycup (3:41 AM) | 15 | `betTime.hour in [0,1,2,3,4,5,6]` |
| Weekend/holiday trading | Some cases | 10 | `betTime.isWeekend || betTime.isHoliday` |
| 30-minute pre-announcement | bigwinner01 | 30 | `announcementTime - betTime < 30min` |
| Returns from dormancy before event | ricosuave666 | 20 | `dormantMonths > 3 && newBetBeforeEvent` |
| Coordinated timing (cluster) | Iran wallets, Portugal | 20 | `clusterBetTimeSpread < 60min` |
| News-trade correlation | All cases | 20 | `betTime.correlatesWith(newsBreak)` |

---

### Category 4: Funding & Withdrawal (8 Signals)

| Signal | Source Cases | Weight | Detection Method |
|--------|--------------|--------|------------------|
| Direct CEX deposit (Coinbase/Kraken) | Burdensome-Mix, Théo | 15 | `fundingSource.type == 'cex'` |
| Same funding source as flagged wallet | Maduro cohort | 30 | `fundingSource in FLAG_FILE` |
| Large deposit immediately before bet | 0xafEe ($3M), Fed Rate | 20 | `depositAmount > 50000 && depositTobet < 48h` |
| No privacy measures used | Burdensome-Mix | 10 | `!usedMixer && !usedVPN` |
| Immediate withdrawal after win | Multiple | 15 | `withdrawalTime - resolutionTime < 24h` |
| Withdrawal to same CEX as funding | Burdensome-Mix | 10 | `withdrawDest == fundingSource` |
| Memecoin purchase after win | WLFI connection (Fartcoin) | 5 | `postWinActivity.includes('memecoin')` |
| Cross-chain movement (Solana/ETH) | WLFI connection | 10 | `hasActivity(otherChains)` |

---

### Category 5: Cluster/Sybil Signals (7 Signals)

| Signal | Source Cases | Weight | Detection Method |
|--------|--------------|--------|------------------|
| Same funding source (multiple wallets) | Théo (11), Maduro (3) | 30 | `wallets.groupBy(fundingSource).size > 1` |
| Synchronized bet timing | Iran wallets, Portugal | 25 | `betTimeSpread < 60min` |
| > 90% market overlap | Théo cluster | 20 | `marketOverlap(walletA, walletB) > 0.90` |
| Account creation proximity | Maduro cohort | 15 | `creationTimeSpread < 7days` |
| Proportional position sizing | Théo cluster | 15 | `sizingRatio.variance < 0.20` |
| Same exchange origin | Théo (Kraken), Maduro (Coinbase) | 15 | `wallets.all(w => w.exchange == X)` |
| Coordinated exit | Some clusters | 15 | `exitTimeSpread < 2h` |

---

### Category 6: Market Context Signals (6 Signals)

| Signal | Source Cases | Weight | Detection Method |
|--------|--------------|--------|------------------|
| Military operation market | ricosuave666, Maduro | 30 | `market.category == 'military'` |
| Government policy market | bigwinner01, Fed Rate | 25 | `market.category == 'policy'` |
| Election market (final hours) | Portugal | 25 | `market.category == 'election' && hoursToClose < 6` |
| Awards/committee market | Nobel cases | 20 | `market.category == 'awards'` |
| Tech product launch market | 0xafEe, OpenAI | 15 | `market.category == 'tech_launch'` |
| Controllable outcome market | Annica (Musk tweets) | -10 | `market.outcomeControllable` |

---

### Category 7: Evasion Behaviors (5 Signals)

| Signal | Source Cases | Weight | Detection Method |
|--------|--------------|--------|------------------|
| Username changed after publicity | ricosuave666→Rundeep | 15 | `usernameChangedAfter(mediaAttention)` |
| Account went inactive after win | 0x31a5, 0xa72D | 10 | `inactiveDays > 7 && recentLargeWin` |
| Funds moved to new wallet | Various | 10 | `transferToNewWallet.afterWin` |
| Deleted social presence | Some cases | 5 | `socialLinks.removed` |
| Denied allegations publicly | bigwinner01 | 5 | `publicDenial && continuesTrading` |

---

## Master Signal Weights Table

### Tier 1: Critical Signals (20+ points)

| Signal | Max Points |
|--------|------------|
| Same funding source as flagged wallet | 30 |
| 100% category win rate | 30 |
| Exact date prediction | 30 |
| 30-minute pre-announcement timing | 30 |
| Military operation market | 30 |
| Synchronized cluster betting | 25 |
| Account age < 24 hours | 25 |
| Zero prior transactions | 25 |
| Single market focus | 25 |
| Bet < 6 hours before event | 25 |
| Large bet on < 5% odds | 25 |
| Government policy market | 25 |
| Election final hours | 25 |
| Private info window correlation | 25 |

### Tier 2: Strong Signals (15-19 points)

| Signal | Max Points |
|--------|------------|
| 90%+ category win rate | 25→20 |
| Category concentration > 90% | 20 |
| Return multiple > 10x | 20 |
| Large deposit before bet | 20 |
| Returns from dormancy | 20 |
| Position size > 50% capital | 20 |
| Contrarian longshot wins | 20 |
| News-trade correlation | 20 |
| Awards market | 20 |
| Market overlap > 90% | 20 |
| Account age < 7 days | 20 |
| First tx is large bet | 20 |
| Created for specific event | 20 |
| Coordinated timing (cluster) | 20 |
| Bet < 24 hours before event | 20 |

### Tier 3: Supporting Signals (10-14 points)

| Signal | Max Points |
|--------|------------|
| < 5 prior transactions | 15 |
| No hedging | 15 |
| Profit > $100K | 15 |
| Quick position exit | 15 |
| Off-hours trading | 15 |
| Direct CEX deposit | 15 |
| Immediate withdrawal | 15 |
| Account creation proximity | 15 |
| Proportional sizing | 15 |
| Same exchange origin | 15 |
| Coordinated exit | 15 |
| Tech launch market | 15 |
| Username change | 15 |
| ENS domain match | 15 |
| Dormancy pattern | 15 |

### Tier 4: Minor Signals (5-9 points)

| Signal | Max Points |
|--------|------------|
| Account age < 30 days | 10 |
| 80%+ overall win rate | 15→10 |
| Weekend trading | 10 |
| No privacy measures | 10 |
| Withdrawal to same CEX | 10 |
| Cross-chain movement | 10 |
| Account inactive after win | 10 |
| Funds moved to new wallet | 10 |
| Memecoin purchase | 5 |
| Deleted social presence | 5 |
| Public denial | 5 |

---

## Negative Modifiers (Reduce Score)

| Signal | Adjustment |
|--------|------------|
| Controllable outcome (Musk tweets) | -10 |
| Betting WITH consensus (Fed Rate case) | -15 |
| Bet lost (mutualdelta) | -30 |
| Long account history (> 100 trades) | -15 |
| High market diversity (> 10 categories) | -10 |
| Published analysis justifying trade | -5 |
| Known public analyst/researcher | -10 |

---

## Signal Combinations (Multipliers)

### Combo 1: Perfect Insider Template
```
fresh_wallet + single_market + pre_event_timing + won
→ 1.5x multiplier
```

### Combo 2: Cluster Attack
```
multiple_wallets + same_funding + synchronized_timing
→ 1.4x multiplier
```

### Combo 3: Government Insider
```
policy_market + 30min_timing + large_profit + fresh_wallet
→ 1.5x multiplier
```

### Combo 4: Corporate Insider
```
tech_launch_market + exact_date + category_specialist
→ 1.3x multiplier
```

---

## Case-by-Case Signal Summary

### Burdensome-Mix (Score: 95)
- Account age 7 days (+20)
- Single market focus (+25)
- 100% win rate (+30)
- 3-hour pre-event (+25)
- Large longshot bet (+20)
- Direct Coinbase (+15)
- Geopolitical market (+30)
- **Combo: Perfect Insider** (1.5x)

### ricosuave666 (Score: 92)
- 100% category win rate (+30)
- Exact date prediction (+30)
- Category concentration 100% (+25)
- Military market (+30)
- Dormancy return pattern (+20)
- Username change (+15)
- **Combo: Government Insider** (1.5x)

### 6741 (Score: 94)
- Account age 24h (+25)
- Zero prior trades (+25)
- Single market (+25)
- Entry odds 4% (+25)
- 25x return (+20)
- Awards market (+20)
- **Combo: Perfect Insider** (1.5x)

### 0xafEe (Score: 88)
- 95.6% win rate (+25)
- Category concentration 100% (+25)
- Large deposit $3M (+20)
- Exact date (Gemini) (+30)
- Tech market (+15)
- Longshot wins (+20)
- **Combo: Corporate Insider** (1.3x)

### bigwinner01 (Score: 78)
- 30-min timing (+30)
- Policy market (+25)
- Cross-platform correlation (+15)
- Large position $190M (+20)
- Public denial (-5)
- Longer account history (-10)

---

## Implementation: Signal Checker Functions

```python
class SignalChecker:
    def check_all_signals(self, wallet, market=None):
        signals = []

        # Account signals
        signals.extend(self.check_account_signals(wallet))

        # Trading signals
        signals.extend(self.check_trading_signals(wallet))

        # Timing signals
        if market:
            signals.extend(self.check_timing_signals(wallet, market))

        # Funding signals
        signals.extend(self.check_funding_signals(wallet))

        # Cluster signals
        signals.extend(self.check_cluster_signals(wallet))

        # Context signals
        if market:
            signals.extend(self.check_context_signals(market))

        # Evasion signals
        signals.extend(self.check_evasion_signals(wallet))

        return signals

    def calculate_score(self, signals):
        base_score = sum(s.points for s in signals)

        # Apply combos
        multiplier = self.detect_combos(signals)

        # Apply negative modifiers
        adjustments = self.calculate_adjustments(signals)

        final_score = (base_score * multiplier) + adjustments
        return min(final_score, 100)
```

---

## Total Unique Signals: 60

- Account: 11 signals
- Trading: 14 signals
- Timing: 9 signals
- Funding: 8 signals
- Cluster: 7 signals
- Context: 6 signals
- Evasion: 5 signals

**This is the most comprehensive signal set extracted from all documented cases.**
