# Insider Footprint: Iran Strike Wallets (Synchronized Bets)

## Summary
- **Wallets:** 4+ newly created, synchronized
- **Event:** US Strikes Iran predictions
- **Pattern:** Coordinated low-odds betting
- **Notable Failure:** mutualdelta lost $40K (Jan 14 market)
- **Status:** Mixed results - some wins, some losses

---

## Wallet Cluster Details

### The 4 Synchronized Wallets (Jan 2026)

| Wallet | Position Value | Entry Odds | Market |
|--------|---------------|------------|--------|
| Wallet 1 | $2,888 | < 18% | US strikes Iran by Jan 31 |
| Wallet 2 | $3,863 | < 18% | US strikes Iran by Jan 31 |
| Wallet 3 | $1,167 | < 18% | US strikes Iran by Jan 31 |
| Wallet 4 | $9,933 | < 18% | US strikes Iran by Jan 31 |

**Total Combined:** ~$17,851

### Common Characteristics
- All wallets newly created
- All placed bets at same low odds (< 18%)
- None made any other bets
- Synchronized timing

### mutualdelta Wallet (Failed Insider)

| Attribute | Value |
|-----------|-------|
| Wallet | mutualdelta |
| Market | US strikes Iran by January 14, 2026 |
| Position | YES |
| Investment | $40,000 |
| Shares | 255,817 |
| Potential Payout | $160,000 |
| Result | **TOTAL LOSS (-100%)** |
| Market Resolution | NO |

**Key Lesson:** Not all "insider" patterns win. False positives exist.

---

## Why This Pattern Is Important

### Insider Signals Present
1. **New wallets:** Created specifically for these bets
2. **Synchronized timing:** All bets placed in same window
3. **Single market focus:** No diversification
4. **Low-odds entry:** Betting on unlikely outcomes
5. **Coordinated sizing:** Suggests same controller

### But Also: Failure Example
- mutualdelta followed same pattern but **lost everything**
- Demonstrates that pattern matching isn't 100% predictive
- May be copycats, not actual insiders
- Information may have been wrong or plans changed

---

## Detection Footprint

```json
{
  "cluster_signals": {
    "wallets_created_same_period": {"count_min": 2, "days": 7, "weight": 25},
    "same_market_entry": {"overlap": 1.0, "weight": 25},
    "synchronized_timing": {"minutes_apart": 60, "weight": 20},
    "low_odds_entry": {"max_odds": 0.20, "weight": 15},
    "no_other_activity": {"other_markets": 0, "weight": 15}
  },
  "category": {
    "type": "military_operations",
    "nations": ["iran", "us"],
    "event_type": "airstrikes"
  },
  "confidence_modifier": {
    "reason": "mixed_outcomes",
    "adjustment": -10
  }
}
```

---

## Pattern Analysis: Geopolitical Betting Clusters

### Why Geopolitical Markets Attract Insiders

1. **Classified information:** Military ops are secret
2. **Binary outcomes:** Clear yes/no resolution
3. **High leverage:** Low odds = high payout
4. **Limited participants:** Easier to move market
5. **Government employees:** Large pool with access

### Detection Strategy

```python
def detect_geopolitical_cluster(market):
    if market.category in ['military', 'geopolitical', 'regime_change']:
        new_wallets = get_new_wallets_entering(market, hours=24)

        if len(new_wallets) >= 3:
            # Check for synchronization
            entry_times = [w.first_trade_time for w in new_wallets]
            if max(entry_times) - min(entry_times) < timedelta(hours=2):
                # Check for low-odds betting
                avg_entry_odds = mean([w.entry_price for w in new_wallets])
                if avg_entry_odds < 0.20:
                    alert("SYNCHRONIZED_CLUSTER", market, {
                        "wallet_count": len(new_wallets),
                        "avg_odds": avg_entry_odds,
                        "time_spread": max(entry_times) - min(entry_times)
                    })
```

---

## Information Laundering Concern

### The Feedback Loop

1. **Insider places bet** → Odds move
2. **Lookonchain/analysts flag** → Social media amplifies
3. **Copy traders pile in** → Odds spike further
4. **Original trader exits** → Profits from momentum
5. **Market may not resolve as predicted** → Late followers lose

### mutualdelta as Example
- Followed "insider" pattern
- Got flagged by analysts
- Market resolved NO
- Lost $40K
- May have been information laundering victim

---

## Scoring Implications

### Confidence Levels for Geopolitical Clusters

| Signal Combination | Confidence | Action |
|-------------------|------------|--------|
| New wallet + low odds + single market | Medium | Watchlist |
| + Synchronized with 2+ other wallets | High | Alert |
| + Prior wallet has win history | Very High | Immediate Alert |
| But: Market resolves NO | Reduce confidence | Review pattern |

---

## Sources

- https://blockchain.news/flashnews/polymarket-us-strikes-iran-by-jan-31-2026-bets-4-new-wallets-make-synchronized-wagers-below-18-odds-trading-signal-watch
- https://blockchain.news/flashnews/polymarket-insider-bet-fails-wallet-mutualdelta-loses-40k-on-us-strikes-iran-by-jan-14-2026-market
- https://www.cryptopolitan.com/insider-traders-polymarket-us-attack-iran/
- https://beincrypto.com/polymarket-information-laundering-maduro-iran-bets/
