# Insider Footprint: dirtycup (Nobel Peace Prize)

## Summary
- **Username:** dirtycup
- **Wallet:** Unknown (investigation pending)
- **Event:** 2025 Nobel Peace Prize
- **Bet:** $68,340 on María Corina Machado
- **Profit:** ~$31,000
- **Date:** October 9-10, 2025
- **Status:** Norwegian investigation ongoing, possible cyber espionage

---

## Account Characteristics

| Attribute | Value | Detection Signal |
|-----------|-------|------------------|
| Account Creation | Weeks before event | Not fresh, but new |
| Prior Betting History | None | First-time bettor |
| Markets Traded | 1 (Nobel Prize only) | Single-market focus |
| Bet Timing | 3:41 AM UTC | Off-hours, pre-announcement |
| Position Size | $68,340 | Large single bet |
| Entry Odds | ~8 cents ($0.08) | Extreme longshot |

---

## Trading Timeline

| Timestamp (UTC) | Event | Details |
|-----------------|-------|---------|
| Weeks prior | Account Created | No prior activity |
| Oct 9, 2025 ~midnight | Buying begins | Market-wide surge starts |
| Oct 9, 2025 3:41 AM | dirtycup bets $68,340 | YES on Machado |
| Oct 9-10 (6 hours) | Price surge | 4% → 73% |
| Oct 10, 2025 9:00 AM | Nobel Announced | Machado wins |
| - | Resolution | dirtycup profits ~$31K |

---

## Related Accounts (Same Event)

| Account | Investment | Entry Timing | Profit |
|---------|------------|--------------|--------|
| dirtycup | $68,340 | 3:41 AM UTC | ~$31,000 |
| 6741 | $2,000 | 24 hrs before | $51,000 |
| GayPride | Unknown | Unknown | Unknown |

**Combined:** ~$90,000+ profit from coordinated trades

---

## Behavioral Signals

### Pre-Event Indicators
1. **Zero prior history:** Account existed but never traded
2. **Single market:** 100% concentration on Nobel Prize
3. **Off-hours trading:** 3:41 AM UTC (suspicious timing)
4. **Large size on longshot:** $68K on 4% probability
5. **No hedging:** All-in on single outcome
6. **11-hour foreknowledge:** Price moved before announcement

### Investigation Findings

**Initial Theory:** Human mole within Nobel Committee
**Revised Theory:** "Systematic espionage" / cyber breach

> "We likely fell prey to a sophisticated cyber breach" - Kristian Berg Harpviken, Norwegian Nobel Institute Director

---

## Detection Footprint

```json
{
  "signals": {
    "prior_trade_history": {"count": 0, "weight": 25},
    "markets_traded": {"max": 1, "weight": 20},
    "bet_timing_hours": {"off_peak": true, "weight": 15},
    "longshot_bet": {"entry_odds_max": 0.10, "weight": 20},
    "single_bet_size_usd": {"min": 50000, "weight": 20}
  },
  "timing": {
    "hours_before_announcement": {"max": 12},
    "trading_hour_utc": {"suspicious_range": [0, 6]}
  },
  "category": {
    "type": "award_announcement",
    "entity": "governmental_committee"
  }
}
```

---

## Nobel Prize Market Characteristics

### Why This Market Is Vulnerable
1. **Information asymmetry:** Committee decides in secret
2. **Small information circle:** Few people know in advance
3. **Precise timing:** Announced at specific time
4. **Binary outcome:** Clear winner
5. **Low baseline probability:** Many candidates = low individual odds

### Detection Strategy for Award Markets
1. Monitor for sudden price spikes 6-24 hours before announcement
2. Flag accounts with zero history entering award markets
3. Track off-hours trading activity (overnight bets)
4. Watch for multiple accounts entering simultaneously

---

## API Queries

### Historical Analysis
```
# Get all trades on Nobel Prize market in 24 hours before announcement
GET https://data-api.polymarket.com/trades?conditionId={nobel_market}&after={timestamp_24h_before}
```

### Pattern Detection
```
# For each trader in above results:
# 1. Check account age
GET https://gamma-api.polymarket.com/public-profile?address={wallet}

# 2. Check prior trading history
GET https://data-api.polymarket.com/trades?user={wallet}

# 3. Flag if: prior_trades == 0 AND bet_size > $10K
```

---

## Regulatory Implications

### Norwegian Investigation Scope
1. Identify account owners via exchange KYC
2. Subpoena Polymarket for user data
3. Investigate Nobel Committee security breach
4. Potential cyber espionage charges

### Legal Uncertainty
- Polymarket TOS doesn't prohibit insider trading
- Offshore platform = unclear jurisdiction
- May require international cooperation

---

## Sources

- https://markets.financialcontent.com/stocks/article/predictstreet-2026-1-21-the-nobel-leak-how-prediction-markets-unmasked-the-2025-peace-prize
- https://blockworks.co/news/polymarket-nobel-probe
- https://protos.com/polymarket-traders-accused-of-insider-trading-nobel-peace-prize/
- https://www.securitiesdocket.com/2025/10/12/norwegian-officials-probe-major-polymarket-bets-on-nobel-peace-winner/
