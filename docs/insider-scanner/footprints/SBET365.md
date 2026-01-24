# Insider Footprint: SBet365 (Multi-Event Insider)

## Summary
- **Username:** SBet365
- **Wallet:** Known (exact address not public)
- **Events:** Venezuela Maduro, Iran Khamenei
- **Profit (Maduro):** $145,600 (from $25,000)
- **Date:** January 2026
- **Status:** Active, continues trading geopolitical markets

---

## Account Characteristics

| Attribute | Value | Detection Signal |
|-----------|-------|------------------|
| Account Creation | Days before Maduro event | Fresh account |
| Markets Traded | Geopolitical regime change only | Category specialization |
| Bet Focus | "Leader out of office" markets | Pattern specialization |
| Funding Source | Unknown (likely Coinbase) | Similar to cohort |
| Current Status | ACTIVE (as of Jan 2026) | Still trading |

---

## Trading History

### Event 1: Venezuela Maduro (January 2026)

| Date | Action | Amount | Result |
|------|--------|--------|--------|
| Late Dec 2025 | Account Created | - | Pre-event setup |
| Jan 2-3, 2026 | Buy YES "Maduro out" | $25,000 | Entry |
| Jan 3, 2026 | Resolution | $145,600 | WIN |

### Event 2: Iran Khamenei (January 2026)

| Date | Action | Details |
|------|--------|---------|
| ~Jan 14, 2026 | Buy YES "Khamenei out by Jan 31" | Active position |
| Current | Holding | Average entry: $0.20 |

**Market:** "Khamenei out as Supreme Leader of Iran by January 31"
- Volume: $28M+
- SBet365 buying YES at $0.20 average

---

## Behavioral Signals

### Pattern Recognition
1. **Regime change specialization:** Only bets on leaders leaving office
2. **Geopolitical focus:** Venezuela, Iran (US adversary nations)
3. **Government-adjacent events:** Military/covert operations
4. **Timing:** Created account days before major event
5. **Continues pattern:** Same strategy on Iran after Maduro success

### Why This Is Critical
- **Repeat offender:** Same pattern across multiple events
- **Predictive:** Current Iran position may indicate foreknowledge
- **Intelligence access:** Focus on US adversary regime change suggests possible intel community connection

---

## Related Wallets (Maduro Cohort)

| Wallet | Status Post-Maduro |
|--------|-------------------|
| `0x31a5...` (Burdensome-Mix) | Inactive since Jan 8 |
| `0xa72D...` | Inactive 11+ days |
| `SBet365` | **ACTIVE** - New Iran bets |

**Key Insight:** SBet365 is the only wallet from the cohort still active.

---

## Detection Footprint

```json
{
  "signals": {
    "account_age_days": {"max": 14, "weight": 15},
    "market_category": {"value": "geopolitical_regime_change", "weight": 25},
    "event_type": {"value": "leader_removal", "weight": 25},
    "cohort_membership": {"related_to_known_insiders": true, "weight": 20},
    "repeat_pattern": {"same_strategy_new_event": true, "weight": 15}
  },
  "behavior": {
    "specialization": "regime_change",
    "nations_focus": ["venezuela", "iran", "adversary_states"],
    "continues_after_exposure": true
  }
}
```

---

## Monitoring Priority: HIGH

### Rationale
1. **Active account:** Currently trading
2. **Predictive value:** Iran bet may indicate upcoming event
3. **Pattern match:** Same behavior as successful Maduro trade
4. **Intel hypothesis:** Possible ongoing access to classified information

### Recommended Actions
1. Monitor all SBet365 trades in real-time
2. Alert on any new "regime change" market entries
3. Cross-reference with US policy announcements
4. Track if Iran bet resolves profitably

---

## API Queries

### Real-Time Monitoring
```
# Subscribe to SBet365 activity via WebSocket or polling
GET https://data-api.polymarket.com/activity?user={SBet365_wallet}
```

### Position Tracking
```
GET https://data-api.polymarket.com/positions?user={SBet365_wallet}
```

### Alert Trigger
When SBet365 opens new position:
1. Check market category
2. If "regime_change" or "leader_removal" → HIGH ALERT
3. Log funding source for FLAG file

---

## Sources

- https://blockchain.news/flashnews/polymarket-insider-wallets-sbet365-bets-khamenei-out-by-jan-31
- https://www.cryptoninjas.net/news/630k-insider-bet-exposed-as-polymarket-wallets-predicted-maduros-fall-hours-before-arrest/
- https://finance.yahoo.com/news/trump-jails-venezuela-leaker-suspicious-163206487.html
- https://cointelegraph.com/news/trump-venezuela-leaker-jail-polymarket-accounts-go-quiet
