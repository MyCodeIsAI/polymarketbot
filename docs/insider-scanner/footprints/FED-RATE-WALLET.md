# Insider Footprint: Fed Rate Bet Wallet (January 2026)

## Summary
- **Username:** Unknown
- **Wallet:** Unknown (flagged by Lookonchain)
- **Event:** Fed January 2026 FOMC Meeting
- **Bet:** "No change in Fed interest rates"
- **Investment:** $17,100
- **Account Age:** 2 hours old at time of bet
- **Date:** January 2026
- **Status:** Flagged, outcome pending

---

## Account Characteristics

| Attribute | Value | Detection Signal |
|-----------|-------|------------------|
| Account Age | **2 hours** | Extremely fresh |
| First Transaction | This bet | Single-purpose wallet |
| Bet Size | $17,100 | Significant |
| Market | Fed Rate Decision | Government policy |
| Market Consensus | 98% "No Change" | Betting with consensus |

---

## Why This Is Flagged

### Red Flags

1. **2-hour old wallet:** Created specifically for this bet
2. **Single transaction:** No other activity
3. **Government policy market:** FOMC decisions are sensitive
4. **Immediate large bet:** No warmup activity

### Counter-Arguments

1. **Betting WITH consensus:** 98% already expected no change
2. **Limited edge:** Not a contrarian longshot bet
3. **Lower suspicion:** May be savvy trader, not insider

---

## Detection Footprint

```json
{
  "signals": {
    "account_age_hours": {"max": 24, "weight": 25},
    "first_transaction_is_large_bet": {"min_usd": 5000, "weight": 20},
    "government_policy_market": {"fed_fomc": true, "weight": 15},
    "single_market_focus": {"only_one_bet": true, "weight": 15}
  },
  "category": {
    "type": "monetary_policy",
    "entity": "federal_reserve",
    "event": "fomc_meeting"
  },
  "confidence_modifier": {
    "betting_with_consensus": -15,
    "reason": "Not contrarian, lower insider likelihood"
  }
}
```

---

## FOMC Market Insider Patterns

### Why Fed Markets Are High-Risk

1. **Centralized decision:** 12 FOMC members know outcome
2. **Advance preparation:** Staff prepare materials days ahead
3. **Leak history:** Fed leaks have occurred historically
4. **Market impact:** Massive financial implications

### Detection Strategy for FOMC Markets

```python
def monitor_fomc_market(market):
    # Alert on new wallets entering FOMC markets
    if market.type == "fed_rate_decision":
        for wallet in get_new_wallets(market, hours=48):
            if wallet.age_hours < 24 and wallet.first_bet_size > 5000:
                alert("FOMC_FRESH_WALLET", {
                    "wallet": wallet,
                    "age_hours": wallet.age_hours,
                    "bet_size": wallet.first_bet_size
                })
```

---

## Similar Policy Markets to Monitor

| Market Type | Decision Maker | Risk Level |
|-------------|---------------|------------|
| Fed Rate Decision | FOMC | High |
| Treasury Actions | Treasury Dept | High |
| Executive Orders | White House | Very High |
| Regulatory Rulings | SEC/CFTC/FDA | High |
| Trade Policy | USTR | High |

---

## Sources

- https://x.com/lookonchain/status/2008790669025046937
- Lookonchain Twitter flagging
