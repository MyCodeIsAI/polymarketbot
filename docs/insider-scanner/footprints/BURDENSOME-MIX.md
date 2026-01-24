# Insider Footprint: Burdensome-Mix (Maduro Trade)

## Summary
- **Username:** Burdensome-Mix
- **Wallet Prefix:** `0x31a5...` (partial, `0x31a56e...` mentioned)
- **Event:** Venezuela Maduro Removal
- **Profit:** $409,900 (from ~$34,000 investment)
- **Date:** January 2-3, 2026
- **Status:** CFTC Investigation, Suspect arrested per Trump statement

---

## Account Characteristics

| Attribute | Value | Detection Signal |
|-----------|-------|------------------|
| Account Creation | Dec 27, 2025 | 7 days before event |
| Days to First Bet | ~6 days | Fresh account |
| Total Markets Traded | 1 (Venezuela only) | Single-market focus |
| Funding Source | Coinbase (direct) | US exchange, traceable |
| Privacy Tools Used | None | No VPN/mixer |
| Withdraw Destination | Coinbase | Same as funding |

---

## Trading Timeline

| Timestamp (ET) | Action | Amount | Price | Notes |
|----------------|--------|--------|-------|-------|
| Dec 27, 2025 | Account Created | - | - | 7 days pre-event |
| Jan 2, 2026 9:00 PM | Buy YES | ~$12,000 | $0.08 | Aggressive accumulation |
| Jan 2, 2026 9:00-11:30 PM | Buy YES | ~$12,000 | $0.08-0.22 | Pushed price from 8% to 22% |
| Jan 3, 2026 1:38-2:58 AM | Buy YES | $20,000 | Low | "Last-minute infusion" |
| Jan 3, 2026 ~6:00 AM | Event Occurs | - | - | Operation Absolute Resolve |
| Jan 3, 2026 | Resolution | $436,000 | $1.00 | 1262% return |

---

## Behavioral Signals

### Pre-Event Indicators
1. **Account age at first trade:** < 7 days
2. **Market concentration:** 100% Venezuela-related
3. **No hedging:** All-in on single outcome
4. **Timing precision:** Trades 1-6 hours before event
5. **Size relative to odds:** $34K on 8% probability = extreme conviction
6. **Trading hours:** 1:38 AM - 2:58 AM (suspicious timing)

### Funding Pattern
```
Coinbase → Polymarket Proxy Wallet → Bet → Win → Polymarket → Coinbase
```
- No intermediary wallets
- No privacy measures
- Direct cash-out route

---

## Related Wallets (Same Event)

| Wallet | Investment | Profit | Creation |
|--------|------------|--------|----------|
| `0x31a5...` (Burdensome-Mix) | $34,000 | $409,900 | Dec 27, 2025 |
| `0xa72D...` | $5,800 | $75,000 | Days before |
| `SBet365` | $25,000 | $145,600 | Days before |

**Combined:** $630,484 profit from 3 wallets

---

## Detection Footprint

```json
{
  "signals": {
    "account_age_days": {"max": 7, "weight": 20},
    "markets_traded": {"max": 1, "weight": 25},
    "win_rate": {"min": 1.0, "weight": 15},
    "bet_vs_probability": {"ratio_min": 4, "weight": 20},
    "timing_hours_before_event": {"max": 24, "weight": 20}
  },
  "funding": {
    "source_type": "centralized_exchange",
    "exchange": "coinbase",
    "direct_deposit": true
  },
  "behavior": {
    "trading_hours": "off_peak",
    "hedge_positions": false,
    "market_diversity": "none"
  }
}
```

---

## API Queries to Replicate

### Get Profile
```
GET https://gamma-api.polymarket.com/public-profile?address=0x31a5...
```

### Get Trades
```
GET https://data-api.polymarket.com/trades?user=0x31a5...
```

### Get Activity
```
GET https://data-api.polymarket.com/activity?user=0x31a5...&type=TRADE
```

### On-Chain Funding Source
```
GET https://api.polygonscan.com/api?module=account&action=tokentx&address=0x31a5...&apikey=KEY
```
Look for first USDC transfer TO this address to identify funding source.

---

## Sources

- https://www.npr.org/2026/01/05/nx-s1-5667232/polymarket-maduro-bet-insider-trading
- https://markets.financialcontent.com/stocks/article/predictstreet-2026-1-23-the-400000-whistleblower-how-the-maduro-trade-shook-polymarket-and-washington
- https://x.com/lookonchain/status/2007639475497881625
- https://www.cryptoninjas.net/news/630k-insider-bet-exposed-as-polymarket-wallets-predicted-maduros-fall-hours-before-arrest/
