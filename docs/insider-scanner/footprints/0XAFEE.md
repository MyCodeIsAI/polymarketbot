# Insider Footprint: 0xafEe (Google Year in Search)

## Summary
- **Username:** 0xafEe (formerly AlphaRacoon)
- **Wallet:** `0xafEe...` (changed name to hide, still on-chain)
- **Event:** Google Year in Search 2025
- **Profit:** ~$1,150,000
- **Date:** December 2025
- **Status:** No investigation announced, circumstantial evidence

---

## Account Characteristics

| Attribute | Value | Detection Signal |
|-----------|-------|------------------|
| Initial Deposit | $3,000,000 | Massive funding |
| Deposit Date | "Last Friday" before bets | Fresh capital |
| Win Rate (Google markets) | 22/23 (95.6%) | Near-perfect |
| Strategy | Bet YES on longshots, NO on favorites | Information arbitrage |
| Username Change | AlphaRacoon → 0xafEe | Attempted obfuscation |

---

## Trading Pattern

### Key Winning Trades

| Market | Position | Entry Odds | Investment | Profit |
|--------|----------|------------|------------|--------|
| d4vd #1 Searched Person | YES | 0.2% | $10,647 | ~$200,000 |
| Pope Leo XIV | NO | High | Large | Significant |
| Bianca Censori | NO | High | Large | Significant |
| Donald Trump | NO | High | Large | Significant |
| Gemini 3.0 Launch Date | YES | Low | Unknown | $150,000 |

### Historical Pattern

| Date | Event | Accuracy |
|------|-------|----------|
| November 2025 | Gemini 3.0 exact launch day | 100% |
| December 2025 | Google Year in Search | 22/23 (95.6%) |

---

## Behavioral Signals

### Pre-Event Indicators
1. **Large deposit immediately before betting:** $3M deposited Friday, bets placed same weekend
2. **Category specialization:** Google-related markets only
3. **Contrarian bets on longshots:** Betting YES on 0.2% probability outcomes
4. **Perfect streak:** 22/23 correct = statistically improbable
5. **Name change after exposure:** Attempted to hide identity
6. **Prior success:** Same pattern on Gemini 3.0 launch

### Timing Analysis
- Bets placed "hours before" Google published official data
- Possible insider access or early API leak

---

## Detection Footprint

```json
{
  "signals": {
    "deposit_size_usd": {"min": 100000, "weight": 15},
    "deposit_to_bet_days": {"max": 3, "weight": 15},
    "win_rate_category": {"min": 0.90, "weight": 25},
    "longshot_bets": {"prob_max": 0.05, "win_rate_min": 0.80, "weight": 25},
    "category_concentration": {"min": 0.95, "weight": 20}
  },
  "behavior": {
    "contrarian_strategy": true,
    "name_change_after_win": true,
    "multiple_large_longshot_wins": true
  }
}
```

---

## Comparison: Normal Trader vs 0xafEe

| Metric | Normal Trader | 0xafEe |
|--------|---------------|--------|
| Win rate on < 5% odds | < 10% | 95.6% |
| Category concentration | < 30% | > 95% |
| Deposit-to-bet timing | Varies | Immediate |
| Bet sizing | Conservative | Aggressive |
| Market diversity | High | Single category |

---

## API Queries to Replicate

### Find Profile by Username (scrape leaderboard or search)
```
GET https://polymarket.com/profile/0xafEe
```
Extract wallet address from profile page.

### Get All Trades
```
GET https://data-api.polymarket.com/trades?user={wallet_address}
```

### Calculate Win Rate
1. Filter trades by `type=TRADE`
2. Group by `conditionId`
3. Check resolution: price = 1.0 → WIN, price = 0 → LOSS
4. Calculate: wins / total

---

## Potential Corporate Insider Indicators

Given the Google-specific accuracy:
1. **IP Address Analysis:** May originate from Google offices/VPN
2. **Timing Correlation:** Bets coincide with internal data availability
3. **Pattern Repetition:** Same success on Gemini 3.0 suggests ongoing access
4. **Data Source:** Possible early access to Year in Search rankings

---

## Sources

- https://finance.yahoo.com/news/polymarket-trader-makes-1-million-090001027.html
- https://thedefiant.io/news/defi/polymarket-users-suspect-insider-trading-after-google-trend-markets-crown-surprise-winner
- https://gizmodo.com/polymarket-user-accused-of-1-million-insider-trade-on-google-search-markets-2000696258
- https://beincrypto.com/alleged-google-insider-trade-polymarket/
