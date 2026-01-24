# Insider Footprint: Portugal Election Insiders (January 2026)

## Summary
- **Event:** Portugal Presidential Election
- **Candidate:** António José Seguro (winner)
- **Suspicious Volume:** €5+ million in 2-hour window
- **Timing:** 1-2 hours before polls closed, exit polls circulating
- **Outcome:** Portugal banned Polymarket
- **Status:** Under regulatory investigation

---

## Event Timeline

| Time (Local) | Event | Odds Movement |
|--------------|-------|---------------|
| 6:00 PM | Exit polls begin circulating privately | Seguro: 68.6% |
| 6:30 PM | Massive buying surge begins | Seguro: 68.6% → 93.2% |
| 7:00 PM | Polls close | Seguro: ~96% |
| 8:00 PM | Official projections released | Seguro: 100% |

**Key Window:** 6:00 PM - 8:00 PM (2 hours)
**Volume:** €5+ million traded

---

## Trading Pattern Analysis

### Odds Movements

| Candidate | 6:00 PM | 6:30 PM | 7:00 PM | Change |
|-----------|---------|---------|---------|--------|
| Seguro | 60% | 68.6% → 93.2% | 96% | +36% |
| Cotrim | 22% | 22% → 2.5% | ~1% | -19.5% |
| Ventura | 30% | Declining | Low | Crashed |

### Volume Statistics
- Main presidential market: $120M+ total (~€103M)
- Alternative markets: ~$10M (~€8.1M)
- 2-hour suspicious window: €5M+

---

## Detection Signals

### Why This Is Suspicious

1. **Exit poll timing:** Odds moved when private exit polls circulated
2. **Speed of movement:** 25% swing in 1 hour
3. **Volume concentration:** €5M in 2-hour window
4. **Pre-announcement accuracy:** 96% certainty before official results
5. **Information asymmetry:** Private exit polls not public

### Regulatory Findings
- SRIJ (Portugal gaming regulator) ordered shutdown
- Cited illegal political wagering
- No license to operate in Portugal
- 48-hour shutdown order issued

---

## Behavioral Footprint

```json
{
  "signals": {
    "timing_window": {"hours_before_result": 2, "weight": 25},
    "odds_movement_speed": {"percent_per_hour": 25, "weight": 25},
    "volume_spike": {"vs_baseline_multiple": 10, "weight": 20},
    "exit_poll_correlation": {"timing_match": true, "weight": 20},
    "multi_account_coordinated": {"likely": true, "weight": 10}
  },
  "category": {
    "type": "election",
    "nation": "portugal",
    "event": "presidential"
  },
  "detection_method": "volume_spike_before_result"
}
```

---

## Pattern: Election Insider Trading

### Common Characteristics Across Elections

| Signal | Portugal | Trump 2024 | Other |
|--------|----------|------------|-------|
| Exit poll timing | Yes | N/A | Common |
| Volume spike | 10x+ | High | Common |
| Odds accuracy | 96%+ pre-result | Accurate | Common |
| Coordinated buying | Likely | Yes (Théo) | Common |

### Detection Strategy for Elections

1. **Monitor exit poll timing:** Know when private polls circulate
2. **Set volume alerts:** Flag 5x+ baseline in final hours
3. **Track odds velocity:** Alert on >10%/hour movements
4. **Identify coordinated accounts:** Same timing, same direction

---

## Regulatory Context

### Countries That Have Banned Polymarket (as of Jan 2026)
- Portugal (Jan 2026)
- Hungary (Jan 2026)
- France
- Switzerland
- Poland
- Italy
- Belgium
- Germany

### Common Trigger: Election Insider Trading
Most bans followed suspicious election-related betting activity.

---

## Application to Scanner

### Election Market Monitoring Rules

```python
def monitor_election_market(market):
    # Set baseline volume
    baseline = get_average_daily_volume(market, days=7)

    # Calculate hours until resolution
    hours_remaining = get_hours_until_resolution(market)

    # Alert conditions
    if hours_remaining < 24:
        if current_hour_volume > baseline * 3:
            alert("VOLUME_SPIKE", market, "3x baseline in final 24h")

        if odds_change_per_hour > 10:
            alert("ODDS_VELOCITY", market, f"{odds_change_per_hour}% per hour")

        if new_accounts_entering > 5:
            alert("NEW_ACCOUNT_CLUSTER", market, "Multiple new accounts")
```

---

## Sources

- https://finance.yahoo.com/news/portugal-bans-polymarket-over-4m-155854731.html
- https://www.yahoo.com/news/articles/portugal-cracks-down-polymarket-following-112901661.html
- https://www.tradingview.com/news/cointelegraph:924f2c135094b:0-polymarket-hit-by-fresh-european-crackdowns-as-hungary-portugal-block-access/
- https://news.worldcasinodirectory.com/portugal-shuts-down-polymarket-amid-suspicious-betting-on-presidential-election-121384
