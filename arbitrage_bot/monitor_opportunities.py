#!/usr/bin/env python3
"""
Continuous opportunity monitor for the arbitrage bot.
Audits the system when signals should fire.
"""

import json
import time
import sys
from datetime import datetime
import requests

SERVER_URL = "http://localhost:8766"

# Thresholds (must match server config)
AGGRESSIVE_BUY_THRESHOLD = 0.25
STANDARD_BUY_THRESHOLD = 0.35
TARGET_PAIR_COST = 0.95
MAX_PAIR_COST = 1.10
MIN_TIME_TO_RESOLUTION_SEC = 60

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def colored(text, color):
    return f"{color}{text}{Colors.RESET}"

def check_server():
    """Check if server is responding."""
    try:
        resp = requests.get(f"{SERVER_URL}/arbitrage/api/status", timeout=5)
        return resp.ok
    except:
        return False

def get_markets():
    """Get current market data."""
    try:
        resp = requests.get(f"{SERVER_URL}/arbitrage/api/markets", timeout=5)
        if resp.ok:
            return resp.json()
    except Exception as e:
        print(f"Error fetching markets: {e}")
    return {}

def get_status():
    """Get server status."""
    try:
        resp = requests.get(f"{SERVER_URL}/arbitrage/api/status", timeout=5)
        if resp.ok:
            return resp.json()
    except Exception as e:
        print(f"Error fetching status: {e}")
    return {}

def get_signals():
    """Get recent signals."""
    try:
        resp = requests.get(f"{SERVER_URL}/arbitrage/api/signals", timeout=5)
        if resp.ok:
            return resp.json()
    except Exception as e:
        print(f"Error fetching signals: {e}")
    return []

def analyze_opportunity(market):
    """Analyze if a market has an opportunity and what the system should do."""
    up_price = market['up_price']
    down_price = market['down_price']
    pair_cost = market['pair_cost']
    resolution_time = market.get('resolution_time')

    analysis = {
        'has_opportunity': False,
        'opportunity_type': None,
        'side': None,
        'price': None,
        'should_fire': False,
        'blocking_reason': None,
    }

    # Check for opportunity
    if up_price < AGGRESSIVE_BUY_THRESHOLD:
        analysis['has_opportunity'] = True
        analysis['opportunity_type'] = 'AGGRESSIVE'
        analysis['side'] = 'UP'
        analysis['price'] = up_price
    elif up_price < STANDARD_BUY_THRESHOLD:
        analysis['has_opportunity'] = True
        analysis['opportunity_type'] = 'STANDARD'
        analysis['side'] = 'UP'
        analysis['price'] = up_price
    elif down_price < AGGRESSIVE_BUY_THRESHOLD:
        analysis['has_opportunity'] = True
        analysis['opportunity_type'] = 'AGGRESSIVE'
        analysis['side'] = 'DOWN'
        analysis['price'] = down_price
    elif down_price < STANDARD_BUY_THRESHOLD:
        analysis['has_opportunity'] = True
        analysis['opportunity_type'] = 'STANDARD'
        analysis['side'] = 'DOWN'
        analysis['price'] = down_price

    if not analysis['has_opportunity']:
        return analysis

    # Check blocking conditions
    if pair_cost > MAX_PAIR_COST:
        analysis['blocking_reason'] = f"Pair cost ${pair_cost:.2f} > MAX ${MAX_PAIR_COST:.2f}"
        return analysis

    if resolution_time:
        try:
            res_time = datetime.fromisoformat(resolution_time.replace('Z', '+00:00'))
            res_time = res_time.replace(tzinfo=None)
            time_to_res = (res_time - datetime.utcnow()).total_seconds()
            if time_to_res < MIN_TIME_TO_RESOLUTION_SEC:
                analysis['blocking_reason'] = f"Too close to resolution ({time_to_res:.0f}s < {MIN_TIME_TO_RESOLUTION_SEC}s)"
                return analysis
        except:
            pass

    # Passed all checks - should fire!
    analysis['should_fire'] = True
    return analysis

def print_market_status(markets, status, signals):
    """Print formatted market status."""
    now = datetime.now().strftime('%H:%M:%S')

    print(f"\n{colored('='*70, Colors.CYAN)}")
    print(f"{colored(f' ARBITRAGE MONITOR - {now}', Colors.BOLD)}")
    print(colored('='*70, Colors.CYAN))

    # Status summary
    print(f"\n{colored('Server Status:', Colors.BOLD)}")
    print(f"  Running: {colored('YES', Colors.GREEN) if status.get('running') else colored('NO', Colors.RED)}")
    print(f"  Markets: {status.get('markets_count', 0)}")
    print(f"  Signals: {status.get('signals_count', 0)}")

    latency = status.get('latency', {})
    if latency.get('e2e', {}).get('count', 0) > 0:
        e2e = latency['e2e']
        print(f"  E2E Latency: avg={e2e['avg']:.0f}ms p50={e2e['p50']:.0f}ms p95={e2e['p95']:.0f}ms")

    # Market analysis
    print(f"\n{colored('Market Analysis:', Colors.BOLD)}")
    print(f"  {'Asset':<6} {'Up':>8} {'Down':>8} {'Pair':>8} {'Status':<30}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*30}")

    opportunities = []
    for cid, m in markets.items():
        asset = m['slug'].split('-')[0].upper()
        up = m['up_price']
        down = m['down_price']
        pair = m['pair_cost']

        analysis = analyze_opportunity(m)

        if analysis['should_fire']:
            status_str = colored(f"FIRE {analysis['opportunity_type']} {analysis['side']} @ ${analysis['price']:.2f}", Colors.GREEN + Colors.BOLD)
            opportunities.append((asset, m, analysis))
        elif analysis['has_opportunity']:
            status_str = colored(f"BLOCKED: {analysis['blocking_reason']}", Colors.YELLOW)
        elif pair < 0.98:
            status_str = colored(f"Profitable pair (${pair:.2f})", Colors.CYAN)
        elif pair < 1.10:
            status_str = f"Waiting (pair ${pair:.2f})"
        else:
            status_str = colored(f"No opp (pair ${pair:.2f})", Colors.RED)

        print(f"  {asset:<6} ${up:>6.2f} ${down:>6.2f} ${pair:>6.2f} {status_str}")

    # If opportunities exist, perform audit
    if opportunities:
        print(f"\n{colored('!! OPPORTUNITY DETECTED !!', Colors.GREEN + Colors.BOLD)}")
        for asset, m, analysis in opportunities:
            print(f"\n{colored(f'AUDIT - {asset}:', Colors.YELLOW + Colors.BOLD)}")
            print(f"  Type: {analysis['opportunity_type']}")
            print(f"  Side: {analysis['side']}")
            print(f"  Price: ${analysis['price']:.2f}")
            print(f"  Pair Cost: ${m['pair_cost']:.2f}")
            print(f"  Expected profit per pair: ${1.00 - m['pair_cost']:.2f}")

            # Check if signal was generated
            recent_signals = [s for s in signals if s.get('condition_id') == m['condition_id']]
            if recent_signals:
                latest = recent_signals[-1]
                print(f"  {colored('Signal Generated:', Colors.GREEN)} {latest.get('signal_type')} at {latest.get('timestamp')}")
                print(f"  Signal latency: {latest.get('latency_ms', 'N/A')}ms")
            else:
                print(f"  {colored('WARNING: No signal detected!', Colors.RED)}")
                print(f"  System should have fired but didn't - investigate!")

    # Recent signals
    if signals:
        print(f"\n{colored('Recent Signals (last 5):', Colors.BOLD)}")
        for s in signals[-5:]:
            sig_asset = s.get('slug', '').split('-')[0].upper()[:4]
            print(f"  [{s.get('timestamp', '')[-12:-7]}] {sig_asset} {s.get('signal_type', '')} "
                  f"side={s.get('side', '')} price=${s.get('price', 0):.2f} latency={s.get('latency_ms', 'N/A')}ms")

    print()

def main():
    print(colored("\nStarting Arbitrage Opportunity Monitor...", Colors.BOLD))
    print(f"Server: {SERVER_URL}")
    print(f"Thresholds: Aggressive < ${AGGRESSIVE_BUY_THRESHOLD:.2f}, Standard < ${STANDARD_BUY_THRESHOLD:.2f}")
    print(f"Max Pair Cost: ${MAX_PAIR_COST:.2f}")
    print(colored("Press Ctrl+C to stop\n", Colors.YELLOW))

    if not check_server():
        print(colored("ERROR: Server not responding!", Colors.RED))
        sys.exit(1)

    poll_interval = 10  # seconds

    try:
        while True:
            markets = get_markets()
            status = get_status()
            signals = get_signals()

            if markets:
                print_market_status(markets, status, signals)
            else:
                print(f"{datetime.now().strftime('%H:%M:%S')} - No market data available")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(colored("\nMonitor stopped.", Colors.YELLOW))

if __name__ == "__main__":
    main()
