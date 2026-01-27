#!/bin/bash
# =============================================================================
# Polymarket Arbitrage Bot - Deployment Script
# =============================================================================
# This script handles full deployment including:
# - Python virtual environment setup
# - Dependency installation
# - Service configuration
# - Firewall setup
# - Auto-restart on failure
#
# Usage:
#   ./deploy.sh              # Deploy in simulation mode (default)
#   ./deploy.sh --live       # Deploy in LIVE trading mode
#   ./deploy.sh --port 8769  # Specify custom port
#   ./deploy.sh --stop       # Stop the running bot
#   ./deploy.sh --status     # Check bot status
#   ./deploy.sh --logs       # View recent logs
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BOT_NAME="arbitrage_bot"
DEFAULT_PORT=8769
LOG_FILE="/var/log/arbitrage_bot.log"
PID_FILE="/var/run/arbitrage_bot.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
PORT=$DEFAULT_PORT
LIVE_MODE=""
ACTION="deploy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --live)
            LIVE_MODE="--live"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --logs)
            ACTION="logs"
            shift
            ;;
        --restart)
            ACTION="restart"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Action: Stop
# =============================================================================
stop_bot() {
    log_info "Stopping arbitrage bot..."

    # Find and kill the process
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$PID_FILE"
            log_success "Bot stopped (PID: $PID)"
        else
            log_warn "PID file exists but process not running"
            rm -f "$PID_FILE"
        fi
    else
        # Try to find by process name
        PIDS=$(pgrep -f "arbitrage_bot/server.py" || true)
        if [ -n "$PIDS" ]; then
            echo "$PIDS" | xargs kill 2>/dev/null || true
            log_success "Bot stopped"
        else
            log_warn "Bot not running"
        fi
    fi
}

# =============================================================================
# Action: Status
# =============================================================================
check_status() {
    log_info "Checking arbitrage bot status..."

    # Check if process is running
    PIDS=$(pgrep -f "arbitrage_bot/server.py" || true)
    if [ -n "$PIDS" ]; then
        log_success "Bot is running (PID: $PIDS)"

        # Check API
        if curl -s --connect-timeout 5 "http://localhost:$PORT/arbitrage/api/status" > /dev/null 2>&1; then
            log_success "API is responding on port $PORT"

            # Get quick stats
            STATUS=$(curl -s "http://localhost:$PORT/arbitrage/api/status")
            SIGNALS=$(echo "$STATUS" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("signals_count",0))' 2>/dev/null || echo "?")
            ORDERS=$(echo "$STATUS" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("account",{}).get("total_order_attempts",0))' 2>/dev/null || echo "?")
            MARKETS=$(echo "$STATUS" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("markets_count",0))' 2>/dev/null || echo "?")

            echo ""
            echo "  Signals: $SIGNALS"
            echo "  Orders:  $ORDERS"
            echo "  Markets: $MARKETS"
        else
            log_warn "API not responding on port $PORT"
        fi
    else
        log_warn "Bot is not running"
    fi
}

# =============================================================================
# Action: Logs
# =============================================================================
show_logs() {
    log_info "Recent logs:"
    echo ""

    if [ -f "$LOG_FILE" ]; then
        tail -50 "$LOG_FILE"
    elif [ -f "/root/arbitrage_bot.log" ]; then
        tail -50 "/root/arbitrage_bot.log"
    else
        log_warn "No log file found"
    fi
}

# =============================================================================
# Action: Deploy
# =============================================================================
deploy_bot() {
    echo ""
    echo "=============================================="
    echo " Polymarket Arbitrage Bot - Deployment"
    echo "=============================================="
    echo ""

    # Step 1: Check Python
    log_info "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found. Installing..."
        apt-get update && apt-get install -y python3 python3-pip python3-venv
    fi
    PYTHON_VERSION=$(python3 --version)
    log_success "Python: $PYTHON_VERSION"

    # Step 2: Create/update virtual environment
    log_info "Setting up virtual environment..."
    VENV_DIR="$REPO_DIR/venv"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        log_success "Created virtual environment"
    else
        log_info "Virtual environment exists"
    fi

    # Step 3: Activate venv and install dependencies
    log_info "Installing dependencies..."
    source "$VENV_DIR/bin/activate"

    # Core dependencies for arbitrage bot
    pip install --quiet --upgrade pip
    pip install --quiet \
        fastapi>=0.100.0 \
        uvicorn>=0.20.0 \
        httpx>=0.24.0 \
        websockets>=11.0 \
        python-dotenv>=1.0.0

    log_success "Dependencies installed"

    # Step 4: Stop existing bot if running
    log_info "Checking for existing bot..."
    stop_bot 2>/dev/null || true

    # Step 5: Configure firewall
    log_info "Configuring firewall..."
    if command -v ufw &> /dev/null; then
        ufw allow "$PORT/tcp" 2>/dev/null || true
        log_success "Firewall: Port $PORT opened"
    fi

    # Step 6: Start the bot
    log_info "Starting arbitrage bot..."

    MODE_MSG="SIMULATION"
    if [ -n "$LIVE_MODE" ]; then
        MODE_MSG="*** LIVE TRADING ***"
    fi

    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"

    # Start bot with nohup
    cd "$SCRIPT_DIR"
    nohup "$VENV_DIR/bin/python" -u server.py \
        --host 0.0.0.0 \
        --port "$PORT" \
        --auto-start \
        $LIVE_MODE \
        >> "$LOG_FILE" 2>&1 &

    BOT_PID=$!
    echo "$BOT_PID" > "$PID_FILE"

    # Wait for startup
    sleep 3

    # Verify bot started
    if kill -0 "$BOT_PID" 2>/dev/null; then
        log_success "Bot started (PID: $BOT_PID)"
    else
        log_error "Bot failed to start. Check logs: $LOG_FILE"
        exit 1
    fi

    # Verify API
    sleep 2
    if curl -s --connect-timeout 5 "http://localhost:$PORT/arbitrage/api/status" > /dev/null 2>&1; then
        log_success "API responding on port $PORT"
    else
        log_warn "API not yet responding (may still be starting)"
    fi

    echo ""
    echo "=============================================="
    echo " Deployment Complete!"
    echo "=============================================="
    echo ""
    echo "  Mode:      $MODE_MSG"
    echo "  Port:      $PORT"
    echo "  PID:       $BOT_PID"
    echo "  Logs:      $LOG_FILE"
    echo ""
    echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):$PORT/arbitrage/"
    echo "  API:       http://$(hostname -I | awk '{print $1}'):$PORT/arbitrage/api/status"
    echo ""
    echo "  Commands:"
    echo "    ./deploy.sh --status   Check status"
    echo "    ./deploy.sh --logs     View logs"
    echo "    ./deploy.sh --stop     Stop bot"
    echo "    ./deploy.sh --restart  Restart bot"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
case $ACTION in
    deploy)
        deploy_bot
        ;;
    stop)
        stop_bot
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    restart)
        stop_bot
        sleep 2
        deploy_bot
        ;;
esac
