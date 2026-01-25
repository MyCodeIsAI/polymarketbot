#!/bin/bash
# PolymarketBot Services Installation Script
# Run with: sudo ./scripts/install-services.sh
# Installs both the dashboard and insider scanner services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/var/log/polymarketbot"

echo "=== PolymarketBot Services Installer ==="
echo "Repository: $REPO_DIR"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./scripts/install-services.sh)"
    exit 1
fi

# Create log directory
echo "Creating log directory..."
mkdir -p "$LOG_DIR"

# Create virtual environment if not exists
if [ ! -d "$REPO_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$REPO_DIR/venv"
    echo "Installing dependencies..."
    "$REPO_DIR/venv/bin/pip" install -r "$REPO_DIR/requirements.txt"
fi

# Check for .env file
if [ ! -f "$REPO_DIR/.env" ]; then
    echo "WARNING: .env file not found!"
    echo "Copy .env.example to .env and configure your API keys:"
    echo "  cp $REPO_DIR/.env.example $REPO_DIR/.env"
    echo "  nano $REPO_DIR/.env"
    echo ""
fi

# Install polymarketbot service (Dashboard)
echo ""
echo "=== Installing Dashboard Service ==="
SERVICE_FILE="/etc/systemd/system/polymarketbot.service"
if [ -f "$REPO_DIR/deploy/polymarketbot.service" ]; then
    cp "$REPO_DIR/deploy/polymarketbot.service" "$SERVICE_FILE"
    # Update paths to match actual repo location
    sed -i "s|/root/polymarketbot|$REPO_DIR|g" "$SERVICE_FILE"
    echo "Installed: polymarketbot.service"
else
    echo "WARNING: deploy/polymarketbot.service not found"
fi

# Install insider-scanner service
echo ""
echo "=== Installing Scanner Service ==="
SCANNER_SERVICE="/etc/systemd/system/insider-scanner.service"
if [ -f "$REPO_DIR/deploy/insider-scanner.service" ]; then
    cp "$REPO_DIR/deploy/insider-scanner.service" "$SCANNER_SERVICE"
    # Update paths to match actual repo location
    sed -i "s|/root/polymarketbot|$REPO_DIR|g" "$SCANNER_SERVICE"
    echo "Installed: insider-scanner.service"
else
    echo "WARNING: deploy/insider-scanner.service not found"
fi

# Reload systemd
echo ""
echo "Reloading systemd..."
systemctl daemon-reload

# Enable services
echo "Enabling services..."
systemctl enable polymarketbot 2>/dev/null || true
systemctl enable insider-scanner 2>/dev/null || true

# Start services
echo ""
echo "Starting services..."
systemctl start polymarketbot 2>/dev/null || echo "Failed to start polymarketbot"
systemctl start insider-scanner 2>/dev/null || echo "Failed to start insider-scanner"

# Wait for startup
sleep 3

# Check status
echo ""
echo "=== Service Status ==="
echo ""
echo "Dashboard (polymarketbot):"
systemctl is-active polymarketbot && echo "  Status: Running" || echo "  Status: Not running"

echo ""
echo "Scanner (insider-scanner):"
systemctl is-active insider-scanner && echo "  Status: Running" || echo "  Status: Not running"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Services installed:"
echo "  - polymarketbot (Dashboard on port 8765)"
echo "  - insider-scanner (Scanner API on port 8080)"
echo ""
echo "The scanner will auto-import sybil_defaults.json on first start."
echo ""
echo "Useful commands:"
echo "  systemctl status polymarketbot     # Dashboard status"
echo "  systemctl status insider-scanner   # Scanner status"
echo "  journalctl -u insider-scanner -f   # Scanner logs"
echo ""
echo "Web UI: http://localhost:8765/scanner"
echo ""
