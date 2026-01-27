#!/bin/bash
# PolymarketBot Service Installation Script
# Run with: sudo ./scripts/install-service.sh

set -e

SERVICE_NAME="polymarketbot"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_DIR="/var/log/polymarketbot"

echo "=== PolymarketBot Service Installer ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./scripts/install-service.sh)"
    exit 1
fi

# Create log directory
echo "Creating log directory..."
mkdir -p "$LOG_DIR"
chown user:user "$LOG_DIR"

# Copy service file
echo "Installing systemd service..."
cp "$(dirname "$0")/../polymarketbot.service" "$SERVICE_FILE"

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service (start on boot)
echo "Enabling service..."
systemctl enable "$SERVICE_NAME"

# Start service
echo "Starting service..."
systemctl start "$SERVICE_NAME"

# Check status
echo ""
echo "=== Service Status ==="
systemctl status "$SERVICE_NAME" --no-pager || true

echo ""
echo "=== Installation Complete ==="
echo ""
echo "The scanner will now:"
echo "  - Start automatically on boot"
echo "  - Restart automatically if it crashes"
echo "  - Run in 'full' mode (both sybil and insider detection)"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status $SERVICE_NAME  # Check status"
echo "  sudo systemctl stop $SERVICE_NAME    # Stop service"
echo "  sudo systemctl start $SERVICE_NAME   # Start service"
echo "  sudo systemctl restart $SERVICE_NAME # Restart service"
echo "  sudo journalctl -u $SERVICE_NAME -f  # View live logs"
echo "  tail -f /var/log/polymarketbot/app.log  # View app logs"
echo ""
echo "Configuration:"
echo "  Edit $(dirname "$0")/../.env to change settings"
echo "  Set SCANNER_MODE=sybil or SCANNER_MODE=insider for single mode"
echo "  Set AUTO_START_SCANNER=false to disable auto-start"
