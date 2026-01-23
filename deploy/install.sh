#!/bin/bash
# PolymarketBot Installation Script
# Run as root or with sudo

set -e

# Configuration
INSTALL_DIR="/opt/polymarketbot"
BOT_USER="polybot"
BOT_GROUP="polybot"
PYTHON_VERSION="3.11"

echo "=== PolymarketBot Installation ==="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root or with sudo"
   exit 1
fi

# Create user and group
echo "Creating service user..."
if ! id "$BOT_USER" &>/dev/null; then
    useradd --system --no-create-home --shell /bin/false "$BOT_USER"
    echo "Created user: $BOT_USER"
else
    echo "User $BOT_USER already exists"
fi

# Create installation directory
echo "Creating installation directory..."
mkdir -p "$INSTALL_DIR"/{data,logs,config}
chown -R "$BOT_USER:$BOT_GROUP" "$INSTALL_DIR"

# Install Python if needed
echo "Checking Python installation..."
if ! command -v python$PYTHON_VERSION &> /dev/null; then
    echo "Installing Python $PYTHON_VERSION..."
    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y python$PYTHON_VERSION python$PYTHON_VERSION-venv python$PYTHON_VERSION-dev
    elif command -v dnf &> /dev/null; then
        dnf install -y python$PYTHON_VERSION python$PYTHON_VERSION-devel
    else
        echo "Please install Python $PYTHON_VERSION manually"
        exit 1
    fi
fi

# Copy application files
echo "Copying application files..."
if [[ -d "../src" ]]; then
    cp -r ../src "$INSTALL_DIR/"
    cp -r ../config "$INSTALL_DIR/"
    cp ../requirements.txt "$INSTALL_DIR/"
    cp ../pyproject.toml "$INSTALL_DIR/"
fi

# Create virtual environment
echo "Creating virtual environment..."
python$PYTHON_VERSION -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "$INSTALL_DIR/requirements.txt"

# Create .env template
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    echo "Creating .env template..."
    cat > "$INSTALL_DIR/.env" << 'EOF'
# PolymarketBot Environment Variables
# Copy this file and fill in your values

# Your Polymarket private key (KEEP SECRET!)
POLY_PRIVATE_KEY=

# API credentials (from CLOB API)
POLY_API_KEY=
POLY_API_SECRET=
POLY_PASSPHRASE=

# Optional: Polygon RPC URL
POLYGON_RPC_URL=https://polygon-rpc.com

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
EOF
    chmod 600 "$INSTALL_DIR/.env"
    chown "$BOT_USER:$BOT_GROUP" "$INSTALL_DIR/.env"
fi

# Install systemd service
echo "Installing systemd service..."
cp polymarketbot.service /etc/systemd/system/
systemctl daemon-reload

# Set permissions
echo "Setting permissions..."
chown -R "$BOT_USER:$BOT_GROUP" "$INSTALL_DIR"
chmod 750 "$INSTALL_DIR"
chmod 640 "$INSTALL_DIR/.env"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit the configuration: sudo nano $INSTALL_DIR/.env"
echo "2. Configure accounts: sudo nano $INSTALL_DIR/config/accounts.yaml"
echo "3. Test configuration: sudo -u $BOT_USER $INSTALL_DIR/venv/bin/python -m src.cli validate"
echo "4. Test connectivity: sudo -u $BOT_USER $INSTALL_DIR/venv/bin/python -m src.cli test-connection"
echo "5. Start the service: sudo systemctl start polymarketbot"
echo "6. Enable on boot: sudo systemctl enable polymarketbot"
echo ""
echo "View logs: journalctl -u polymarketbot -f"
echo ""
