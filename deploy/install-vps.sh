#!/bin/bash
#
# PolymarketBot VPS Auto-Installer
# ================================
# One-command deployment for fresh VPS (Ubuntu/Debian)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/MyCodeIsAI/polymarketbot/main/deploy/install-vps.sh | bash
#
# Or clone and run:
#   git clone https://github.com/MyCodeIsAI/polymarketbot.git
#   cd polymarketbot && chmod +x deploy/install-vps.sh && ./deploy/install-vps.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "========================================================================"
echo "  PolymarketBot VPS Auto-Installer"
echo "========================================================================"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run as root (sudo)${NC}"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo -e "${RED}Error: Cannot detect OS${NC}"
    exit 1
fi

echo -e "${GREEN}[1/8]${NC} Detected OS: $OS"

# ============================================================================
# STEP 1: Install system dependencies
# ============================================================================
echo -e "\n${GREEN}[2/8]${NC} Installing system dependencies..."

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    apt-get update -qq
    apt-get install -y -qq python3 python3-venv python3-pip git curl
elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "fedora" ]; then
    yum install -y python3 python3-pip git curl
else
    echo -e "${YELLOW}Warning: Unknown OS, assuming Python3 is installed${NC}"
fi

# ============================================================================
# STEP 2: Clone or update repository
# ============================================================================
echo -e "\n${GREEN}[3/8]${NC} Setting up repository..."

INSTALL_DIR="/root/polymarketbot"

if [ -d "$INSTALL_DIR" ]; then
    echo "  Repository exists, pulling latest changes..."
    cd "$INSTALL_DIR"
    git pull --quiet
else
    echo "  Cloning repository..."
    git clone --quiet https://github.com/MyCodeIsAI/polymarketbot.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# ============================================================================
# STEP 3: Create virtual environment and install dependencies
# ============================================================================
echo -e "\n${GREEN}[4/8]${NC} Setting up Python environment..."

if [ ! -d "$INSTALL_DIR/venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ============================================================================
# STEP 4: Create .env file if it doesn't exist
# ============================================================================
echo -e "\n${GREEN}[5/8]${NC} Checking environment configuration..."

if [ ! -f "$INSTALL_DIR/.env" ]; then
    echo "  Creating .env file (you'll need to add your RPC URL)..."
    cat > "$INSTALL_DIR/.env" << 'EOF'
# PolymarketBot Environment Configuration
# ========================================
# Add your Polygon RPC URL for blockchain monitoring (~2-5s latency)
# Get a free one at: https://dashboard.alchemy.com
#
# POLYGON_RPC_URL=wss://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY
EOF
    echo -e "  ${YELLOW}Note: Edit .env to add your Polygon RPC URL for faster trade detection${NC}"
else
    echo "  .env file already exists"
fi

# ============================================================================
# STEP 5: Apply system optimizations
# ============================================================================
echo -e "\n${GREEN}[6/8]${NC} Applying system optimizations..."

# File descriptor limits
if ! grep -q "polymarketbot" /etc/security/limits.conf 2>/dev/null; then
    cat >> /etc/security/limits.conf << 'EOF'

# Polymarketbot file descriptor limits
root soft nofile 65535
root hard nofile 65535
* soft nofile 65535
* hard nofile 65535
EOF
    echo "  Added file descriptor limits"
fi

# TCP/kernel optimizations for low-latency trading
if ! grep -q "polymarketbot" /etc/sysctl.conf 2>/dev/null; then
    cat >> /etc/sysctl.conf << 'EOF'

# Polymarketbot network optimizations for low-latency trading
fs.file-max = 2097152
fs.nr_open = 2097152
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6
net.ipv4.tcp_max_syn_backlog = 65535
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
EOF
    sysctl -p > /dev/null 2>&1
    echo "  Applied TCP optimizations"
fi

# ============================================================================
# STEP 6: Create systemd service
# ============================================================================
echo -e "\n${GREEN}[7/8]${NC} Creating systemd service..."

cat > /etc/systemd/system/polymarketbot.service << 'EOF'
[Unit]
Description=PolymarketBot Copy Trading Dashboard
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/polymarketbot
Environment=PORT=8765
ExecStart=/root/polymarketbot/venv/bin/python run_ghost_mode.py
Restart=always
RestartSec=5
LimitNOFILE=65535

# Security hardening
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable polymarketbot --quiet

# ============================================================================
# STEP 7: Start the service
# ============================================================================
echo -e "\n${GREEN}[8/8]${NC} Starting PolymarketBot service..."

systemctl restart polymarketbot
sleep 2

# Check if service is running
if systemctl is-active --quiet polymarketbot; then
    echo -e "  ${GREEN}Service is running!${NC}"
else
    echo -e "  ${RED}Service failed to start. Check logs: journalctl -u polymarketbot -n 50${NC}"
    exit 1
fi

# ============================================================================
# DONE - Print success message and instructions
# ============================================================================
VPS_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VPS_IP")

echo -e "\n${GREEN}"
echo "========================================================================"
echo "  Installation Complete!"
echo "========================================================================"
echo -e "${NC}"
echo -e "  ${GREEN}Status:${NC} PolymarketBot is running and will auto-start on reboot"
echo ""
echo -e "  ${BLUE}SECURITY:${NC} Dashboard binds to localhost only (not exposed to internet)"
echo ""
echo -e "  ${YELLOW}To access the dashboard:${NC}"
echo ""
echo "  1. From your LOCAL computer, create an SSH tunnel:"
echo -e "     ${GREEN}ssh -L 8765:localhost:8765 root@${VPS_IP}${NC}"
echo ""
echo "  2. Then open in your browser:"
echo -e "     ${GREEN}http://localhost:8765${NC}"
echo ""
echo -e "  ${YELLOW}Optional: Add Polygon RPC for faster trade detection:${NC}"
echo "     nano /root/polymarketbot/.env"
echo "     Add: POLYGON_RPC_URL=wss://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY"
echo "     Then: systemctl restart polymarketbot"
echo ""
echo -e "  ${BLUE}Useful commands:${NC}"
echo "     systemctl status polymarketbot   # Check status"
echo "     journalctl -u polymarketbot -f   # View live logs"
echo "     systemctl restart polymarketbot  # Restart service"
echo ""
echo "========================================================================"
