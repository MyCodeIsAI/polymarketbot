# PolymarketBot VPS Deployment

This guide covers deploying PolymarketBot on a VPS (root account).

## Quick Start

### 1. Clone the Repository

```bash
cd /root
git clone https://github.com/yourusername/polymarketbot.git
cd polymarketbot
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Create Environment File

```bash
nano /root/polymarketbot/.env
```

Add your Polygon RPC URL (required for blockchain monitoring):
```
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY
```

### 4. Create systemd Service

```bash
cat > /etc/systemd/system/polymarketbot.service << 'EOF'
[Unit]
Description=PolymarketBot Copy Trading Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/polymarketbot
Environment=PORT=8765
ExecStart=/root/polymarketbot/venv/bin/python run_ghost_mode.py
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF
```

### 5. Start the Service

```bash
systemctl daemon-reload
systemctl enable polymarketbot
systemctl start polymarketbot

# Verify it's running
systemctl status polymarketbot
```

## Service Management

```bash
# Start the bot
systemctl start polymarketbot

# Stop the bot
systemctl stop polymarketbot

# Restart the bot
systemctl restart polymarketbot

# Check status
systemctl status polymarketbot

# View live logs
journalctl -u polymarketbot -f

# View recent logs
journalctl -u polymarketbot -n 100
```

## Directory Structure

```
/root/polymarketbot/
├── .env                 # Environment variables (RPC URL, etc.)
├── venv/                # Python virtual environment
├── src/                 # Application source code
├── ghost_state.json     # Persistent state (auto-generated)
└── run_ghost_mode.py    # Main entry point
```

## Security - SSH Tunnel Access

**CRITICAL:** The dashboard binds to `localhost:8765` and is NOT exposed to the internet.

To access the dashboard securely:

```bash
# From your LOCAL machine, create an SSH tunnel:
ssh -L 8765:localhost:8765 root@YOUR_VPS_IP

# Then open in your browser:
# http://localhost:8765
```

This ensures:
- Dashboard is only accessible through your encrypted SSH connection
- No one can access the UI without your VPS credentials
- All traffic is encrypted end-to-end

**Never expose port 8765 directly to the internet!**

## Troubleshooting

### Bot won't start

1. Check logs:
   ```bash
   journalctl -u polymarketbot -n 50
   ```

2. Verify .env file exists with RPC URL:
   ```bash
   cat /root/polymarketbot/.env
   ```

3. Test manually:
   ```bash
   cd /root/polymarketbot
   source venv/bin/activate
   python run_ghost_mode.py
   ```

### "Too many open files" error

Add file descriptor limits:
```bash
# In /etc/systemd/system/polymarketbot.service, ensure:
LimitNOFILE=65535

# Then reload:
systemctl daemon-reload
systemctl restart polymarketbot
```

### Geo-blocking Issues

Ensure your VPS is in an allowed region (Amsterdam, Frankfurt, Dublin - NOT UK/US).

## Updating

```bash
# Stop the service
systemctl stop polymarketbot

# Pull latest code
cd /root/polymarketbot
git pull

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Start the service
systemctl start polymarketbot
```

## Performance Tuning (Optional)

For optimal latency, add TCP optimizations:

```bash
cat >> /etc/sysctl.conf << 'EOF'
# Polymarketbot optimizations
fs.file-max = 2097152
net.core.somaxconn = 65535
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_low_latency = 1
EOF

sysctl -p
```
