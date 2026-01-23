# PolymarketBot VPS Deployment

Deploy PolymarketBot on a VPS with full security, auto-start, and optimizations.

---

## One-Command Install (Recommended)

Run this single command on a fresh Ubuntu/Debian VPS:

```bash
curl -fsSL https://raw.githubusercontent.com/MyCodeIsAI/polymarketbot/main/deploy/install-vps.sh | bash
```

**This automatically:**
- Installs Python and all dependencies
- Clones the repository to `/root/polymarketbot`
- Creates virtual environment and installs packages
- Applies TCP/network optimizations for low-latency trading
- Sets up file descriptor limits (fixes "too many files" error)
- Creates systemd service with auto-restart
- Enables auto-start on reboot
- Binds dashboard to localhost only (security)

After installation, access via SSH tunnel (see below).

---

## Security: SSH Tunnel Access

**CRITICAL:** The dashboard binds to `localhost:8765` only - it is NOT exposed to the internet.

### To Access the Dashboard:

**Step 1:** From your LOCAL machine, create an SSH tunnel:
```bash
ssh -L 8765:localhost:8765 root@YOUR_VPS_IP
```

**Step 2:** Open in your browser:
```
http://localhost:8765
```

### Why SSH Tunnel?
- Dashboard is only accessible through your encrypted SSH connection
- No one can access the UI without your VPS credentials
- All traffic is encrypted end-to-end
- Credentials stored on VPS are protected

**Never expose port 8765 directly to the internet!**

---

## Post-Install: Add Polygon RPC (Optional but Recommended)

For faster trade detection (~2-5s instead of ~15-21s):

```bash
nano /root/polymarketbot/.env
```

Add your Alchemy RPC URL:
```
POLYGON_RPC_URL=wss://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY
```

Then restart:
```bash
systemctl restart polymarketbot
```

Get a free API key at [dashboard.alchemy.com](https://dashboard.alchemy.com/signup).

---

## Service Management

```bash
# Check status
systemctl status polymarketbot

# View live logs
journalctl -u polymarketbot -f

# Restart service
systemctl restart polymarketbot

# Stop service
systemctl stop polymarketbot

# Start service
systemctl start polymarketbot
```

---

## Updating

```bash
systemctl stop polymarketbot
cd /root/polymarketbot
git pull
source venv/bin/activate
pip install -r requirements.txt
systemctl start polymarketbot
```

---

## Manual Installation

If you prefer manual setup instead of the one-command installer:

### 1. Install Dependencies
```bash
apt-get update && apt-get install -y python3 python3-venv python3-pip git
```

### 2. Clone Repository
```bash
cd /root
git clone https://github.com/MyCodeIsAI/polymarketbot.git
cd polymarketbot
```

### 3. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Create .env File
```bash
nano /root/polymarketbot/.env
```
Add:
```
POLYGON_RPC_URL=wss://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY
```

### 5. Create systemd Service
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
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
```

### 6. Enable and Start
```bash
systemctl daemon-reload
systemctl enable polymarketbot
systemctl start polymarketbot
```

### 7. Apply TCP Optimizations (Optional)
```bash
cat >> /etc/sysctl.conf << 'EOF'
# Polymarketbot network optimizations
fs.file-max = 2097152
net.core.somaxconn = 65535
net.ipv4.tcp_fin_timeout = 10
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_keepalive_time = 60
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
EOF
sysctl -p
```

---

## Troubleshooting

### Bot won't start
```bash
journalctl -u polymarketbot -n 50
```

### "Too many open files" error
The installer handles this automatically. For manual installs, ensure `LimitNOFILE=65535` is in the service file.

### Geo-blocking
Your VPS must be in an allowed region: **Amsterdam, Frankfurt, or Dublin** (NOT UK or US).

### Test manually
```bash
cd /root/polymarketbot
source venv/bin/activate
python run_ghost_mode.py
```

---

## Directory Structure

```
/root/polymarketbot/
├── .env                 # Environment variables (RPC URL)
├── venv/                # Python virtual environment
├── src/                 # Application source code
├── ghost_state.json     # Persistent state (auto-generated)
└── run_ghost_mode.py    # Main entry point
```
