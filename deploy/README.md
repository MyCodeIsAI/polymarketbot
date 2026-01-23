# PolymarketBot Deployment

This directory contains files for deploying PolymarketBot as a systemd service.

## Quick Start

1. **Run the install script (as root):**
   ```bash
   sudo ./install.sh
   ```

2. **Configure your credentials:**
   ```bash
   sudo nano /opt/polymarketbot/.env
   ```

3. **Configure target accounts:**
   ```bash
   sudo nano /opt/polymarketbot/config/accounts.yaml
   ```

4. **Validate configuration:**
   ```bash
   sudo -u polybot /opt/polymarketbot/venv/bin/python -m src.cli validate
   ```

5. **Test API connectivity:**
   ```bash
   sudo -u polybot /opt/polymarketbot/venv/bin/python -m src.cli test-connection
   ```

6. **Start the service:**
   ```bash
   sudo systemctl start polymarketbot
   sudo systemctl enable polymarketbot
   ```

## Files

| File | Description |
|------|-------------|
| `polymarketbot.service` | Main systemd service file |
| `polymarketbot-healthcheck.service` | Health check service |
| `polymarketbot-healthcheck.timer` | Timer for periodic health checks |
| `install.sh` | Installation script |

## Commands

### Service Management

```bash
# Start the bot
sudo systemctl start polymarketbot

# Stop the bot
sudo systemctl stop polymarketbot

# Restart the bot
sudo systemctl restart polymarketbot

# Check status
sudo systemctl status polymarketbot

# Enable on boot
sudo systemctl enable polymarketbot

# Disable on boot
sudo systemctl disable polymarketbot
```

### Logs

```bash
# View live logs
journalctl -u polymarketbot -f

# View recent logs
journalctl -u polymarketbot -n 100

# View logs since today
journalctl -u polymarketbot --since today

# View logs with errors only
journalctl -u polymarketbot -p err
```

### Health Checks

```bash
# Enable periodic health checks
sudo systemctl enable --now polymarketbot-healthcheck.timer

# Run manual health check
sudo systemctl start polymarketbot-healthcheck

# View health check status
systemctl list-timers polymarketbot-healthcheck.timer
```

## Directory Structure

After installation:

```
/opt/polymarketbot/
├── .env                 # Environment variables (secrets)
├── venv/                # Python virtual environment
├── src/                 # Application source code
├── config/              # Configuration files
│   ├── accounts.yaml
│   └── settings.yaml
├── data/                # Database and state files
│   └── polymarketbot.db
├── logs/                # Log files (if file logging enabled)
└── polybot.pid          # PID file when running
```

## Security Notes

- The `.env` file contains secrets and is readable only by the `polybot` user
- The service runs with security hardening (NoNewPrivileges, ProtectSystem, etc.)
- Resource limits prevent runaway memory/CPU usage
- The service runs as a non-privileged user

## Troubleshooting

### Bot won't start

1. Check configuration:
   ```bash
   sudo -u polybot /opt/polymarketbot/venv/bin/python -m src.cli validate
   ```

2. Check logs:
   ```bash
   journalctl -u polymarketbot -n 50
   ```

3. Test connectivity:
   ```bash
   sudo -u polybot /opt/polymarketbot/venv/bin/python -m src.cli test-connection
   ```

### VPN Issues

If you see geo-blocking errors:

1. Ensure VPN is connected
2. Verify VPN exit node is in allowed region (Netherlands, Germany, etc.)
3. Check that the VPN passes traffic for the polybot user

### Database Issues

If you see database errors:

1. Check file permissions:
   ```bash
   ls -la /opt/polymarketbot/data/
   ```

2. Try recreating the database:
   ```bash
   sudo -u polybot rm /opt/polymarketbot/data/polymarketbot.db
   sudo systemctl restart polymarketbot
   ```

## Updating

To update the bot:

```bash
# Stop the service
sudo systemctl stop polymarketbot

# Backup current installation
sudo cp -r /opt/polymarketbot /opt/polymarketbot.backup

# Copy new files
sudo cp -r ../src /opt/polymarketbot/

# Update dependencies
sudo -u polybot /opt/polymarketbot/venv/bin/pip install -r requirements.txt

# Start the service
sudo systemctl start polymarketbot
```
