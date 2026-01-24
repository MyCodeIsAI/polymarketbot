/**
 * PolymarketBot Dashboard Application
 */

class PolymarketDashboard {
    constructor() {
        this.ws = null;
        this.wsReconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.pnlChart = null;
        this.alerts = [];
        this.ghostModeEnabled = false;
        this.accounts = [];
        this.trades = [];
        this.accountsPnL = {};  // P/L data keyed by account ID

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadInitialData();
        this.initChart();
    }

    // ==========================================================================
    // WebSocket Management
    // ==========================================================================

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/live`;

        this.updateConnectionStatus('connecting');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.wsReconnectAttempts = 0;
                this.updateConnectionStatus('connected');

                // Subscribe to all channels
                this.ws.send(JSON.stringify({ command: 'subscribe', channel: 'positions' }));
                this.ws.send(JSON.stringify({ command: 'subscribe', channel: 'trades' }));
                this.ws.send(JSON.stringify({ command: 'subscribe', channel: 'alerts' }));
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.updateConnectionStatus('disconnected');
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.wsReconnectAttempts < this.maxReconnectAttempts) {
            this.wsReconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.wsReconnectAttempts - 1);
            console.log(`Reconnecting in ${delay}ms (attempt ${this.wsReconnectAttempts})`);
            setTimeout(() => this.connectWebSocket(), delay);
        } else {
            this.showToast('error', 'Connection lost. Please refresh the page.');
        }
    }

    handleWebSocketMessage(data) {
        const { type, timestamp } = data;

        this.updateLastUpdate(timestamp);

        switch (type) {
            case 'connected':
                console.log('WS: Connected to server');
                break;
            case 'initial_status':
            case 'status':
                this.updateStatus(data.data);
                break;
            case 'initial_positions':
            case 'positions':
                this.updatePositions(data.data.positions || data.data);
                break;
            case 'position_update':
                this.handlePositionUpdate(data.data);
                break;
            case 'trade':
                this.handleNewTrade(data.data);
                break;
            case 'alert':
                this.handleAlert(data.data);
                break;
            case 'pong':
                // Heartbeat response
                break;
            case 'subscribed':
                console.log(`Subscribed to ${data.channel}`);
                break;
            case 'error':
                console.error('WS Error:', data.message);
                this.showToast('error', data.message);
                break;
            case 'latency':
            case 'latency_update':
                this.updateLatency(data.data);
                break;
            case 'ghost_trade':
                this.handleGhostTrade(data.data);
                break;
            case 'ghost_mode_status':
            case 'mode_status':
                this.updateGhostModeStatus(data.data);
                break;
            case 'accounts_update':
                this.updateAccounts(data.data);
                break;
            case 'state_cleared':
                this.handleStateClear();
                break;
            case 'missed_trades':
                this.handleMissedTrades(data.data);
                break;
            default:
                console.log('Unknown message type:', type, data);
        }
    }

    handleGhostTrade(trade) {
        // Add to trades list at top
        this.trades.unshift(trade);
        if (this.trades.length > 100) this.trades.pop();

        // Refresh trades display
        this.updateTrades(this.trades.slice(0, 20));

        // Determine toast and alert based on status
        const marketName = trade.market_name || 'Unknown Market';
        const status = trade.status || 'detected';
        const latencyStr = trade.true_latency_ms ? `${(trade.true_latency_ms/1000).toFixed(1)}s` : '';

        let alertSeverity = 'info';
        let alertTitle = 'Trade Detected';
        let alertMessage = '';
        let showToast = true;

        if (status === 'api_simulated' || status === 'would_execute') {
            alertSeverity = 'success';
            alertTitle = 'Ghost Trade Executed';
            const orderType = trade.order_type === 'limit' ? '[LMT]' : '[MKT]';
            alertMessage = `${trade.side} ${orderType} $${(trade.our_size || 0).toFixed(2)} @ ${(trade.our_price || 0).toFixed(4)} - ${marketName.substring(0, 35)}`;
            this.showToast('success', `${trade.side} ${orderType} ${marketName.substring(0, 30)}... ${latencyStr ? `(${latencyStr})` : ''}`);
        } else if (status === 'filtered_keyword') {
            alertSeverity = 'info';
            alertTitle = 'Filtered (Keyword)';
            alertMessage = marketName.substring(0, 50);
            showToast = false; // Don't spam toasts for filtered trades
        } else if (status === 'filtered_stoploss') {
            alertSeverity = 'warning';
            alertTitle = 'Blocked (Stoploss)';
            alertMessage = marketName.substring(0, 50);
            this.showToast('warning', `Stoploss active: ${marketName.substring(0, 30)}...`);
        } else if (status === 'filtered_slippage') {
            alertSeverity = 'warning';
            alertTitle = 'Filtered (Slippage)';
            const slipPct = trade.actual_slippage_pct ? trade.actual_slippage_pct.toFixed(1) : '?';
            const maxPct = trade.max_allowed_slippage_pct ? trade.max_allowed_slippage_pct.toFixed(0) : '?';
            alertMessage = `${slipPct}% > ${maxPct}% max - ${marketName.substring(0, 30)}`;
            this.showToast('warning', `Slippage ${slipPct}%: ${marketName.substring(0, 25)}...`);
        } else if (status === 'filtered_limit') {
            alertSeverity = 'info';
            alertTitle = 'Filtered (Limit Order)';
            alertMessage = `Price moved - ${marketName.substring(0, 40)}`;
            showToast = false;
        }

        // Add to alerts list for "Recent Alerts" section
        this.handleAlert({
            severity: alertSeverity,
            title: alertTitle,
            message: alertMessage,
            timestamp: trade.timestamp || new Date().toISOString(),
        });
    }

    handleStateClear() {
        // Clear local state when mode switches
        this.trades = [];
        this.updateTrades([]);
        this.updatePositions([]);
        this.showToast('info', 'Trading state cleared');
    }

    handleMissedTrades(data) {
        if (!data || !data.count) return;

        this.showToast('warning', `${data.count} trades missed while offline`, 10000);

        // Show missed trades in alerts
        if (data.trades) {
            data.trades.slice(0, 5).forEach(trade => {
                this.handleAlert({
                    severity: 'warning',
                    title: 'Missed Trade',
                    message: `${trade.side} $${trade.size?.toFixed(2) || '?'} - ${trade.market_name?.substring(0, 30) || 'Unknown'}... (${trade.reason})`,
                    timestamp: new Date(trade.timestamp * 1000).toISOString(),
                });
            });
        }
    }

    updateGhostModeStatus(data) {
        this.ghostModeEnabled = data.enabled || data.ghost_mode;
        this.updateGhostModeUI();
    }

    updateGhostModeUI() {
        const toggle = document.getElementById('ghost-mode-toggle');
        const dot = document.getElementById('ghost-mode-dot');
        const badge = document.getElementById('ghost-mode-badge');
        const info = document.getElementById('ghost-mode-info');
        const panel = document.getElementById('control-panel');

        if (this.ghostModeEnabled) {
            toggle.classList.remove('bg-gray-600');
            toggle.classList.add('bg-purple-600');
            dot.classList.remove('translate-x-1');
            dot.classList.add('translate-x-6');
            badge.classList.remove('hidden');
            info.classList.remove('hidden');
            panel.classList.add('ghost-mode-active');
        } else {
            toggle.classList.add('bg-gray-600');
            toggle.classList.remove('bg-purple-600');
            dot.classList.add('translate-x-1');
            dot.classList.remove('translate-x-6');
            badge.classList.add('hidden');
            info.classList.add('hidden');
            panel.classList.remove('ghost-mode-active');
        }
    }

    updateConnectionStatus(status) {
        const el = document.getElementById('connection-status');
        el.className = 'px-2 py-1 text-xs rounded-full';

        switch (status) {
            case 'connected':
                el.textContent = 'Connected';
                el.classList.add('connection-connected');
                break;
            case 'disconnected':
                el.textContent = 'Disconnected';
                el.classList.add('connection-disconnected');
                break;
            case 'connecting':
                el.textContent = 'Connecting...';
                el.classList.add('connection-connecting');
                break;
        }
    }

    updateLastUpdate(timestamp) {
        const el = document.getElementById('last-update');
        if (timestamp) {
            const date = new Date(timestamp);
            el.textContent = `Updated: ${date.toLocaleTimeString()}`;
        }
    }

    // ==========================================================================
    // API Calls
    // ==========================================================================

    async apiCall(endpoint, method = 'GET', body = null) {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(`/api${endpoint}`, options);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        return response.json();
    }

    async loadInitialData() {
        try {
            // Load data in parallel
            const [status, positions, trades, accounts, health, pnl, stats, accountsPnL] = await Promise.all([
                this.apiCall('/status'),
                this.apiCall('/positions'),
                this.apiCall('/trades?limit=20'),
                this.apiCall('/accounts'),
                this.apiCall('/health'),
                this.apiCall('/pnl'),
                this.apiCall('/trades/stats'),
                this.apiCall('/accounts/pnl').catch(() => ({})),  // P/L tracking may not be enabled
            ]);

            this.updateStatus(status);
            this.updatePositions(positions);
            this.updateTrades(trades);
            this.updateAccountsPnL(accountsPnL);  // Update P/L before accounts so it's available
            this.updateAccounts(accounts);
            this.updateHealth(health);
            this.updatePnL(pnl);
            this.updateStats(stats);

            // Load P&L chart data
            this.loadPnLChartData();

            // Load latency metrics (may not exist)
            this.loadLatencyData();

            // Load blockchain status
            this.loadBlockchainStatus();
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showToast('error', 'Failed to load dashboard data');
        }
    }

    async loadLatencyData() {
        try {
            const latency = await this.apiCall('/latency');
            this.updateLatency(latency);
        } catch (error) {
            // Latency endpoint may not exist in all configurations
            console.log('Latency data not available:', error.message);
        }
    }

    // ==========================================================================
    // UI Updates
    // ==========================================================================

    updateStatus(data) {
        if (!data) return;

        // Bot status
        const statusEl = document.getElementById('bot-status');
        const indicatorEl = document.getElementById('bot-status-indicator');
        const uptimeEl = document.getElementById('bot-uptime');

        const status = data.status || data.bot_status || 'unknown';
        statusEl.textContent = this.capitalizeFirst(status);

        indicatorEl.className = 'w-3 h-3 rounded-full';
        if (status === 'running') {
            indicatorEl.classList.add('status-running');
        } else if (status === 'stopped') {
            indicatorEl.classList.add('status-stopped');
        } else if (status === 'paused') {
            indicatorEl.classList.add('status-paused');
        } else {
            indicatorEl.classList.add('bg-gray-500');
        }

        if (data.uptime) {
            uptimeEl.textContent = `Uptime: ${this.formatDuration(data.uptime)}`;
        }

        // Balance (use simulated_balance for ghost mode)
        const balance = data.balance ?? data.simulated_balance;
        if (balance !== undefined) {
            document.getElementById('balance-total').textContent = this.formatUSD(balance);
        }
        if (data.available_balance !== undefined) {
            document.getElementById('balance-available').textContent =
                `Available: ${this.formatUSD(data.available_balance)}`;
        }

        // Position count
        if (data.positions_count !== undefined) {
            document.getElementById('positions-count').textContent = data.positions_count;
        }
        if (data.positions_value !== undefined) {
            document.getElementById('positions-value').textContent =
                `Value: ${this.formatUSD(data.positions_value)}`;
        }

        // Ghost mode stats
        if (data.trades_detected !== undefined) {
            const el = document.getElementById('stats-total-trades');
            if (el) el.textContent = data.trades_detected;
        }
        if (data.trades_would_execute !== undefined) {
            const el = document.getElementById('stats-would-execute');
            if (el) el.textContent = data.trades_would_execute;
        }
        if (data.trades_filtered_keyword !== undefined) {
            const el = document.getElementById('stats-filtered-keyword');
            if (el) el.textContent = data.trades_filtered_keyword;
        }
        if (data.trades_filtered_stoploss !== undefined) {
            const el = document.getElementById('stats-filtered-stoploss');
            if (el) el.textContent = data.trades_filtered_stoploss;
        }
        if (data.trades_filtered_slippage !== undefined) {
            const el = document.getElementById('stats-filtered-slippage');
            if (el) el.textContent = data.trades_filtered_slippage;
        }
        if (data.trades_missed_offline !== undefined) {
            const el = document.getElementById('stats-missed-offline');
            if (el) el.textContent = data.trades_missed_offline;
        }
        if (data.avg_detection_ms !== undefined) {
            const el = document.getElementById('stats-latency');
            if (el) el.textContent = `${Math.round(data.avg_detection_ms)} ms`;
        }

        // Update ghost mode status
        if (data.ghost_mode !== undefined) {
            this.ghostModeEnabled = data.ghost_mode;
            this.updateGhostModeUI();
        }

        // Update blockchain panel from status data
        if (data.blockchain_enabled !== undefined) {
            this.updateBlockchainPanel({
                enabled: data.blockchain_enabled,
                trades_detected: data.blockchain_trades_detected || 0,
                avg_latency_ms: data.blockchain_avg_latency_ms || 0,
                last_block: data.blockchain_last_block,
                blocks_processed: data.blockchain_blocks_processed,
            });
        }

        // Update control buttons
        this.updateControlButtons(status);
    }

    updateControlButtons(status) {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const pauseBtn = document.getElementById('pause-btn');
        const resumeBtn = document.getElementById('resume-btn');

        startBtn.disabled = status === 'running' || status === 'paused';
        stopBtn.disabled = status === 'stopped';
        pauseBtn.disabled = status !== 'running';
        resumeBtn.disabled = status !== 'paused';
    }

    updatePositions(positions) {
        const tbody = document.getElementById('positions-table');

        if (!positions || positions.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="px-6 py-8 text-center text-gray-500">
                        No open positions
                    </td>
                </tr>
            `;
            document.getElementById('positions-count').textContent = '0';
            return;
        }

        document.getElementById('positions-count').textContent = positions.length;

        let totalValue = 0;
        const rows = positions.map(pos => {
            // Handle both field naming conventions (API vs expected)
            const size = parseFloat(pos.size || pos.our_size || 0);
            const avgPrice = parseFloat(pos.avg_price || pos.average_price || 0);
            const currentPrice = parseFloat(pos.current_price || pos.avg_price || avgPrice);  // Default to avg_price if no current
            const costBasis = parseFloat(pos.cost_basis || (size * avgPrice));

            // Calculate unrealized P&L: current value - cost basis
            const currentValue = size * currentPrice;
            const pnl = parseFloat(pos.unrealized_pnl || (currentValue - costBasis));
            const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            const statusClass = `position-${pos.status || 'open'}`;

            totalValue += currentValue;

            return `
                <tr class="hover:bg-gray-750">
                    <td class="px-6 py-4">
                        <div class="font-medium">${this.truncate(pos.market_name || 'Unknown', 30)}</div>
                    </td>
                    <td class="px-6 py-4">
                        <span class="${pos.outcome === 'Up' || pos.outcome === 'Yes' ? 'side-buy' : 'side-sell'}">${pos.outcome}</span>
                    </td>
                    <td class="px-6 py-4" title="${size.toFixed(2)} shares">${this.formatNumber(size)}</td>
                    <td class="px-6 py-4">${this.formatUSD(currentValue)}</td>
                    <td class="px-6 py-4">${this.formatPrice(avgPrice)}</td>
                    <td class="px-6 py-4">${this.formatPrice(currentPrice)}</td>
                    <td class="px-6 py-4 ${pnlClass}">${this.formatUSD(pnl)}</td>
                </tr>
            `;
        }).join('');

        tbody.innerHTML = rows;
        document.getElementById('positions-total-value').textContent = `Total: ${this.formatUSD(totalValue)}`;
        document.getElementById('positions-value').textContent = `Value: ${this.formatUSD(totalValue)}`;
    }

    handlePositionUpdate(data) {
        // Handle position update from WebSocket
        // data can be: { positions: [...], count, total_value } or a single position
        if (data.positions) {
            // Full positions list sent
            this.updatePositions(data.positions);
        } else if (Array.isArray(data)) {
            // Array of positions
            this.updatePositions(data);
        } else {
            // Single position update - refresh from API
            this.apiCall('/positions').then(positions => this.updatePositions(positions));
            if (data.market_name) {
                this.showToast('info', `Position updated: ${data.market_name}`);
            }
        }
    }

    updateTrades(trades) {
        this.trades = trades || [];
        const tbody = document.getElementById('trades-table');

        if (!trades || trades.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="px-6 py-8 text-center text-gray-500">
                        No recent trades
                    </td>
                </tr>
            `;
            return;
        }

        const rows = trades.map(trade => {
            const sideClass = trade.side === 'BUY' ? 'side-buy' : 'side-sell';
            const status = trade.status || 'filled';
            const statusClass = `trade-${status}`;
            const slippage = parseFloat(trade.slippage_percent || trade.actual_slippage_pct || 0);
            const slippageClass = slippage > 5 ? 'text-red-400' : slippage > 2 ? 'text-yellow-400' : 'text-gray-300';
            const marketName = trade.market_name || 'Unknown';
            const ghostBadge = trade.ghost_mode ? '<span class="text-purple-400 text-xs ml-1">👻</span>' : '';

            // Order type badge (market vs limit)
            const orderType = trade.order_type || 'market';
            const orderTypeBadge = orderType === 'limit'
                ? '<span class="text-yellow-400 text-xs ml-1">[LMT]</span>'
                : '<span class="text-cyan-400 text-xs ml-1">[MKT]</span>';

            // API response info for debugging
            const apiStatus = trade.api_response?.status || trade.api_error || '';
            const apiTitle = trade.order_params ? `Order: ${JSON.stringify(trade.order_params, null, 2)}` : '';

            // Human-readable status display
            const statusDisplayMap = {
                'live_executed': 'executed',
                'live_error': 'error',
                'live_error_balance': 'no balance',
                'live_error_rejected': 'rejected',
                'live_error_rate_limit': 'rate limited',
                'live_error_network': 'network error',
                'live_error_auth': 'auth error',
                'live_error_setup': 'setup error',
                'filtered_keyword': 'filtered',
                'filtered_stoploss': 'stoploss',
                'filtered_slippage': 'slippage',
                'would_execute': 'ghost',
                'api_simulated': 'simulated',
            };
            const statusDisplay = statusDisplayMap[status] || status;

            return `
                <tr class="hover:bg-gray-750" title="${apiTitle.replace(/"/g, '&quot;')}">
                    <td class="px-6 py-4 text-sm">${this.formatTime(trade.detected_at || trade.timestamp)}</td>
                    <td class="px-6 py-4">
                        <div class="font-medium text-sm">${trade.account_name || trade.account_id}${ghostBadge}</div>
                        <div class="text-xs text-gray-400 truncate max-w-xs" title="${marketName}">${marketName.substring(0, 35)}${marketName.length > 35 ? '...' : ''}</div>
                    </td>
                    <td class="px-6 py-4 ${sideClass}">${trade.side}${orderTypeBadge} ${trade.outcome || ''}</td>
                    <td class="px-6 py-4">${this.formatNumber(trade.our_size || trade.execution_size || trade.size || trade.target_size)}</td>
                    <td class="px-6 py-4">${this.formatPrice(trade.our_price || trade.execution_price || trade.price || trade.target_price)}</td>
                    <td class="px-6 py-4 ${slippageClass}">${slippage.toFixed(2)}%</td>
                    <td class="px-6 py-4">
                        <span class="px-2 py-1 text-xs rounded-full ${statusClass}" title="${status}: ${apiStatus}">${statusDisplay}</span>
                    </td>
                </tr>
            `;
        }).join('');

        tbody.innerHTML = rows;
    }

    handleNewTrade(trade) {
        // Add to trades list
        this.apiCall('/trades?limit=20').then(trades => this.updateTrades(trades));

        // Only show toast for live mode trades (ghost mode handled separately)
        if (!trade.ghost_mode) {
            const message = `${trade.side} ${trade.target_size} @ ${this.formatPrice(trade.execution_price)}`;
            this.showToast('success', `Trade executed: ${message}`);
        }
    }

    updateAccountsPnL(pnlData) {
        // Store P/L data keyed by account ID
        this.accountsPnL = {};
        if (pnlData && pnlData.accounts) {
            for (const [accountId, data] of Object.entries(pnlData.accounts)) {
                this.accountsPnL[accountId] = data;
            }
        }
    }

    formatPnL(value) {
        if (value === null || value === undefined) return '--';
        const num = parseFloat(value);
        const formatted = Math.abs(num).toFixed(2);
        if (num > 0) return `+$${formatted}`;
        if (num < 0) return `-$${formatted}`;
        return `$${formatted}`;
    }

    getPnLClass(value) {
        if (value === null || value === undefined) return 'text-gray-400';
        const num = parseFloat(value);
        if (num > 0) return 'text-green-400';
        if (num < 0) return 'text-red-400';
        return 'text-gray-400';
    }

    updateAccounts(accounts) {
        this.accounts = accounts || [];
        const container = document.getElementById('accounts-list');

        if (!accounts || accounts.length === 0) {
            container.innerHTML = `
                <div class="px-6 py-8 text-center text-gray-500">
                    No accounts configured
                </div>
            `;
            return;
        }

        const cards = accounts.map(account => {
            const enabledClass = account.enabled !== false ? 'account-enabled' : 'account-disabled';
            const fullWallet = account.target_wallet || account.wallet;
            const wallet = this.truncateWallet(fullWallet);
            const profileUrl = `https://polymarket.com/profile/${fullWallet}`;
            const keywords = account.keywords && account.keywords.length > 0
                ? account.keywords.join(', ')
                : 'All markets';
            const stoplossStatus = account.stoploss_triggered
                ? '<span class="text-red-400 text-xs">STOPLOSS</span>'
                : '';

            // Get P/L data for this account
            const pnl = this.accountsPnL[account.id] || {};
            const hasPnL = pnl.total_trades > 0;
            const totalPnL = hasPnL ? pnl.total_pnl : null;
            const winRate = hasPnL ? pnl.win_rate : null;
            const tradeCount = hasPnL ? pnl.total_trades : 0;

            // P/L display section
            const pnlSection = hasPnL ? `
                <div class="col-span-2 mt-1 pt-1 border-t border-gray-600">
                    <div class="flex justify-between items-center">
                        <span class="font-medium ${this.getPnLClass(totalPnL)}">${this.formatPnL(totalPnL)}</span>
                        <span class="text-gray-500">${tradeCount} trades${winRate !== null ? ` · ${(winRate * 100).toFixed(0)}% win` : ''}</span>
                    </div>
                </div>
            ` : '';

            return `
                <div class="account-card ${enabledClass} p-4 bg-gray-750 rounded-lg mb-2">
                    <div class="flex items-center justify-between mb-2">
                        <div>
                            <div class="font-medium flex items-center gap-2">
                                <a href="${profileUrl}" target="_blank" class="text-blue-400 hover:text-blue-300 hover:underline">
                                    ${account.name}
                                </a>
                                ${stoplossStatus}
                            </div>
                            <div class="wallet-address text-gray-400 text-xs">${wallet}</div>
                        </div>
                        <div class="flex gap-1">
                            <button
                                onclick="dashboard.openAccountModal(dashboard.accounts.find(a => a.id === ${account.id}))"
                                class="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 rounded"
                            >
                                Edit
                            </button>
                            <button
                                onclick="dashboard.deleteAccount(${account.id}, '${account.name}')"
                                class="px-2 py-1 text-xs bg-red-600/30 hover:bg-red-600 text-red-400 hover:text-white rounded"
                                title="Delete account"
                            >
                                ✕
                            </button>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-xs text-gray-400">
                        <div>Ratio: ${(parseFloat(account.position_ratio || 0.01) * 100).toFixed(1)}%</div>
                        <div>Max: ${this.formatUSD(account.max_position_usd || 500)}</div>
                        <div>Drawdown: ${account.max_drawdown_percent || 15}%</div>
                        <div>
                            <button
                                onclick="dashboard.toggleOrderType(${account.id})"
                                class="px-2 py-0.5 rounded text-xs font-medium ${(account.order_type || 'market') === 'market' ? 'bg-blue-600/30 text-blue-400' : 'bg-green-600/30 text-green-400'}"
                                title="${(account.order_type || 'market') === 'market' ? 'Market: Execute immediately at best price' : 'Limit: Only execute at same price or better'}"
                            >
                                ${(account.order_type || 'market') === 'market' ? 'MARKET' : 'LIMIT'}
                            </button>
                        </div>
                        <div class="col-span-2 truncate" title="${keywords}">Keywords: ${keywords.substring(0, 30)}${keywords.length > 30 ? '...' : ''}</div>
                        ${pnlSection}
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = cards;
    }

    updateHealth(health) {
        if (!health) return;

        // API health
        this.setHealthBadge('health-api', health.api_healthy);

        // WebSocket health
        this.setHealthBadge('health-ws', health.websocket_healthy);

        // Database health
        this.setHealthBadge('health-db', health.database_healthy);

        // Circuit breaker
        const cbEl = document.getElementById('health-cb');
        if (health.circuit_breaker_open) {
            cbEl.textContent = 'OPEN';
            cbEl.className = 'px-2 py-1 text-xs rounded-full health-unhealthy';
        } else {
            cbEl.textContent = 'Closed';
            cbEl.className = 'px-2 py-1 text-xs rounded-full health-healthy';
        }

        // Resource usage
        if (health.memory_percent !== undefined) {
            document.getElementById('health-memory').textContent = `${health.memory_percent.toFixed(1)}%`;
        }
        if (health.cpu_percent !== undefined) {
            document.getElementById('health-cpu').textContent = `${health.cpu_percent.toFixed(1)}%`;
        }
    }

    setHealthBadge(elementId, isHealthy) {
        const el = document.getElementById(elementId);
        if (isHealthy) {
            el.textContent = 'Healthy';
            el.className = 'px-2 py-1 text-xs rounded-full health-healthy';
        } else {
            el.textContent = 'Unhealthy';
            el.className = 'px-2 py-1 text-xs rounded-full health-unhealthy';
        }
    }

    updatePnL(pnl) {
        if (!pnl) return;

        const realized = parseFloat(pnl.realized_pnl || 0);
        const unrealized = parseFloat(pnl.unrealized_pnl || 0);
        const total = realized + unrealized;
        const roi = parseFloat(pnl.roi_percent || 0);

        const totalEl = document.getElementById('total-pnl');
        totalEl.textContent = this.formatUSD(total);
        totalEl.className = `text-2xl font-bold ${total >= 0 ? 'pnl-positive' : 'pnl-negative'}`;

        const roiEl = document.getElementById('pnl-roi');
        roiEl.textContent = `ROI: ${roi >= 0 ? '+' : ''}${roi.toFixed(2)}%`;
        roiEl.className = `text-sm mt-1 ${roi >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
    }

    updateStats(stats) {
        if (!stats) return;

        if (stats.total_trades !== undefined) {
            document.getElementById('stats-total-trades').textContent = stats.total_trades;
        }
        if (stats.would_execute !== undefined) {
            const el = document.getElementById('stats-would-execute');
            if (el) el.textContent = stats.would_execute;
        }
        if (stats.filtered_keyword !== undefined) {
            const el = document.getElementById('stats-filtered-keyword');
            if (el) el.textContent = stats.filtered_keyword;
        }
        if (stats.filtered_stoploss !== undefined) {
            const el = document.getElementById('stats-filtered-stoploss');
            if (el) el.textContent = stats.filtered_stoploss;
        }
        if (stats.win_rate !== undefined) {
            const el = document.getElementById('stats-win-rate');
            if (el) el.textContent = `${(stats.win_rate * 100).toFixed(1)}%`;
        }
        if (stats.avg_latency_ms !== undefined) {
            document.getElementById('stats-latency').textContent = `${Math.round(stats.avg_latency_ms)} ms`;
        }
        if (stats.avg_slippage !== undefined) {
            const el = document.getElementById('stats-slippage');
            if (el) el.textContent = `${(stats.avg_slippage * 100).toFixed(2)}%`;
        }
    }

    // ==========================================================================
    // Latency Monitoring (CRITICAL for copy trading)
    // ==========================================================================

    updateLatency(data) {
        if (!data) return;

        const stages = data.stages || data;
        const health = data.health || {};

        // Update health badge
        const healthEl = document.getElementById('latency-health');
        if (health.status === 'degraded' || health.health_score < 70) {
            healthEl.textContent = 'Degraded';
            healthEl.className = 'px-2 py-1 text-xs rounded-full bg-red-600';
        } else if (health.health_score < 90) {
            healthEl.textContent = 'Warning';
            healthEl.className = 'px-2 py-1 text-xs rounded-full bg-yellow-600';
        } else {
            healthEl.textContent = 'Healthy';
            healthEl.className = 'px-2 py-1 text-xs rounded-full bg-green-600';
        }

        // E2E latency (most important)
        const e2e = stages.e2e || stages.E2E || {};
        if (e2e.avg_ms !== undefined || e2e.recent_avg_ms !== undefined) {
            const avgMs = e2e.recent_avg_ms || e2e.avg_ms || 0;
            const p95Ms = e2e.p95_ms || avgMs;

            // Update display
            const e2eEl = document.getElementById('latency-e2e');
            e2eEl.textContent = `${avgMs.toFixed(0)} ms`;

            // Color based on threshold (target: <200ms)
            if (avgMs < 150) {
                e2eEl.className = 'text-lg font-bold text-green-400';
            } else if (avgMs < 300) {
                e2eEl.className = 'text-lg font-bold text-yellow-400';
            } else {
                e2eEl.className = 'text-lg font-bold text-red-400';
            }

            // Progress bar (0-500ms scale)
            const barWidth = Math.min(100, (avgMs / 500) * 100);
            const barEl = document.getElementById('latency-e2e-bar');
            barEl.style.width = `${barWidth}%`;
            if (avgMs < 150) {
                barEl.className = 'bg-green-500 h-2 rounded-full';
            } else if (avgMs < 300) {
                barEl.className = 'bg-yellow-500 h-2 rounded-full';
            } else {
                barEl.className = 'bg-red-500 h-2 rounded-full';
            }

            document.getElementById('latency-e2e-p95').textContent = `P95: ${p95Ms.toFixed(0)} ms`;

            // Min/Avg/Max
            if (e2e.min_ms !== undefined) {
                document.getElementById('latency-min').textContent = `${e2e.min_ms.toFixed(0)} ms`;
            }
            if (e2e.avg_ms !== undefined) {
                document.getElementById('latency-avg').textContent = `${e2e.avg_ms.toFixed(0)} ms`;
            }
            if (e2e.max_ms !== undefined) {
                document.getElementById('latency-max').textContent = `${e2e.max_ms.toFixed(0)} ms`;
            }
        }

        // Stage breakdown
        const stageMap = {
            'detection': 'latency-detection',
            'DETECTION': 'latency-detection',
            'validation': 'latency-validation',
            'VALIDATION': 'latency-validation',
            'signing': 'latency-signing',
            'SIGNING': 'latency-signing',
            'submission': 'latency-submission',
            'SUBMISSION': 'latency-submission',
        };

        for (const [stage, elementId] of Object.entries(stageMap)) {
            const stageData = stages[stage] || stages[stage.toLowerCase()];
            if (stageData && stageData.avg_ms !== undefined) {
                const el = document.getElementById(elementId);
                if (el) {
                    el.textContent = `${stageData.avg_ms.toFixed(0)} ms`;
                }
            }
        }
    }

    handleAlert(alert) {
        this.alerts.unshift(alert);
        if (this.alerts.length > 50) {
            this.alerts.pop();
        }
        this.renderAlerts();

        // Show toast for critical/warning alerts
        if (alert.severity === 'critical' || alert.severity === 'warning') {
            this.showToast(alert.severity === 'critical' ? 'error' : 'warning', alert.message);
        }
    }

    renderAlerts() {
        const container = document.getElementById('alerts-list');

        if (this.alerts.length === 0) {
            container.innerHTML = `
                <div class="px-6 py-8 text-center text-gray-500">
                    No recent alerts
                </div>
            `;
            return;
        }

        const alertHtml = this.alerts.slice(0, 10).map(alert => {
            const severityClass = `alert-${alert.severity || 'info'}`;
            return `
                <div class="px-4 py-3 ${severityClass}">
                    <div class="flex items-start justify-between">
                        <div class="font-medium text-sm">${alert.title || 'Alert'}</div>
                        <div class="text-xs text-gray-400">${this.formatTime(alert.timestamp)}</div>
                    </div>
                    <div class="text-sm text-gray-300 mt-1">${alert.message}</div>
                </div>
            `;
        }).join('');

        container.innerHTML = alertHtml;
    }

    // ==========================================================================
    // Chart
    // ==========================================================================

    initChart() {
        const ctx = document.getElementById('pnl-chart').getContext('2d');

        this.pnlChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cumulative P&L',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    },
                },
                scales: {
                    x: {
                        grid: {
                            color: '#374151',
                        },
                        ticks: {
                            color: '#9ca3af',
                        },
                    },
                    y: {
                        grid: {
                            color: '#374151',
                        },
                        ticks: {
                            color: '#9ca3af',
                            callback: (value) => `$${value}`,
                        },
                    },
                },
            },
        });
    }

    async loadPnLChartData() {
        try {
            const dailyPnL = await this.apiCall('/pnl/daily?days=30');

            if (dailyPnL && dailyPnL.length > 0) {
                const labels = dailyPnL.map(d => d.date);
                const data = dailyPnL.map(d => parseFloat(d.cumulative_pnl || 0));

                this.pnlChart.data.labels = labels;
                this.pnlChart.data.datasets[0].data = data;
                this.pnlChart.update();
            }
        } catch (error) {
            console.error('Failed to load P&L chart data:', error);
        }
    }

    // ==========================================================================
    // Event Handlers
    // ==========================================================================

    setupEventListeners() {
        // Control buttons
        document.getElementById('start-btn').addEventListener('click', () => this.controlBot('start'));
        document.getElementById('stop-btn').addEventListener('click', () => this.controlBot('stop'));
        document.getElementById('pause-btn').addEventListener('click', () => this.controlBot('pause'));
        document.getElementById('resume-btn').addEventListener('click', () => this.controlBot('resume'));

        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => this.loadInitialData());

        // Settings modal
        document.getElementById('settings-btn').addEventListener('click', () => this.openSettingsModal());
        document.getElementById('close-settings-modal').addEventListener('click', () => this.hideModal('settings-modal'));
        document.getElementById('save-rpc-btn').addEventListener('click', () => this.saveRpcUrl());
        document.getElementById('test-rpc-btn').addEventListener('click', () => this.testRpcConnection());

        // Ghost mode toggle
        document.getElementById('ghost-mode-toggle').addEventListener('click', () => this.toggleGhostMode());

        // Add account modal
        document.getElementById('add-account-btn').addEventListener('click', () => this.openAccountModal());
        document.getElementById('cancel-account-modal').addEventListener('click', () => this.hideModal('account-modal'));
        document.getElementById('account-form').addEventListener('submit', (e) => this.handleSaveAccount(e));

        // Username field - auto-lookup wallet on blur
        const nameInput = document.querySelector('#account-form input[name="name"]');
        if (nameInput) {
            nameInput.addEventListener('blur', (e) => this.lookupWalletFromUsername(e.target.value));
        }

        // Tiered slippage toggle
        const tieredCheckbox = document.querySelector('#account-form input[name="use_tiered_slippage"]');
        if (tieredCheckbox) {
            tieredCheckbox.addEventListener('change', (e) => {
                const tiersSection = document.getElementById('slippage-tiers');
                const flatSection = document.getElementById('flat-slippage-section');
                if (e.target.checked) {
                    tiersSection.classList.remove('hidden', 'opacity-50');
                    flatSection.classList.add('hidden');
                } else {
                    tiersSection.classList.add('opacity-50');
                    flatSection.classList.remove('hidden');
                }
            });
        }

        // Clear alerts
        document.getElementById('clear-alerts-btn').addEventListener('click', () => {
            this.alerts = [];
            this.renderAlerts();
        });

        // View all trades
        document.getElementById('show-all-trades').addEventListener('click', () => {
            this.apiCall('/trades?limit=100').then(trades => this.updateTrades(trades));
        });

        // Heartbeat
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ command: 'ping' }));
            }
        }, 30000);
    }

    async lookupWalletFromUsername(username) {
        if (!username || username.length < 2) return;

        // Check if wallet field is already filled
        const walletInput = document.querySelector('#account-form input[name="wallet"]');
        if (walletInput && walletInput.value && walletInput.value.startsWith('0x')) {
            return; // Already has a wallet address
        }

        try {
            // Show loading state
            if (walletInput) {
                walletInput.placeholder = 'Looking up wallet...';
                walletInput.classList.add('animate-pulse');
            }

            const result = await this.apiCall(`/lookup-wallet/${encodeURIComponent(username)}`);

            if (result.success && result.wallet) {
                if (walletInput) {
                    walletInput.value = result.wallet;
                    walletInput.classList.remove('animate-pulse');
                    walletInput.placeholder = '0x...';
                }
                this.showToast('success', `Found wallet for ${username}`);
            } else {
                if (walletInput) {
                    walletInput.classList.remove('animate-pulse');
                    walletInput.placeholder = 'Wallet not found - enter manually';
                }
                this.showToast('warning', `Could not find wallet for "${username}"`);
            }
        } catch (error) {
            console.error('Wallet lookup failed:', error);
            if (walletInput) {
                walletInput.classList.remove('animate-pulse');
                walletInput.placeholder = '0x...';
            }
        }
    }

    async toggleGhostMode() {
        try {
            // Toggle between ghost and live mode WITHOUT stopping the bot
            const result = await this.apiCall('/mode/toggle', 'POST');

            // Update state based on response
            this.ghostModeEnabled = result.ghost_mode;
            this.updateGhostModeUI();

            // Clear simulated data when switching to live mode
            if (result.live_mode) {
                this.trades = [];
                this.updateTrades([]);
                // Positions will be fetched from Polymarket API in live mode
            }

            const modeName = result.live_mode ? 'Live' : 'Ghost';
            this.showToast('success', `Switched to ${modeName} Mode`);
            this.loadInitialData();  // Refresh everything with new mode's data
        } catch (error) {
            this.showToast('error', `Failed to toggle mode: ${error.message}`);
        }
    }

    async deleteAccount(accountId, accountName) {
        if (!confirm(`Delete account "${accountName}"?\n\nThis will stop copy-trading this account.`)) {
            return;
        }

        try {
            const response = await fetch(`/api/accounts/${accountId}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete account');
            }

            this.showToast('success', `Account "${accountName}" deleted`);
            this.loadInitialData();  // Refresh accounts
        } catch (error) {
            this.showToast('error', `Failed to delete: ${error.message}`);
        }
    }

    async toggleOrderType(accountId) {
        const account = this.accounts.find(a => a.id === accountId);
        if (!account) return;

        const currentType = account.order_type || 'market';
        const newType = currentType === 'market' ? 'limit' : 'market';

        try {
            const response = await fetch(`/api/accounts/${accountId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ order_type: newType }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to update order type');
            }

            // Update local state
            account.order_type = newType;
            this.updateAccounts(this.accounts);

            const typeLabel = newType === 'market' ? 'Market (instant)' : 'Limit (best price)';
            this.showToast('success', `Order type changed to ${typeLabel}`);
        } catch (error) {
            this.showToast('error', `Failed to update: ${error.message}`);
        }
    }

    openAccountModal(account = null) {
        const modal = document.getElementById('account-modal');
        const title = document.getElementById('account-modal-title');
        const form = document.getElementById('account-form');
        const tiersSection = document.getElementById('slippage-tiers');
        const flatSection = document.getElementById('flat-slippage-section');

        // Default tier values
        const defaultTiers = { '5': 300, '10': 200, '20': 100, '35': 50, '50': 30, '70': 20, '85': 12, '100': 6 };

        if (account) {
            // Edit mode
            title.textContent = 'Edit Tracked Account';
            form.account_id.value = account.id;
            form.name.value = account.name;
            form.wallet.value = account.target_wallet || account.wallet;
            form.ratio.value = parseFloat(account.position_ratio) || 0.01;
            form.max_position.value = parseFloat(account.max_position_usd) || 500;
            form.keywords.value = (account.keywords || []).join(', ');
            form.max_drawdown.value = parseFloat(account.max_drawdown_percent) || 15;

            // Tiered slippage settings
            const useTiered = account.use_tiered_slippage !== false;
            form.use_tiered_slippage.checked = useTiered;

            // Populate tier values
            const tiers = account.slippage_tiers || defaultTiers;
            form.tier_5.value = tiers['5'] || defaultTiers['5'];
            form.tier_10.value = tiers['10'] || defaultTiers['10'];
            form.tier_20.value = tiers['20'] || defaultTiers['20'];
            form.tier_35.value = tiers['35'] || defaultTiers['35'];
            form.tier_50.value = tiers['50'] || defaultTiers['50'];
            form.tier_70.value = tiers['70'] || defaultTiers['70'];
            form.tier_85.value = tiers['85'] || defaultTiers['85'];
            form.tier_100.value = tiers['100'] || defaultTiers['100'];

            // Flat slippage
            form.flat_slippage.value = (parseFloat(account.flat_slippage_tolerance) || 0.05) * 100;

            // Advanced risk settings
            form.take_profit_pct.value = account.take_profit_pct || 0;
            form.stop_loss_pct.value = account.stop_loss_pct || 0;
            form.max_concurrent.value = account.max_concurrent || 0;
            form.max_holding_hours.value = account.max_holding_hours || 0;
            form.min_liquidity.value = account.min_liquidity || 0;
            form.cooldown_seconds.value = account.cooldown_seconds || 10;

            // Show/hide sections
            if (useTiered) {
                tiersSection.classList.remove('hidden', 'opacity-50');
                flatSection.classList.add('hidden');
            } else {
                tiersSection.classList.add('opacity-50');
                flatSection.classList.remove('hidden');
            }
        } else {
            // Add mode
            title.textContent = 'Add Tracked Account';
            form.reset();
            form.account_id.value = '';
            form.ratio.value = '0.01';
            form.max_position.value = '500';
            form.max_drawdown.value = '15';
            form.use_tiered_slippage.checked = true;

            // Set default tier values
            form.tier_5.value = defaultTiers['5'];
            form.tier_10.value = defaultTiers['10'];
            form.tier_20.value = defaultTiers['20'];
            form.tier_35.value = defaultTiers['35'];
            form.tier_50.value = defaultTiers['50'];
            form.tier_70.value = defaultTiers['70'];
            form.tier_85.value = defaultTiers['85'];
            form.tier_100.value = defaultTiers['100'];
            form.flat_slippage.value = '5';

            // Default advanced settings
            form.take_profit_pct.value = '0';
            form.stop_loss_pct.value = '0';
            form.max_concurrent.value = '0';
            form.max_holding_hours.value = '0';
            form.min_liquidity.value = '0';
            form.cooldown_seconds.value = '10';

            tiersSection.classList.remove('hidden', 'opacity-50');
            flatSection.classList.add('hidden');
        }

        modal.classList.remove('hidden');
        modal.classList.add('flex');
    }

    async handleSaveAccount(e) {
        e.preventDefault();

        const form = e.target;
        const accountId = form.account_id.value;
        const isEdit = !!accountId;

        // Build slippage tiers object
        const slippageTiers = {
            '5': parseFloat(form.tier_5.value),
            '10': parseFloat(form.tier_10.value),
            '20': parseFloat(form.tier_20.value),
            '35': parseFloat(form.tier_35.value),
            '50': parseFloat(form.tier_50.value),
            '70': parseFloat(form.tier_70.value),
            '85': parseFloat(form.tier_85.value),
            '100': parseFloat(form.tier_100.value),
        };

        const data = {
            name: form.name.value,
            wallet: form.wallet.value,
            position_ratio: parseFloat(form.ratio.value),
            max_position_usd: parseFloat(form.max_position.value),
            use_tiered_slippage: form.use_tiered_slippage.checked,
            slippage_tiers: slippageTiers,
            flat_slippage_tolerance: parseFloat(form.flat_slippage.value) / 100,
            keywords: form.keywords.value.split(',').map(k => k.trim()).filter(k => k),
            max_drawdown_percent: parseFloat(form.max_drawdown.value),
            // Advanced risk settings
            take_profit_pct: parseFloat(form.take_profit_pct.value) || 0,
            stop_loss_pct: parseFloat(form.stop_loss_pct.value) || 0,
            max_concurrent: parseInt(form.max_concurrent.value) || 0,
            max_holding_hours: parseInt(form.max_holding_hours.value) || 0,
            min_liquidity: parseFloat(form.min_liquidity.value) || 0,
            cooldown_seconds: parseInt(form.cooldown_seconds.value) || 10,
        };

        try {
            if (isEdit) {
                await this.apiCall(`/accounts/${accountId}`, 'PUT', data);
                this.showToast('success', `Account "${data.name}" updated`);
            } else {
                await this.apiCall('/accounts', 'POST', data);
                this.showToast('success', `Account "${data.name}" added`);
            }

            this.hideModal('account-modal');

            // Refresh accounts
            const accounts = await this.apiCall('/accounts');
            this.updateAccounts(accounts);
        } catch (error) {
            this.showToast('error', `Failed to ${isEdit ? 'update' : 'add'} account: ${error.message}`);
        }
    }

    async controlBot(action) {
        try {
            const btn = document.getElementById(`${action}-btn`);
            btn.classList.add('btn-loading');
            btn.disabled = true;

            await this.apiCall(`/control/${action}`, 'POST');
            this.showToast('success', `Bot ${action} command sent`);

            // Refresh status
            setTimeout(() => this.loadInitialData(), 1000);
        } catch (error) {
            this.showToast('error', `Failed to ${action} bot: ${error.message}`);
        } finally {
            const btn = document.getElementById(`${action}-btn`);
            btn.classList.remove('btn-loading');
        }
    }

    showModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    }

    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }

    // ==========================================================================
    // Settings & Blockchain Monitoring
    // ==========================================================================

    async openSettingsModal() {
        this.showModal('settings-modal');
        await this.loadBlockchainStatus();
    }

    async loadBlockchainStatus() {
        try {
            const status = await this.apiCall('/blockchain/status');
            this.updateBlockchainSettings(status);
            this.updateBlockchainPanel(status);
        } catch (error) {
            console.log('Blockchain status not available:', error.message);
            // Set defaults when endpoint not available
            this.updateBlockchainSettings({
                available: false,
                enabled: false,
                rpc_configured: false,
            });
        }
    }

    updateBlockchainSettings(status) {
        // Config status badge
        const configStatus = document.getElementById('blockchain-config-status');
        const enabledBadge = document.getElementById('blockchain-enabled-badge');

        if (status.enabled) {
            configStatus.textContent = 'Connected';
            configStatus.className = 'px-2 py-1 text-xs rounded-full bg-green-600 text-white';
            enabledBadge.classList.remove('hidden');
            enabledBadge.className = 'px-2 py-1 text-xs rounded-full bg-green-600 text-white';
            enabledBadge.textContent = 'Active';
        } else if (status.rpc_configured) {
            configStatus.textContent = 'Configured';
            configStatus.className = 'px-2 py-1 text-xs rounded-full bg-yellow-600 text-white';
            enabledBadge.classList.add('hidden');
        } else {
            configStatus.textContent = 'Not Configured';
            configStatus.className = 'px-2 py-1 text-xs rounded-full bg-gray-600';
            enabledBadge.classList.add('hidden');
        }

        // Current configuration section
        const moduleEl = document.getElementById('config-blockchain-module');
        const rpcEl = document.getElementById('config-rpc-status');
        const activeEl = document.getElementById('config-monitor-active');
        const walletsEl = document.getElementById('config-wallets-count');

        if (moduleEl) {
            moduleEl.textContent = status.available ? 'Available' : 'Not Installed';
            moduleEl.className = status.available ? 'font-medium text-green-400' : 'font-medium text-gray-500';
        }
        if (rpcEl) {
            rpcEl.textContent = status.rpc_configured ? 'Yes' : 'No';
            rpcEl.className = status.rpc_configured ? 'font-medium text-green-400' : 'font-medium text-gray-500';
        }
        if (activeEl) {
            activeEl.textContent = status.enabled ? 'Yes' : 'No';
            activeEl.className = status.enabled ? 'font-medium text-green-400' : 'font-medium text-gray-500';
        }
        if (walletsEl) {
            walletsEl.textContent = status.wallets_monitored || '0';
        }

        // Health section blockchain badge
        const healthBlockchain = document.getElementById('health-blockchain');
        if (healthBlockchain) {
            if (status.enabled) {
                healthBlockchain.textContent = `~${Math.round(status.avg_latency_ms || 3000)}ms`;
                healthBlockchain.className = 'px-2 py-1 text-xs rounded-full health-healthy';
            } else if (status.rpc_configured) {
                healthBlockchain.textContent = 'Standby';
                healthBlockchain.className = 'px-2 py-1 text-xs rounded-full bg-yellow-600 text-white';
            } else {
                healthBlockchain.textContent = 'Off';
                healthBlockchain.className = 'px-2 py-1 text-xs rounded-full bg-gray-600';
            }
        }
    }

    updateBlockchainPanel(status) {
        const panel = document.getElementById('blockchain-stats-panel');
        if (!panel) return;

        if (status.enabled) {
            panel.classList.remove('hidden');

            const statusEl = document.getElementById('blockchain-status');
            const lastBlockEl = document.getElementById('blockchain-last-block');
            const blocksEl = document.getElementById('blockchain-blocks-processed');
            const tradesEl = document.getElementById('blockchain-trades-detected');
            const latencyEl = document.getElementById('blockchain-avg-latency');
            const latencyBadge = document.getElementById('blockchain-latency-badge');

            if (statusEl) {
                statusEl.textContent = 'Connected';
                statusEl.className = 'font-medium text-green-400';
            }
            if (lastBlockEl) lastBlockEl.textContent = status.last_block?.toLocaleString() || '--';
            if (blocksEl) blocksEl.textContent = status.blocks_processed?.toLocaleString() || '--';
            if (tradesEl) tradesEl.textContent = status.trades_detected || '0';
            if (latencyEl) {
                const latency = status.avg_latency_ms || 0;
                latencyEl.textContent = `${Math.round(latency)} ms`;
                latencyEl.className = latency < 5000 ? 'font-mono text-sm text-green-400' : 'font-mono text-sm text-yellow-400';
            }
            if (latencyBadge) {
                const latency = status.avg_latency_ms || 3000;
                latencyBadge.textContent = `~${(latency / 1000).toFixed(1)}s`;
            }
        } else {
            panel.classList.add('hidden');
        }
    }

    async saveRpcUrl() {
        const input = document.getElementById('rpc-url-input');
        const rpcUrl = input.value.trim();

        if (!rpcUrl) {
            this.showToast('error', 'Please enter an RPC URL');
            return;
        }

        // Validate URL format
        if (!rpcUrl.startsWith('wss://') && !rpcUrl.startsWith('https://') && !rpcUrl.startsWith('http://')) {
            this.showToast('error', 'RPC URL must start with wss://, https://, or http://');
            return;
        }

        const btn = document.getElementById('save-rpc-btn');
        btn.textContent = 'Connecting...';
        btn.disabled = true;

        try {
            const result = await this.apiCall('/blockchain/configure', 'POST', { rpc_url: rpcUrl });

            if (result.success) {
                this.showToast('success', 'Blockchain monitoring configured and connected!');
                await this.loadBlockchainStatus();
                await this.loadInitialData();
            } else {
                this.showToast('error', result.error || 'Failed to connect');
            }
        } catch (error) {
            this.showToast('error', `Failed to save: ${error.message}`);
        } finally {
            btn.textContent = 'Save & Connect';
            btn.disabled = false;
        }
    }

    async testRpcConnection() {
        const input = document.getElementById('rpc-url-input');
        const rpcUrl = input.value.trim();
        const resultEl = document.getElementById('rpc-test-result');

        if (!rpcUrl) {
            resultEl.textContent = 'Enter a URL first';
            resultEl.className = 'text-sm text-yellow-400';
            return;
        }

        const btn = document.getElementById('test-rpc-btn');
        btn.textContent = 'Testing...';
        btn.disabled = true;
        resultEl.textContent = '';

        try {
            const result = await this.apiCall('/blockchain/test', 'POST', { rpc_url: rpcUrl });

            if (result.success) {
                resultEl.textContent = `Connected! Block: ${result.block_number?.toLocaleString() || 'OK'}`;
                resultEl.className = 'text-sm text-green-400';
            } else {
                resultEl.textContent = result.error || 'Connection failed';
                resultEl.className = 'text-sm text-red-400';
            }
        } catch (error) {
            resultEl.textContent = error.message;
            resultEl.className = 'text-sm text-red-400';
        } finally {
            btn.textContent = 'Test Connection';
            btn.disabled = false;
        }
    }

    // ==========================================================================
    // Toast Notifications
    // ==========================================================================

    showToast(type, message, duration = 5000) {
        const container = document.getElementById('toast-container');

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'fadeOut 0.3s ease-out forwards';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // ==========================================================================
    // Utility Functions
    // ==========================================================================

    formatUSD(value) {
        if (value === null || value === undefined) return '$--';
        const num = parseFloat(value);
        return `$${num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }

    formatPrice(value) {
        if (value === null || value === undefined) return '--';
        return parseFloat(value).toFixed(4);
    }

    formatNumber(value) {
        if (value === null || value === undefined) return '--';
        return parseFloat(value).toLocaleString('en-US', { maximumFractionDigits: 4 });
    }

    formatTime(timestamp) {
        if (!timestamp) return '--';
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    }

    formatDuration(seconds) {
        if (!seconds) return '--';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }
        return `${minutes}m`;
    }

    capitalizeFirst(str) {
        if (!str) return '';
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    truncate(str, maxLength) {
        if (!str) return '';
        return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
    }

    truncateWallet(wallet) {
        if (!wallet || wallet.length < 12) return wallet || '';
        return `${wallet.substring(0, 6)}...${wallet.substring(wallet.length - 6)}`;
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new PolymarketDashboard();
});
