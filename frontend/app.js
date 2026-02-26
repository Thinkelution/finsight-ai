// FinSight AI Dashboard — Frontend Application

const API = '';

// --- Tab Navigation ---
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.content').forEach(c => c.classList.add('hidden'));
        const tab = document.getElementById(`tab-${btn.dataset.tab}`);
        if (tab) tab.classList.remove('hidden');
    });
});

// --- Health & Stats ---
async function loadHealth() {
    try {
        const res = await fetch(`${API}/health`);
        const data = await res.json();

        const dot = document.querySelector('.status-dot');
        const statusText = document.querySelector('#header-status span:last-child');

        if (data.status === 'healthy') {
            dot.className = 'status-dot online';
            statusText.textContent = 'All Systems Online';
        } else {
            dot.className = 'status-dot';
            statusText.textContent = data.status === 'degraded' ? 'Degraded' : 'Offline';
        }

        document.getElementById('stat-chunks').textContent = data.chunks_count ?? '0';
        document.getElementById('stat-health').textContent =
            data.status === 'healthy' ? 'Healthy' : data.status;

    } catch (e) {
        const dot = document.querySelector('.status-dot');
        dot.className = 'status-dot error';
        document.querySelector('#header-status span:last-child').textContent = 'API Offline';
        document.getElementById('stat-health').textContent = 'Offline';
    }
}

async function loadStats() {
    try {
        const res = await fetch(`${API}/data/stats`);
        const data = await res.json();

        document.getElementById('stat-chunks').textContent = data.total_chunks ?? '0';
        document.getElementById('stat-model').textContent = data.llm_model || '—';
    } catch (e) {
        console.error('Stats load failed:', e);
    }
}

// --- Market Data ---
async function loadMarketData() {
    const grid = document.getElementById('market-grid');
    grid.innerHTML = '<div class="loading-placeholder"><span class="loading-spinner"></span> Fetching live prices...</div>';

    try {
        const res = await fetch(`${API}/market/live`);
        const data = await res.json();

        const rates = data.rates || {};
        const changes = data.changes || {};

        document.getElementById('stat-symbols').textContent = data.symbols_fetched ?? '0';

        if (Object.keys(rates).length === 0) {
            grid.innerHTML = '<div class="empty-state">No market data available. This can happen outside trading hours.</div>';
            return;
        }

        grid.innerHTML = '';
        const sorted = Object.entries(rates).sort((a, b) => {
            const ca = Math.abs(changes[a[0]] || 0);
            const cb = Math.abs(changes[b[0]] || 0);
            return cb - ca;
        });

        for (const [symbol, price] of sorted) {
            const change = changes[symbol] || 0;
            const direction = change > 0.01 ? 'up' : change < -0.01 ? 'down' : 'flat';
            const arrow = change > 0.01 ? '▲' : change < -0.01 ? '▼' : '—';
            const displayPrice = price > 1000 ? price.toLocaleString('en-US', {maximumFractionDigits: 0})
                : price > 10 ? price.toFixed(2) : price.toFixed(4);

            const card = document.createElement('div');
            card.className = 'market-card';
            card.innerHTML = `
                <div class="market-symbol">${cleanSymbol(symbol)}</div>
                <div class="market-price">${displayPrice}</div>
                <div class="market-change ${direction}">${arrow} ${Math.abs(change).toFixed(2)}%</div>
            `;
            grid.appendChild(card);
        }
    } catch (e) {
        grid.innerHTML = '<div class="empty-state">Failed to load market data. Check if the API server is running.</div>';
    }
}

function cleanSymbol(s) {
    return s.replace('=X', '').replace('-USD', '').replace('^', '');
}

// --- Alerts ---
async function loadAlerts() {
    try {
        const res = await fetch(`${API}/alerts?limit=10`);
        const data = await res.json();
        const container = document.getElementById('alerts-container');

        if (!data.alerts || data.alerts.length === 0) {
            container.innerHTML = '<div class="empty-state">No alerts yet. The system monitors prices for significant moves.</div>';
            return;
        }

        container.innerHTML = '';
        for (const alert of data.alerts) {
            const type = alert.type || 'info';
            const cls = type.includes('spike') ? 'spike' : type.includes('news') ? 'news' : 'sentiment';
            const item = document.createElement('div');
            item.className = `alert-item ${cls}`;
            item.innerHTML = `
                <span class="alert-type ${cls}">${type.replace('_', ' ')}</span>
                <div class="alert-body">
                    <div class="alert-message">${alert.message || alert.description || JSON.stringify(alert)}</div>
                    <div class="alert-time">${alert.timestamp || ''}</div>
                </div>
            `;
            container.appendChild(item);
        }
    } catch (e) {
        console.error('Alerts load failed:', e);
    }
}

// --- News Feed ---
let currentCategory = 'all';

function filterFeed(category, btnEl) {
    currentCategory = category;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');
    loadNewsFeed();
}

async function loadNewsFeed() {
    const container = document.getElementById('feed-container');
    container.innerHTML = '<div class="loading-placeholder"><span class="loading-spinner"></span> Loading news feed...</div>';

    try {
        const res = await fetch(`${API}/data/feed?limit=50&category=${currentCategory}`);
        const data = await res.json();

        if (!data.items || data.items.length === 0) {
            container.innerHTML = `<div class="empty-state">
                No news ingested yet. Run the ingestion pipeline to start collecting news.
                <br><br>
                <code style="font-size:12px;color:var(--text-muted)">
                    celery -A finsight.workers.celery_app worker --loglevel=info
                </code>
            </div>`;
            return;
        }

        container.innerHTML = '';
        for (const item of data.items) {
            const sentClass = item.sentiment_label === 'positive' ? 'positive'
                : item.sentiment_label === 'negative' ? 'negative' : 'neutral';

            const entitiesHtml = (item.entities || []).slice(0, 5).map(e =>
                `<span class="entity-tag">${escHtml(e)}</span>`
            ).join('');

            const geoHtml = (item.geopolitical_tags || []).slice(0, 4).map(t =>
                `<span class="geo-tag">${escHtml(t)}</span>`
            ).join('');

            const assetHtml = (item.asset_classes || []).filter(a => a !== 'macro').slice(0, 3).map(a =>
                `<span class="asset-tag">${escHtml(a)}</span>`
            ).join('');

            const el = document.createElement('div');
            el.className = 'feed-item';
            el.innerHTML = `
                <div class="feed-sentiment ${sentClass}"></div>
                <div class="feed-body">
                    <div class="feed-title">
                        ${item.url ? `<a href="${item.url}" target="_blank">${escHtml(item.title)}</a>` : escHtml(item.title)}
                    </div>
                    <div class="feed-text">${escHtml(item.text)}</div>
                    <div class="feed-meta">
                        <span class="feed-tag ${sentClass}">${item.sentiment_label}</span>
                        <span>${escHtml(item.source)}</span>
                        <span>${formatTime(item.published_at)}</span>
                    </div>
                    <div class="feed-tags-row">
                        ${entitiesHtml}${geoHtml}${assetHtml}
                    </div>
                </div>
            `;
            container.appendChild(el);
        }
    } catch (e) {
        container.innerHTML = '<div class="empty-state">Failed to load news feed. Check if the API server is running.</div>';
    }
}

// --- Ask AI ---
let isAsking = false;

function askSuggestion(el) {
    document.getElementById('ask-input').value = el.textContent;
    askQuestion();
}

async function askQuestion() {
    const input = document.getElementById('ask-input');
    const question = input.value.trim();
    if (!question || isAsking) return;

    isAsking = true;
    input.value = '';
    document.getElementById('btn-send').disabled = true;

    const messages = document.getElementById('chat-messages');
    const welcome = messages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    // User message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-msg user';
    userMsg.innerHTML = `
        <div class="chat-avatar">U</div>
        <div>
            <div class="chat-bubble">${escHtml(question)}</div>
        </div>
    `;
    messages.appendChild(userMsg);

    // Loading
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'chat-msg assistant';
    loadingMsg.innerHTML = `
        <div class="chat-avatar">AI</div>
        <div>
            <div class="chat-bubble"><span class="loading-spinner"></span> Analyzing...</div>
        </div>
    `;
    messages.appendChild(loadingMsg);
    messages.scrollTop = messages.scrollHeight;

    try {
        const res = await fetch(`${API}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, hours_back: 24 }),
        });
        const data = await res.json();

        loadingMsg.innerHTML = `
            <div class="chat-avatar">AI</div>
            <div>
                <div class="chat-bubble">${formatAnswer(data.answer || data.detail || 'No response received.')}</div>
                <div class="chat-meta">
                    <span>Provider: ${data.provider || 'unknown'}</span>
                    <span>Chunks used: ${data.chunks_used ?? 0}</span>
                    ${data.sources?.length ? `<span>Sources: ${data.sources.length}</span>` : ''}
                </div>
            </div>
        `;
    } catch (e) {
        loadingMsg.innerHTML = `
            <div class="chat-avatar">AI</div>
            <div>
                <div class="chat-bubble" style="border-color:var(--red)">Error: ${escHtml(e.message)}. Make sure the API server is running.</div>
            </div>
        `;
    }

    isAsking = false;
    document.getElementById('btn-send').disabled = false;
    messages.scrollTop = messages.scrollHeight;
}

// --- Utilities ---
function escHtml(str) {
    const d = document.createElement('div');
    d.textContent = str || '';
    return d.innerHTML;
}

function formatAnswer(text) {
    return escHtml(text)
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.*?)`/g, '<code style="background:var(--bg-primary);padding:1px 4px;border-radius:3px;font-size:12px">$1</code>');
}

function formatTime(iso) {
    if (!iso) return '';
    try {
        const d = new Date(iso);
        const now = new Date();
        const diff = (now - d) / 1000;
        if (diff < 60) return 'just now';
        if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
        return d.toLocaleDateString();
    } catch { return iso; }
}

// --- Initial Load ---
async function init() {
    loadHealth();
    loadStats();
    loadMarketData();
    loadAlerts();
}

init();
setInterval(loadHealth, 30000);
setInterval(loadMarketData, 60000);
