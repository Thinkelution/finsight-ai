// FinSight AI Dashboard — Frontend Application

const API = '';

// --- Tab Navigation ---
let predictionsLoaded = false;
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.content').forEach(c => c.classList.add('hidden'));
        const tabId = btn.dataset.tab;
        const tab = document.getElementById(`tab-${tabId}`);
        if (tab) tab.classList.remove('hidden');
        if (tabId === 'predictions' && !predictionsLoaded) {
            predictionsLoaded = true;
            loadPredictions();
        }
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
    if (!text) return '';
    const lines = text.split('\n');
    let html = '';
    let inList = false;
    let inOl = false;

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];

        // Headers
        if (/^#{1,3}\s/.test(line)) {
            if (inList) { html += '</ul>'; inList = false; }
            if (inOl) { html += '</ol>'; inOl = false; }
            const level = line.match(/^(#+)/)[1].length;
            const content = escHtml(line.replace(/^#+\s*/, ''));
            html += `<h${level + 1} style="margin:12px 0 6px;color:var(--text-primary)">${applyInline(content)}</h${level + 1}>`;
            continue;
        }

        // Unordered list
        if (/^\s*[-*•]\s/.test(line)) {
            if (inOl) { html += '</ol>'; inOl = false; }
            if (!inList) { html += '<ul style="margin:6px 0;padding-left:20px">'; inList = true; }
            const content = escHtml(line.replace(/^\s*[-*•]\s*/, ''));
            html += `<li style="margin:3px 0">${applyInline(content)}</li>`;
            continue;
        }

        // Ordered list
        if (/^\s*\d+[.)]\s/.test(line)) {
            if (inList) { html += '</ul>'; inList = false; }
            if (!inOl) { html += '<ol style="margin:6px 0;padding-left:20px">'; inOl = true; }
            const content = escHtml(line.replace(/^\s*\d+[.)]\s*/, ''));
            html += `<li style="margin:3px 0">${applyInline(content)}</li>`;
            continue;
        }

        if (inList) { html += '</ul>'; inList = false; }
        if (inOl) { html += '</ol>'; inOl = false; }

        // Empty line = paragraph break
        if (line.trim() === '') {
            html += '<div style="height:8px"></div>';
            continue;
        }

        // Regular paragraph
        html += `<p style="margin:4px 0;line-height:1.6">${applyInline(escHtml(line))}</p>`;
    }

    if (inList) html += '</ul>';
    if (inOl) html += '</ol>';
    return html;
}

function applyInline(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code style="background:var(--bg-primary);padding:1px 5px;border-radius:3px;font-size:0.9em">$1</code>')
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" style="color:var(--accent)">$1</a>')
        .replace(/(https?:\/\/[^\s<\]]+)/g, (match, url) => {
            if (match.includes('</a>') || text.indexOf(`(${url})`) !== -1) return match;
            return `<a href="${url}" target="_blank" rel="noopener" style="color:var(--accent);word-break:break-all">${url}</a>`;
        });
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

// --- Predictions ---
async function loadPredictions() {
    const grid = document.getElementById('pred-grid');
    const parallelsEl = document.getElementById('parallels-container');
    const analysisEl = document.getElementById('pred-analysis');
    const confValue = document.getElementById('pred-confidence-value');
    const confFill = document.getElementById('pred-confidence-fill');

    let seconds = 0;
    const timerEl = document.createElement('span');
    timerEl.className = 'loading-timer';
    grid.innerHTML = '';
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-placeholder';
    loadingDiv.innerHTML = '<span class="loading-spinner"></span> Analyzing historical patterns and generating predictions... <span class="loading-timer">0s</span>';
    grid.appendChild(loadingDiv);
    const timer = setInterval(() => {
        seconds++;
        const t = loadingDiv.querySelector('.loading-timer');
        if (t) t.textContent = `${seconds}s`;
    }, 1000);

    parallelsEl.innerHTML = '<div class="loading-placeholder"><span class="loading-spinner"></span> Searching 527 historical patterns...</div>';
    analysisEl.innerHTML = '<div class="loading-placeholder"><span class="loading-spinner"></span> Waiting for AI model...</div>';

    try {
        const res = await fetch(`${API}/predictions`);
        clearInterval(timer);
        const data = await res.json();

        const confidence = data.confidence || 0;
        confValue.textContent = `${confidence}%`;
        confFill.style.width = `${confidence}%`;
        confFill.className = `pred-confidence-fill ${confidence >= 70 ? 'high' : confidence >= 40 ? 'mid' : 'low'}`;

        const predictions = data.predictions || [];
        if (predictions.length === 0) {
            grid.innerHTML = '<div class="empty-state">No predictions available yet. Ensure historical data has been collected and indexed.</div>';
        } else {
            grid.innerHTML = '';
            for (const pred of predictions) {
                const dirClass = pred.direction === 'BULLISH' ? 'bullish' : pred.direction === 'BEARISH' ? 'bearish' : 'neutral-dir';
                const arrow = pred.direction === 'BULLISH' ? '▲' : pred.direction === 'BEARISH' ? '▼' : '—';
                const card = document.createElement('div');
                card.className = `pred-card ${dirClass}`;
                card.innerHTML = `
                    <div class="pred-card-header">
                        <span class="pred-asset">${escHtml(pred.asset)}</span>
                        <span class="pred-direction ${dirClass}">${arrow} ${pred.direction}</span>
                    </div>
                    <div class="pred-card-confidence">
                        <div class="pred-mini-bar">
                            <div class="pred-mini-fill ${dirClass}" style="width:${pred.confidence}%"></div>
                        </div>
                        <span>${pred.confidence}% confidence</span>
                    </div>
                    <div class="pred-card-reasoning">${escHtml(pred.reasoning || '')}</div>
                `;
                grid.appendChild(card);
            }
        }

        const parallels = data.parallels || [];
        if (parallels.length === 0) {
            parallelsEl.innerHTML = '<div class="empty-state">No historical parallels found. Run the historical data collection pipeline first.</div>';
        } else {
            parallelsEl.innerHTML = '';
            for (const p of parallels) {
                const matchPct = Math.round((p.similarity || 0) * 100);
                const el = document.createElement('div');
                el.className = 'parallel-card';
                el.innerHTML = `
                    <div class="parallel-header">
                        <span class="parallel-date">Week of ${escHtml(p.week)}</span>
                        <span class="parallel-match">${matchPct}% match</span>
                    </div>
                    <div class="parallel-summary">${escHtml(p.summary || '')}</div>
                `;
                parallelsEl.appendChild(el);
            }
        }

        const analysisText = data.prediction_text || 'No analysis available.';
        analysisEl.innerHTML = `<div class="pred-analysis-text">${formatAnswer(analysisText)}</div>`;

    } catch (e) {
        clearInterval(timer);
        grid.innerHTML = `<div class="empty-state">Failed to load predictions: ${escHtml(e.message)}</div>`;
        parallelsEl.innerHTML = '<div class="empty-state">Error loading parallels</div>';
        analysisEl.innerHTML = '<div class="empty-state">Error loading analysis</div>';
    }

    loadHistoricalStatus();
}

async function loadHistoricalStatus() {
    const container = document.getElementById('historical-status');
    try {
        const res = await fetch(`${API}/predictions/status`);
        const data = await res.json();

        container.innerHTML = `
            <div class="hist-status-grid">
                <div class="hist-status-item ${data.market_data ? 'active' : ''}">
                    <span class="hist-status-icon">${data.market_data ? '✓' : '○'}</span>
                    <div>
                        <div class="hist-status-label">Market Data</div>
                        <div class="hist-status-detail">${data.market_data_rows ? data.market_data_rows.toLocaleString() + ' rows' : 'Not collected'}</div>
                        ${data.market_date_range ? `<div class="hist-status-detail">${data.market_date_range}</div>` : ''}
                    </div>
                </div>
                <div class="hist-status-item ${data.economic_data ? 'active' : ''}">
                    <span class="hist-status-icon">${data.economic_data ? '✓' : '○'}</span>
                    <div>
                        <div class="hist-status-label">Economic Indicators</div>
                        <div class="hist-status-detail">${data.economic_data ? 'Available' : 'Not collected'}</div>
                    </div>
                </div>
                <div class="hist-status-item ${data.wikipedia_events ? 'active' : ''}">
                    <span class="hist-status-icon">${data.wikipedia_events ? '✓' : '○'}</span>
                    <div>
                        <div class="hist-status-label">Wikipedia Events</div>
                        <div class="hist-status-detail">${data.wikipedia_months ? data.wikipedia_months + ' months' : 'Not collected'}</div>
                    </div>
                </div>
                <div class="hist-status-item ${data.gdelt_articles ? 'active' : ''}">
                    <span class="hist-status-icon">${data.gdelt_articles ? '✓' : '○'}</span>
                    <div>
                        <div class="hist-status-label">GDELT News</div>
                        <div class="hist-status-detail">${data.gdelt_weeks ? data.gdelt_weeks + ' weeks' : 'Not collected'}</div>
                    </div>
                </div>
                <div class="hist-status-item ${data.training_pairs > 0 ? 'active' : ''}">
                    <span class="hist-status-icon">${data.training_pairs > 0 ? '✓' : '○'}</span>
                    <div>
                        <div class="hist-status-label">Training Pairs</div>
                        <div class="hist-status-detail">${data.training_pairs || 0} pairs generated</div>
                    </div>
                </div>
                <div class="hist-status-item ${data.indexed_patterns > 0 ? 'active' : ''}">
                    <span class="hist-status-icon">${data.indexed_patterns > 0 ? '✓' : '○'}</span>
                    <div>
                        <div class="hist-status-label">Indexed Patterns</div>
                        <div class="hist-status-detail">${data.indexed_patterns || 0} patterns in vector DB</div>
                    </div>
                </div>
            </div>
        `;
    } catch (e) {
        container.innerHTML = '<div class="empty-state">Unable to check historical data status</div>';
    }
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
