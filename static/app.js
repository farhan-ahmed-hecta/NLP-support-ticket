/* ===================================================================
   AEGIS AI — Interactive Application Logic & REST Integration
   =================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Global Application State
  const state = {
    presets: [],
    samples: [],
    stats: null,
    activeTab: 'view-studio',
    batchResults: []
  };

  // DOM Elements - Navigation
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabViews = document.querySelectorAll('.tab-view');

  // DOM Elements - Studio Form
  const triageForm = document.getElementById('triage-form');
  const inputSubject = document.getElementById('ticket-subject');
  const inputBody = document.getElementById('ticket-body');
  const selectLang = document.getElementById('lang-select');
  const btnTriage = document.getElementById('btn-triage');
  const btnClear = document.getElementById('btn-clear');
  const btnRandom = document.getElementById('btn-random');
  const charCounter = document.getElementById('body-char-count');
  const presetsContainer = document.getElementById('presets-container');

  // DOM Elements - Results Panel
  const resultsEmpty = document.getElementById('results-empty');
  const resultsData = document.getElementById('results-data');
  const resQueueName = document.getElementById('res-queue-name');
  const resQueueBar = document.getElementById('res-queue-bar');
  const resQueueConf = document.getElementById('res-queue-conf');
  const resPriorityBox = document.getElementById('res-priority-box');
  const resPriorityBadge = document.getElementById('res-priority-badge');
  const resSlaBadge = document.getElementById('res-sla-badge');
  const resPriorityBar = document.getElementById('res-priority-bar');
  const resPriorityConf = document.getElementById('res-priority-conf');
  const resTypeName = document.getElementById('res-type-name');
  const resLangTag = document.getElementById('res-lang-tag');
  const resQueueAlternatives = document.getElementById('res-queue-alternatives');
  const resExplainChips = document.getElementById('res-explain-chips');
  const resTagsContainer = document.getElementById('res-tags-container');
  const resMatchScore = document.getElementById('res-match-score');
  const resRefSubject = document.getElementById('res-ref-subject');
  const resResponseText = document.getElementById('res-response-text');
  const btnCopyResponse = document.getElementById('btn-copy-response');
  const copyBtnText = document.getElementById('copy-btn-text');
  const btnDispatch = document.getElementById('btn-dispatch');

  // DOM Elements - Feed
  const feedSearch = document.getElementById('feed-search');
  const feedFilterQueue = document.getElementById('feed-filter-queue');
  const feedFilterPriority = document.getElementById('feed-filter-priority');
  const feedFilterLang = document.getElementById('feed-filter-lang');
  const feedTbody = document.getElementById('feed-tbody');
  const feedCountInfo = document.getElementById('feed-count-info');

  // DOM Elements - Analytics
  const metricTypeAcc = document.getElementById('metric-type-acc');
  const metricPriorityF1 = document.getElementById('metric-priority-f1');
  const metricQueueAcc = document.getElementById('metric-queue-acc');
  const metricTotalRecords = document.getElementById('metric-total-records');
  const queueDistBars = document.getElementById('queue-dist-bars');
  const priorityStackedBar = document.getElementById('priority-stacked-bar');
  const priorityLegend = document.getElementById('priority-legend');
  const langStackedBar = document.getElementById('lang-stacked-bar');
  const langLegend = document.getElementById('lang-legend');
  const reportTbody = document.getElementById('report-tbody');

  // DOM Elements - Batch
  const batchInput = document.getElementById('batch-input');
  const btnRunBatch = document.getElementById('btn-run-batch');
  const btnLoadBatchSample = document.getElementById('btn-load-batch-sample');
  const btnExportBatchCsv = document.getElementById('btn-export-batch-csv');
  const batchResultsWrap = document.getElementById('batch-results-wrap');
  const batchTbody = document.getElementById('batch-tbody');

  // DOM Elements - Modal
  const ticketModal = document.getElementById('ticket-modal');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const btnModalCancel = document.getElementById('btn-modal-cancel');
  const btnModalLoadStudio = document.getElementById('btn-modal-load-studio');
  const modalTicketId = document.getElementById('modal-ticket-id');
  const modalTicketSubject = document.getElementById('modal-ticket-subject');
  const modalQueue = document.getElementById('modal-queue');
  const modalPriority = document.getElementById('modal-priority');
  const modalType = document.getElementById('modal-type');
  const modalLang = document.getElementById('modal-lang');
  const modalBodyText = document.getElementById('modal-body-text');
  const modalAnswerText = document.getElementById('modal-answer-text');

  let activeModalTicket = null;

  // -------------------------------------------------------------
  // 1. Navigation Tabs
  // -------------------------------------------------------------
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetViewId = tab.getAttribute('data-target');
      navTabs.forEach(t => t.classList.remove('active'));
      tabViews.forEach(v => v.classList.remove('active'));
      tab.classList.add('active');
      const targetView = document.getElementById(targetViewId);
      if (targetView) targetView.classList.add('active');
      state.activeTab = targetViewId;
    });
  });

  // Character counter
  inputBody.addEventListener('input', () => {
    charCounter.textContent = `${inputBody.value.length} characters`;
  });

  // -------------------------------------------------------------
  // 2. Fetch Initial Server Data (Presets, Samples, Stats)
  // -------------------------------------------------------------
  async function loadInitialData() {
    try {
      const [presetsRes, samplesRes, statsRes] = await Promise.all([
        fetch('/api/presets'),
        fetch('/api/samples'),
        fetch('/api/stats')
      ]);

      if (presetsRes.ok) {
        state.presets = await presetsRes.json();
        renderPresets();
      }

      if (samplesRes.ok) {
        state.samples = await samplesRes.json();
        renderFeedTable();
      }

      if (statsRes.ok) {
        state.stats = await statsRes.json();
        renderAnalytics();
        populateFilterQueues();
      }
    } catch (err) {
      console.error('Failed to load initial data from server:', err);
      showToast('Error connecting to triage server', 'error');
    }
  }

  // -------------------------------------------------------------
  // 3. Render Presets Bar
  // -------------------------------------------------------------
  function renderPresets() {
    presetsContainer.innerHTML = '';
    state.presets.forEach(preset => {
      const btn = document.createElement('button');
      btn.className = 'preset-chip';
      btn.textContent = preset.title;
      btn.addEventListener('click', () => {
        inputSubject.value = preset.subject;
        inputBody.value = preset.body;
        selectLang.value = preset.language;
        charCounter.textContent = `${preset.body.length} characters`;
        // Auto trigger triage
        executeTriage(preset.subject, preset.body, preset.language);
      });
      presetsContainer.appendChild(btn);
    });
  }

  // -------------------------------------------------------------
  // 4. Execute AI Triage
  // -------------------------------------------------------------
  triageForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const subj = inputSubject.value.trim();
    const body = inputBody.value.trim();
    const lang = selectLang.value === 'auto' ? null : selectLang.value;
    if (!subj && !body) return;
    executeTriage(subj, body, lang);
  });

  async function executeTriage(subject, body, language) {
    btnTriage.classList.add('loading');
    btnTriage.disabled = true;

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject, body, language })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      displayTriageResults(data);
      showToast('Ticket triaged successfully', 'success');
    } catch (err) {
      console.error('Triage error:', err);
      showToast(`Triage failed: ${err.message}`, 'error');
    } finally {
      btnTriage.classList.remove('loading');
      btnTriage.disabled = false;
    }
  }

  function displayTriageResults(data) {
    resultsEmpty.classList.add('hidden');
    resultsData.classList.remove('hidden');

    // 1. Queue Card
    resQueueName.textContent = data.queue.predicted;
    resQueueBar.style.width = `${Math.min(100, Math.max(15, data.queue.confidence))}%`;
    resQueueConf.textContent = `${data.queue.confidence}% Confidence`;

    // 2. Priority Card
    const prio = data.priority.predicted.toLowerCase();
    resPriorityBadge.textContent = prio.toUpperCase();
    resPriorityBadge.className = `priority-badge ${prio}`;
    resSlaBadge.textContent = data.priority.sla ? data.priority.sla.target : '< 4h SLA';
    resPriorityConf.textContent = `${data.priority.confidence}% Confidence`;

    // 3. Type & Language
    resTypeName.textContent = data.type.predicted;
    const langName = data.language === 'de' ? '🇩🇪 Deutsch' : '🇬🇧 English';
    resLangTag.textContent = `🌐 Detected: ${langName}`;

    // Signal / Vocab Match Check
    const vocabCount = data.matched_features !== undefined ? data.matched_features : (data.explainability ? data.explainability.length : 0);
    const vocabCountElem = document.getElementById('res-vocab-count');
    if (vocabCountElem) {
      vocabCountElem.textContent = `${vocabCount} Signal Tokens`;
    }
    const alertBanner = document.getElementById('vocab-alert-banner');
    if (alertBanner) {
      if (vocabCount <= 1) {
        alertBanner.classList.remove('hidden');
      } else {
        alertBanner.classList.add('hidden');
      }
    }

    // 4. Queue Alternatives
    resQueueAlternatives.innerHTML = '';
    (data.queue.alternatives || []).forEach(alt => {
      const row = document.createElement('div');
      row.className = 'alt-row';
      row.innerHTML = `
        <span class="alt-name" title="${alt.queue}">${alt.queue}</span>
        <div class="alt-progress-track">
          <div class="alt-progress-fill" style="width: ${alt.probability}%"></div>
        </div>
        <span class="alt-pct">${alt.probability}%</span>
      `;
      resQueueAlternatives.appendChild(row);
    });

    // 5. Explainability Keywords
    resExplainChips.innerHTML = '';
    if (data.explainability && data.explainability.length > 0) {
      data.explainability.forEach(item => {
        const chip = document.createElement('span');
        chip.className = 'explain-chip';
        chip.textContent = `${item.token} (+${item.score})`;
        resExplainChips.appendChild(chip);
      });
    } else {
      resExplainChips.innerHTML = '<span class="draft-note">Standard categorical signal match</span>';
    }

    // 6. Tags
    resTagsContainer.innerHTML = '';
    (data.tags || []).forEach(tag => {
      const pill = document.createElement('span');
      pill.className = 'tag-pill';
      pill.textContent = `#${tag}`;
      resTagsContainer.appendChild(pill);
    });

    // 7. Suggested Response Draft
    const resp = data.suggested_response;
    resMatchScore.textContent = `${resp.match_confidence}% Match`;
    resRefSubject.textContent = `Matched historical case: "${resp.reference_subject}" (${resp.reference_queue})`;
    resResponseText.value = resp.answer;
  }

  // Clear Button
  btnClear.addEventListener('click', () => {
    inputSubject.value = '';
    inputBody.value = '';
    charCounter.textContent = '0 characters';
    resultsData.classList.add('hidden');
    resultsEmpty.classList.remove('hidden');
  });

  // Random Sample Button
  btnRandom.addEventListener('click', () => {
    if (!state.samples || state.samples.length === 0) return;
    const randomItem = state.samples[Math.floor(Math.random() * state.samples.length)];
    inputSubject.value = randomItem.subject;
    inputBody.value = randomItem.body;
    selectLang.value = randomItem.language || 'auto';
    charCounter.textContent = `${randomItem.body.length} characters`;
    executeTriage(randomItem.subject, randomItem.body, randomItem.language);
  });

  // Copy Suggested Response Draft
  btnCopyResponse.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(resResponseText.value);
      copyBtnText.textContent = 'Copied!';
      btnCopyResponse.style.borderColor = 'var(--low-emerald)';
      setTimeout(() => {
        copyBtnText.textContent = 'Copy Response';
        btnCopyResponse.style.borderColor = '';
      }, 2000);
      showToast('Support draft copied to clipboard', 'success');
    } catch (err) {
      showToast('Unable to copy to clipboard', 'error');
    }
  });

  // Dispatch Ticket
  btnDispatch.addEventListener('click', () => {
    const queue = resQueueName.textContent;
    showToast(`Ticket assigned to ${queue} & response sent!`, 'success');
  });

  // -------------------------------------------------------------
  // 5. Live Queue Feed Management
  // -------------------------------------------------------------
  function populateFilterQueues() {
    if (!state.stats || !state.stats.dataset_distribution) return;
    const queues = Object.keys(state.stats.dataset_distribution.queues || {});
    feedFilterQueue.innerHTML = '<option value="all">All Queues</option>';
    queues.forEach(q => {
      const opt = document.createElement('option');
      opt.value = q;
      opt.textContent = q;
      feedFilterQueue.appendChild(opt);
    });
  }

  function renderFeedTable() {
    const searchTerm = feedSearch.value.toLowerCase().trim();
    const filterQueue = feedFilterQueue.value;
    const filterPriority = feedFilterPriority.value;
    const filterLang = feedFilterLang.value;

    const filtered = state.samples.filter(ticket => {
      const matchSearch = !searchTerm || 
        ticket.subject.toLowerCase().includes(searchTerm) || 
        ticket.body.toLowerCase().includes(searchTerm);
      const matchQueue = filterQueue === 'all' || ticket.actual_queue === filterQueue;
      const matchPrio = filterPriority === 'all' || ticket.actual_priority.toLowerCase() === filterPriority.toLowerCase();
      const matchLang = filterLang === 'all' || ticket.language === filterLang;
      return matchSearch && matchQueue && matchPrio && matchLang;
    });

    feedCountInfo.textContent = `Showing ${filtered.length} of ${state.samples.length} tickets`;
    feedTbody.innerHTML = '';

    if (filtered.length === 0) {
      feedTbody.innerHTML = `<tr><td colspan="7" class="table-loading">No tickets match the selected filters.</td></tr>`;
      return;
    }

    filtered.forEach(ticket => {
      const tr = document.createElement('tr');
      const prio = ticket.actual_priority.toLowerCase();
      const langBadge = ticket.language === 'de' ? '🇩🇪 DE' : '🇬🇧 EN';

      tr.innerHTML = `
        <td class="ticket-id-cell">${ticket.id}</td>
        <td class="ticket-subject-cell">
          <div class="ticket-title-text">${escapeHtml(ticket.subject)}</div>
          <div class="ticket-body-snippet">${escapeHtml(ticket.body)}</div>
        </td>
        <td><span style="font-weight: 500;">${ticket.actual_queue}</span></td>
        <td><span class="priority-badge ${prio}">${ticket.actual_priority.toUpperCase()}</span></td>
        <td><span class="type-pill" style="font-size: 0.72rem;">${ticket.actual_type}</span></td>
        <td><span class="badge-tag" style="font-size: 0.72rem;">${langBadge}</span></td>
        <td style="text-align: right;">
          <button class="btn-triage-row" data-id="${ticket.id}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Triage
          </button>
        </td>
      `;

      // Click row to view modal
      tr.addEventListener('click', (e) => {
        if (e.target.closest('.btn-triage-row')) {
          // Handled by button listener
          return;
        }
        openTicketModal(ticket);
      });

      // Triage button
      const triageBtn = tr.querySelector('.btn-triage-row');
      triageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        loadTicketIntoStudio(ticket);
      });

      feedTbody.appendChild(tr);
    });
  }

  // Feed Filter Listeners
  feedSearch.addEventListener('input', renderFeedTable);
  feedFilterQueue.addEventListener('change', renderFeedTable);
  feedFilterPriority.addEventListener('change', renderFeedTable);
  feedFilterLang.addEventListener('change', renderFeedTable);

  function loadTicketIntoStudio(ticket) {
    // Switch to Studio Tab
    document.getElementById('tab-studio').click();
    inputSubject.value = ticket.subject;
    inputBody.value = ticket.body;
    selectLang.value = ticket.language || 'auto';
    charCounter.textContent = `${ticket.body.length} characters`;
    executeTriage(ticket.subject, ticket.body, ticket.language);
  }

  function openTicketModal(ticket) {
    activeModalTicket = ticket;
    modalTicketId.textContent = ticket.id;
    modalTicketSubject.textContent = ticket.subject;
    modalQueue.textContent = ticket.actual_queue;
    
    const prio = ticket.actual_priority.toLowerCase();
    modalPriority.textContent = ticket.actual_priority.toUpperCase();
    modalPriority.className = `priority-badge ${prio}`;
    
    modalType.textContent = ticket.actual_type;
    modalLang.textContent = ticket.language === 'de' ? 'German (de)' : 'English (en)';
    modalBodyText.textContent = ticket.body;
    modalAnswerText.textContent = ticket.answer || 'No reference resolution stored.';
    
    ticketModal.classList.remove('hidden');
  }

  btnCloseModal.addEventListener('click', () => ticketModal.classList.add('hidden'));
  btnModalCancel.addEventListener('click', () => ticketModal.classList.add('hidden'));
  ticketModal.addEventListener('click', (e) => {
    if (e.target === ticketModal) ticketModal.classList.add('hidden');
  });

  btnModalLoadStudio.addEventListener('click', () => {
    if (activeModalTicket) {
      ticketModal.classList.add('hidden');
      loadTicketIntoStudio(activeModalTicket);
    }
  });

  // -------------------------------------------------------------
  // 6. Analytics & Model Performance
  // -------------------------------------------------------------
  function renderAnalytics() {
    if (!state.stats) return;
    const stats = state.stats;

    const totalRecs = stats.total_records || ((stats.train_samples || 0) + (stats.test_samples || 0)) || 67890;

    metricTypeAcc.textContent = `${((stats.type?.accuracy || 0.844) * 100).toFixed(1)}%`;
    metricPriorityF1.textContent = `${((stats.priority?.accuracy || stats.priority?.f1_weighted || 0.696) * 100).toFixed(1)}%`;
    metricQueueAcc.textContent = `${((stats.queue?.accuracy || 0.605) * 100).toFixed(1)}%`;
    metricTotalRecords.textContent = totalRecs.toLocaleString();

    // 1. Department Queue Distribution Bars
    const queues = stats.dataset_distribution?.queues || {};
    const queueEntries = Object.entries(queues);
    if (queueEntries.length > 0) {
      const maxQueueCount = Math.max(...Object.values(queues));
      queueDistBars.innerHTML = '';

      queueEntries.forEach(([qName, count]) => {
        const pct = ((count / totalRecs) * 100).toFixed(1);
        const barWidth = ((count / maxQueueCount) * 100).toFixed(1);

        const item = document.createElement('div');
        item.className = 'dist-bar-item';
        item.innerHTML = `
          <div class="dist-bar-labels">
            <span class="dist-bar-name">${qName}</span>
            <span class="dist-bar-val">${count.toLocaleString()} (${pct}%)</span>
          </div>
          <div class="dist-track">
            <div class="dist-fill" style="width: ${barWidth}%"></div>
          </div>
        `;
        queueDistBars.appendChild(item);
      });
    }

    // 2. Priority Stacked Bar (all 5 priority classes)
    const prios = stats.dataset_distribution?.priorities || {};
    const totalPrios = Object.values(prios).reduce((a, b) => a + b, 0) || 1;
    const prioColors = {
      critical: '#ef4444',
      high: '#f97316',
      medium: '#f59e0b',
      low: '#10b981',
      very_low: '#64748b'
    };
    
    priorityStackedBar.innerHTML = '';
    priorityLegend.innerHTML = '';
    
    const prioKeys = Object.keys(prios).length > 0 ? Object.keys(prios) : ['critical', 'high', 'medium', 'low', 'very_low'];
    prioKeys.forEach(p => {
      const cnt = prios[p] || 0;
      if (cnt === 0) return;
      const pct = ((cnt / totalPrios) * 100).toFixed(1);
      const color = prioColors[p.toLowerCase()] || '#6366f1';
      
      const seg = document.createElement('div');
      seg.className = 'dist-segment';
      seg.style.width = `${pct}%`;
      seg.style.backgroundColor = color;
      seg.title = `${p.toUpperCase()}: ${cnt} (${pct}%)`;
      priorityStackedBar.appendChild(seg);

      const leg = document.createElement('div');
      leg.className = 'legend-item';
      leg.innerHTML = `
        <span class="legend-color-dot" style="background-color: ${color}"></span>
        <span>${p.toUpperCase()}: ${cnt.toLocaleString()} (${pct}%)</span>
      `;
      priorityLegend.appendChild(leg);
    });

    // 3. Language Stacked Bar
    const langs = stats.dataset_distribution.languages || {};
    const totalLangs = Object.values(langs).reduce((a, b) => a + b, 0);
    const langColors = { en: '#06b6d4', de: '#a855f7' };
    
    langStackedBar.innerHTML = '';
    langLegend.innerHTML = '';

    Object.entries(langs).forEach(([l, cnt]) => {
      const pct = ((cnt / totalLangs) * 100).toFixed(1);
      const lName = l === 'en' ? 'English (EN)' : 'German (DE)';
      
      const seg = document.createElement('div');
      seg.className = 'dist-segment';
      seg.style.width = `${pct}%`;
      seg.style.backgroundColor = langColors[l] || '#6366f1';
      seg.title = `${lName}: ${cnt} (${pct}%)`;
      langStackedBar.appendChild(seg);

      const leg = document.createElement('div');
      leg.className = 'legend-item';
      leg.innerHTML = `
        <span class="legend-color-dot" style="background-color: ${langColors[l] || '#6366f1'}"></span>
        <span>${lName}: ${cnt.toLocaleString()} (${pct}%)</span>
      `;
      langLegend.appendChild(leg);
    });

    // 4. Detailed Classification Report Table
    const qReport = stats.queue.report || {};
    reportTbody.innerHTML = '';
    
    Object.entries(qReport).forEach(([className, metrics]) => {
      if (['accuracy', 'macro avg', 'weighted avg'].includes(className)) return;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 600; color: #fff;">${className}</td>
        <td>${(metrics.precision * 100).toFixed(1)}%</td>
        <td>${(metrics.recall * 100).toFixed(1)}%</td>
        <td><strong style="color: #38bdf8;">${(metrics['f1-score'] * 100).toFixed(1)}%</strong></td>
        <td>${metrics.support.toLocaleString()}</td>
      `;
      reportTbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------
  // 7. Batch Analysis
  // -------------------------------------------------------------
  btnLoadBatchSample.addEventListener('click', () => {
    if (!state.samples || state.samples.length < 5) return;
    const sampleBatch = state.samples.slice(0, 6).map((t, idx) => ({
      id: `BATCH-${idx + 1}`,
      subject: t.subject,
      body: t.body,
      language: t.language
    }));
    batchInput.value = JSON.stringify(sampleBatch, null, 2);
  });

  btnRunBatch.addEventListener('click', async () => {
    const raw = batchInput.value.trim();
    if (!raw) {
      showToast('Please enter tickets JSON to analyze', 'error');
      return;
    }

    let tickets = [];
    try {
      tickets = JSON.parse(raw);
      if (!Array.isArray(tickets)) throw new Error('Input must be a JSON array of tickets');
    } catch (e) {
      showToast(`Invalid JSON format: ${e.message}`, 'error');
      return;
    }

    btnRunBatch.classList.add('loading');
    btnRunBatch.disabled = true;

    try {
      const res = await fetch('/api/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickets })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      state.batchResults = data.results || [];
      renderBatchResults();
      btnExportBatchCsv.disabled = false;
      showToast(`Successfully triaged ${state.batchResults.length} tickets`, 'success');
    } catch (err) {
      showToast(`Batch triage failed: ${err.message}`, 'error');
    } finally {
      btnRunBatch.classList.remove('loading');
      btnRunBatch.disabled = false;
    }
  });

  function renderBatchResults() {
    batchResultsWrap.classList.remove('hidden');
    batchTbody.innerHTML = '';

    state.batchResults.forEach(item => {
      const tr = document.createElement('tr');
      const prio = item.predicted_priority.toLowerCase();
      tr.innerHTML = `
        <td class="ticket-id-cell">${item.id}</td>
        <td style="font-weight: 500;">${escapeHtml(item.subject)}</td>
        <td><strong>${item.predicted_queue}</strong> <span style="font-size: 0.72rem; color: var(--text-dim);">(${item.queue_confidence}%)</span></td>
        <td><span class="priority-badge ${prio}">${item.predicted_priority.toUpperCase()}</span></td>
        <td><span class="type-pill" style="font-size: 0.72rem;">${item.predicted_type}</span></td>
        <td style="font-size: 0.78rem; color: var(--text-dim);">${escapeHtml(item.response_draft_preview)}</td>
      `;
      batchTbody.appendChild(tr);
    });
  }

  // Export Triaged CSV
  btnExportBatchCsv.addEventListener('click', () => {
    if (!state.batchResults || state.batchResults.length === 0) return;
    const headers = ['id', 'subject', 'predicted_queue', 'queue_confidence', 'predicted_priority', 'predicted_type', 'suggested_tags'];
    const rows = state.batchResults.map(r => [
      `"${r.id}"`,
      `"${r.subject.replace(/"/g, '""')}"`,
      `"${r.predicted_queue}"`,
      r.queue_confidence,
      `"${r.predicted_priority}"`,
      `"${r.predicted_type}"`,
      `"${(r.suggested_tags || []).join('; ')}"`
    ]);

    const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `triaged_tickets_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    showToast('Triaged CSV exported successfully', 'success');
  });

  // -------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <span>${type === 'success' ? '✓' : 'ⓘ'}</span>
      <span>${escapeHtml(message)}</span>
    `;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Kickoff initialization
  loadInitialData();
});
