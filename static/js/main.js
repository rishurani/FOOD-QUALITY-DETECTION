/**
 * LexBrief AI - Frontend Client Controller
 * Project ID: CSEAIML2026-P13
 * Handles AJAX requests, IRAC tab switching, file uploads, and export actions.
 */

// Current Brief State
let currentBriefData = null;

// Modal Controls
function openEthicsModal() {
    const modal = document.getElementById('ethicsModal');
    if (modal) modal.style.display = 'flex';
}

function closeEthicsModal() {
    const modal = document.getElementById('ethicsModal');
    if (modal) modal.style.display = 'none';
}

// Close modal on outside click
window.addEventListener('click', function(event) {
    const modal = document.getElementById('ethicsModal');
    if (modal && event.target === modal) {
        modal.style.display = 'none';
    }
});

// Load Landmark Case Preset
function loadPresetCase(caseId) {
    if (!caseId) return;

    fetch(`/api/cases/${caseId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.case) {
                const textarea = document.getElementById('judgmentText');
                if (textarea) {
                    textarea.value = data.case.text;
                }
                triggerSummarization();
            }
        })
        .catch(err => {
            console.error('Error fetching case details:', err);
        });
}

// File Upload Handler (.txt files)
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        const textarea = document.getElementById('judgmentText');
        if (textarea) {
            textarea.value = text;
        }
        triggerSummarization();
    };
    reader.readAsText(file);
}

// Drag & Drop Setup
const dropArea = document.getElementById('dropArea');
if (dropArea) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
    });

    dropArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            document.getElementById('fileInput').files = files;
            handleFileUpload({ target: { files: files } });
        }
    }, false);
}

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.brief-tabs .tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    event.currentTarget.classList.add('active');
    const targetPane = document.getElementById(tabId);
    if (targetPane) targetPane.classList.add('active');
}

// Trigger Summarization & Element Extraction
function triggerSummarization() {
    const text = document.getElementById('judgmentText') ? document.getElementById('judgmentText').value.trim() : '';
    const presetSelect = document.getElementById('presetSelect');
    const caseId = presetSelect ? presetSelect.value : '';
    const mode = document.getElementById('summaryMode') ? document.getElementById('summaryMode').value : 'structured';
    const focus = document.getElementById('summaryFocus') ? document.getElementById('summaryFocus').value : 'all';
    const targetSentences = document.getElementById('targetSentences') ? document.getElementById('targetSentences').value : 8;

    if (!text && !caseId) {
        alert('Please paste a court judgment or select a landmark case from the dropdown.');
        return;
    }

    // UI States
    const loadingState = document.getElementById('loadingState');
    const emptyState = document.getElementById('emptyState');
    const resultsCard = document.getElementById('resultsCard');
    const btnSummarize = document.getElementById('btnSummarize');

    if (loadingState) loadingState.style.display = 'block';
    if (emptyState) emptyState.style.display = 'none';
    if (resultsCard) resultsCard.style.display = 'none';
    if (btnSummarize) {
        btnSummarize.disabled = true;
        btnSummarize.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Judgment with Legal NLP...';
    }

    const payload = {
        text: text,
        case_id: caseId || undefined,
        mode: mode,
        focus: focus,
        target_sentences: parseInt(targetSentences, 10)
    };

    fetch('/api/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        if (!data.success) {
            alert('Error generating brief: ' + (data.error || 'Unknown error'));
            if (emptyState) emptyState.style.display = 'block';
            return;
        }

        currentBriefData = data;
        renderBriefResults(data);

        if (resultsCard) resultsCard.style.display = 'block';
    })
    .catch(err => {
        console.error('Summarization request failed:', err);
        alert('Failed to connect to the legal NLP engine.');
        if (emptyState) emptyState.style.display = 'block';
    })
    .finally(() => {
        if (loadingState) loadingState.style.display = 'none';
        if (btnSummarize) {
            btnSummarize.disabled = false;
            btnSummarize.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Generate Structured Brief & Extract Elements';
        }
    });
}

// Render Results into UI
function renderBriefResults(data) {
    const brief = data.brief || {};
    const ext = data.extracted_elements || {};
    const meta = ext.metadata || {};
    const irac = brief.irac || {};
    const stats = brief.stats || {};
    const outcome = ext.outcome || {};

    // Populate Metadata
    document.getElementById('resTitle').innerText = meta.title || 'In Re: Judicial Record';
    document.getElementById('resCitation').innerHTML = `<i class="fa-solid fa-bookmark"></i> ${meta.citation || 'Citation Pending'}`;
    document.getElementById('resCourt').innerHTML = `<i class="fa-solid fa-building-columns"></i> ${meta.court || 'Court of Record'}`;
    document.getElementById('resDate').innerHTML = `<i class="fa-solid fa-calendar"></i> ${meta.date || 'Record Date'}`;
    
    // Domain & Outcome Badges
    const resDomain = document.getElementById('resDomain');
    if (resDomain) resDomain.innerText = ext.domain || 'General Jurisprudence';

    const resOutcome = document.getElementById('resOutcome');
    if (resOutcome) {
        resOutcome.className = `outcome-badge outcome-${outcome.badge || 'secondary'}`;
        resOutcome.innerText = outcome.status || 'Disposed';
        resOutcome.title = outcome.summary || '';
    }

    // Stats
    document.getElementById('statOriginalWords').innerText = stats.total_words || 0;
    document.getElementById('statSummaryWords').innerText = stats.summary_words || 0;
    document.getElementById('statCompression').innerText = `${brief.compression_ratio || 0}%`;
    document.getElementById('statSentences').innerText = `${stats.summary_sentences || 0} / ${stats.total_sentences || 0}`;

    // IRAC Contents
    renderListToContainer('briefFacts', irac.facts || []);
    renderListToContainer('briefIssues', irac.issues || []);
    renderListToContainer('briefArguments', irac.arguments || []);
    renderListToContainer('briefReasoning', irac.reasoning || []);
    renderListToContainer('briefRuling', irac.ruling || []);

    // Executive Synthesis
    const execBox = document.getElementById('briefExecutiveText');
    if (execBox) execBox.innerText = brief.executive_summary || 'Executive brief generated.';

    // Extracted Entities
    renderTags('listStatutes', ext.statutes_cited || [], 'statute-tag');
    renderTags('listPrecedents', ext.precedents_cited || [], 'precedent-tag');
    renderTags('listBench', meta.bench || [], 'coram-tag');

    // Key Takeaways
    const takeawaysList = document.getElementById('listTakeaways');
    if (takeawaysList) {
        takeawaysList.innerHTML = '';
        const items = brief.key_takeaways || [];
        if (items.length === 0) {
            takeawaysList.innerHTML = '<li>Key holdings and doctrines extracted in structured brief tabs.</li>';
        } else {
            items.forEach(t => {
                const li = document.createElement('li');
                li.innerText = t;
                takeawaysList.appendChild(li);
            });
        }
    }
}

function renderListToContainer(containerId, items) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';
    if (!items || items.length === 0) {
        el.innerHTML = '<p class="text-muted">No specific sentences isolated for this section.</p>';
        return;
    }
    items.forEach(text => {
        const p = document.createElement('p');
        p.innerText = text;
        el.appendChild(p);
    });
}

function renderTags(containerId, items, tagClass) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';
    if (!items || items.length === 0) {
        el.innerHTML = '<span class="text-muted text-sm">None recorded in extracted transcript.</span>';
        return;
    }
    items.forEach(item => {
        const span = document.createElement('span');
        span.className = tagClass;
        span.innerText = item;
        el.appendChild(span);
    });
}

// Copy to Clipboard
function copyBriefText() {
    if (!currentBriefData) return;
    const meta = currentBriefData.extracted_elements.metadata;
    const irac = currentBriefData.brief.irac;

    let text = `==========================================================\n`;
    text += `LEXBRIEF AI - STRUCTURED LEGAL BRIEF\n`;
    text += `==========================================================\n`;
    text += `TITLE: ${meta.title}\n`;
    text += `CITATION: ${meta.citation}\n`;
    text += `COURT: ${meta.court} (${meta.date})\n`;
    text += `OUTCOME: ${currentBriefData.extracted_elements.outcome.status}\n\n`;

    text += `--- I. CASE FACTS ---\n${(irac.facts || []).join('\n')}\n\n`;
    text += `--- II. LEGAL ISSUES ---\n${(irac.issues || []).join('\n')}\n\n`;
    text += `--- III. ARGUMENTS ---\n${(irac.arguments || []).join('\n')}\n\n`;
    text += `--- IV. REASONING & RATIO DECIDENDI ---\n${(irac.reasoning || []).join('\n')}\n\n`;
    text += `--- V. FINAL RULING & DISPOSITION ---\n${(irac.ruling || []).join('\n')}\n\n`;
    text += `DISCLAIMER: Generated by LexBrief AI (Project ID: CSEAIML2026-P13) for academic research only.\n`;

    navigator.clipboard.writeText(text).then(() => {
        alert('Structured legal brief copied to clipboard!');
    });
}

// Export Brief (Markdown or JSON)
function exportBrief(format) {
    if (!currentBriefData) return;
    const meta = currentBriefData.extracted_elements.metadata;
    const irac = currentBriefData.brief.irac;
    const title = meta.title || 'Case_Brief';

    let content = '';
    if (format === 'markdown') {
        content = `# ${meta.title}\n\n` +
            `**Citation:** ${meta.citation}  \n` +
            `**Court:** ${meta.court}  \n` +
            `**Date:** ${meta.date}  \n` +
            `**Domain:** ${currentBriefData.extracted_elements.domain}  \n` +
            `**Disposition Outcome:** ${currentBriefData.extracted_elements.outcome.status}  \n\n` +
            `---\n\n` +
            `## I. Case Facts\n${(irac.facts || []).map(f => `- ${f}`).join('\n')}\n\n` +
            `## II. Legal Issues Framed\n${(irac.issues || []).map(i => `- ${i}`).join('\n')}\n\n` +
            `## III. Contentions of Parties\n${(irac.arguments || []).map(a => `- ${a}`).join('\n')}\n\n` +
            `## IV. Judicial Reasoning & Ratio Decidendi\n${(irac.reasoning || []).join('\n\n')}\n\n` +
            `## V. Final Ruling & Order\n${(irac.ruling || []).join('\n\n')}\n\n` +
            `## VI. Statutes Cited\n${(currentBriefData.extracted_elements.statutes_cited || []).map(s => `- \`${s}\``).join('\n')}\n\n` +
            `---\n*Generated by LexBrief AI (Project ID: CSEAIML2026-P13). Academic research only. Not legal advice.*`;
    } else {
        content = currentBriefData;
    }

    fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            title: title,
            format: format,
            content: content
        })
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${title.replace(/[^a-zA-Z0-9]/g, '_')}_Brief.${format === 'json' ? 'json' : 'md'}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    })
    .catch(err => {
        console.error('Export download failed:', err);
    });
}
