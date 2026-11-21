/**
 * FreshVision AI - Global Application Scripts
 * Manages navigation, model health checking, global toasts, and modals.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Mobile navigation menu toggle
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileDrawer = document.getElementById('mobileDrawer');

    if (mobileMenuBtn && mobileDrawer) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileDrawer.classList.toggle('open');
        });
    }

    // 2. Global Model Health Check
    checkModelHealth();

    // 3. Modal event listeners setup
    setupModalEvents();
});

/**
 * Checks system health and updates the navbar status pill.
 */
async function checkModelHealth() {
    const pill = document.getElementById('nav-model-pill');
    if (!pill) return;

    try {
        const response = await fetch('/api/health');
        if (!response.ok) throw new Error('Health check returned non-200');

        const data = await response.json();
        const label = pill.querySelector('.status-label');

        pill.classList.remove('status-checking', 'status-ready', 'status-warning');

        if (data.model_loaded) {
            pill.classList.add('status-ready');
            if (label) label.textContent = 'Model Online';
            pill.title = `Model: ${data.model_version} (Loaded in memory)`;
        } else {
            pill.classList.add('status-warning');
            if (label) label.textContent = 'Model Untrained';
            pill.title = 'No trained model weights loaded. Run train_model.py.';
        }
    } catch (err) {
        console.warn('Health check unreachable:', err);
        const pill = document.getElementById('nav-model-pill');
        if (pill) {
            pill.classList.remove('status-checking', 'status-ready');
            pill.classList.add('status-warning');
            const label = pill.querySelector('.status-label');
            if (label) label.textContent = 'Server Standby';
        }
    }
}

/**
 * Displays a non-blocking toast alert in the top right.
 * @param {string} message - Message text
 * @param {'success'|'error'|'warning'|'info'} type - Toast type
 * @param {number} durationMs - Auto-dismiss duration in milliseconds
 */
window.showToast = function(message, type = 'info', durationMs = 4000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let iconSvg = '';
    if (type === 'success') {
        iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
    } else if (type === 'error') {
        iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
    } else if (type === 'warning') {
        iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.5"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    } else {
        iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
    }

    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            ${iconSvg}
            <span>${message}</span>
        </div>
        <button style="background: none; border: none; cursor: pointer; color: #94A3B8; font-size: 1.1rem; line-height: 1;" onclick="this.parentElement.remove();">&times;</button>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }
    }, durationMs);
};

// Global modal state holder
let currentModalConfirmCallback = null;

function setupModalEvents() {
    const modal = document.getElementById('globalModal');
    const closeBtn = document.getElementById('modalCloseBtn');
    const cancelBtn = document.getElementById('modalCancelBtn');
    const confirmBtn = document.getElementById('modalConfirmBtn');

    if (!modal) return;

    const closeModal = () => {
        modal.style.display = 'none';
        currentModalConfirmCallback = null;
    };

    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (cancelBtn) cancelBtn.addEventListener('click', closeModal);

    if (confirmBtn) {
        confirmBtn.addEventListener('click', () => {
            if (typeof currentModalConfirmCallback === 'function') {
                currentModalConfirmCallback();
            }
            closeModal();
        });
    }

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
}

/**
 * Open confirmation modal
 */
window.showConfirmModal = function({ title, message, confirmText = 'Confirm', onConfirm }) {
    const modal = document.getElementById('globalModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalMessage = document.getElementById('modalMessage');
    const confirmBtn = document.getElementById('modalConfirmBtn');

    if (!modal) return;

    if (modalTitle) modalTitle.textContent = title;
    if (modalMessage) modalMessage.textContent = message;
    if (confirmBtn) confirmBtn.textContent = confirmText;

    currentModalConfirmCallback = onConfirm;
    modal.style.display = 'flex';
};
