/**
 * FreshVision AI - Analysis History & Audit Trail Script
 * Manages paginated history, real-time search, category filtering,
 * record deletion, and clear history modal operations.
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentPage = 1;
    let currentCategory = 'all';
    let currentSort = 'newest';
    let currentSearch = '';
    let searchDebounceTimer = null;

    const historyGrid = document.getElementById('historyGrid');
    const emptyState = document.getElementById('historyEmptyState');
    const searchInput = document.getElementById('searchInput');
    const sortSelect = document.getElementById('sortSelect');
    const filterButtons = document.querySelectorAll('.filter-btn');
    const clearAllBtn = document.getElementById('clearAllHistoryBtn');
    const paginationControls = document.getElementById('paginationControls');
    const prevPageBtn = document.getElementById('prevPageBtn');
    const nextPageBtn = document.getElementById('nextPageBtn');
    const pageIndicator = document.getElementById('pageIndicator');

    // 1. Initial Load
    fetchHistory();

    // 2. Search Input with 350ms debounce
    if (searchInput) {
        // If query param 'search' exists in URL, prefill
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('search')) {
            searchInput.value = urlParams.get('search');
            currentSearch = urlParams.get('search');
        }

        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchDebounceTimer);
            searchDebounceTimer = setTimeout(() => {
                currentSearch = e.target.value.trim();
                currentPage = 1;
                fetchHistory();
            }, 350);
        });
    }

    // 3. Category Filter Buttons
    filterButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            filterButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentCategory = btn.dataset.category || 'all';
            currentPage = 1;
            fetchHistory();
        });
    });

    // 4. Sort Dropdown Change
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            currentSort = e.target.value;
            currentPage = 1;
            fetchHistory();
        });
    }

    // 5. Pagination Buttons
    if (prevPageBtn) {
        prevPageBtn.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                fetchHistory();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    }

    if (nextPageBtn) {
        nextPageBtn.addEventListener('click', () => {
            currentPage++;
            fetchHistory();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // 6. Clear All History with Global Confirmation Modal
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', () => {
            if (window.showConfirmModal) {
                window.showConfirmModal({
                    title: 'Purge Entire Analysis History?',
                    message: 'This action will permanently delete all logged predictions and remove their corresponding image files from disk. This cannot be undone.',
                    confirmText: 'Delete All Records',
                    onConfirm: async () => {
                        try {
                            const res = await fetch('/api/history', { method: 'DELETE' });
                            const data = await res.json();
                            if (data.success) {
                                if (window.showToast) window.showToast('All analysis records cleared.', 'success');
                                currentPage = 1;
                                fetchHistory();
                            } else {
                                throw new Error(data.message || 'Failed to clear history');
                            }
                        } catch (err) {
                            console.error('Clear history error:', err);
                            if (window.showToast) window.showToast(err.message, 'error');
                        }
                    }
                });
            }
        });
    }

    // 7. Fetch History Function
    async function fetchHistory() {
        if (!historyGrid) return;

        const params = new URLSearchParams({
            page: currentPage,
            per_page: 12,
            category: currentCategory,
            sort: currentSort,
            search: currentSearch
        });

        try {
            const response = await fetch(`/api/history?${params.toString()}`);
            if (!response.ok) throw new Error('Failed to retrieve history');

            const result = await response.json();
            const items = result.items || [];
            const pagination = result.pagination || {};

            renderHistoryCards(items);
            renderPagination(pagination);

        } catch (err) {
            console.error('History fetch error:', err);
            historyGrid.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--spoiled-color);">
                    Failed to load history records. Please try again.
                </div>
            `;
            if (emptyState) emptyState.style.display = 'none';
        }
    }

    // 8. Render Grid Cards
    function renderHistoryCards(items) {
        if (items.length === 0) {
            historyGrid.innerHTML = '';
            if (emptyState) emptyState.style.display = 'block';
            if (paginationControls) paginationControls.style.display = 'none';
            return;
        }

        if (emptyState) emptyState.style.display = 'none';

        historyGrid.innerHTML = items.map(item => {
            let badgeClass = 'badge-fresh';
            let labelText = item.predicted_label;

            if (item.is_uncertain) {
                badgeClass = 'badge-uncertain';
                labelText = 'Uncertain';
            } else if (item.predicted_label === 'Spoiled') {
                badgeClass = 'badge-spoiled';
            }

            return `
                <div class="history-card" id="history-card-${item.id}">
                    <div class="history-img-wrap">
                        <img src="${item.image_url}" alt="Sample #${item.id}" class="history-card-img"
                             onerror="this.src='https://placehold.co/400x250?text=Food+Sample';">
                        <div class="history-badge-pos">
                            <span class="badge ${badgeClass}">${labelText}</span>
                        </div>
                    </div>
                    <div class="history-card-body">
                        <div class="history-card-header">
                            <span class="history-card-title">Record #${item.id}</span>
                            <span class="history-card-conf">${item.confidence_percentage}</span>
                        </div>
                        <div class="history-card-date">
                            ${item.created_at_formatted || item.created_at}
                        </div>
                        <p class="history-card-desc">
                            ${item.explanation || 'No description available for this analysis.'}
                        </p>
                        <div class="history-card-footer">
                            <button class="btn btn-outline-danger btn-sm" onclick="window.deleteHistoryRecord(${item.id})">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <polyline points="3 6 5 6 21 6"/>
                                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                </svg>
                                <span>Delete</span>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    // 9. Update Pagination Bar
    function renderPagination(pagination) {
        if (!paginationControls || !pageIndicator) return;

        const totalPages = pagination.total_pages || 1;
        const page = pagination.page || 1;

        if (totalPages <= 1) {
            paginationControls.style.display = 'none';
            return;
        }

        paginationControls.style.display = 'flex';
        pageIndicator.textContent = `Page ${page} of ${totalPages} (${pagination.total_items} records)`;

        if (prevPageBtn) prevPageBtn.disabled = !pagination.has_prev;
        if (nextPageBtn) nextPageBtn.disabled = !pagination.has_next;
    }

    // 10. Delete Single Record Handler
    window.deleteHistoryRecord = function(itemId) {
        if (window.showConfirmModal) {
            window.showConfirmModal({
                title: `Delete Record #${itemId}?`,
                message: `Are you sure you want to delete analysis record #${itemId} and remove its uploaded image file?`,
                confirmText: 'Delete Record',
                onConfirm: async () => {
                    try {
                        const res = await fetch(`/api/history/${itemId}`, { method: 'DELETE' });
                        const data = await res.json();
                        if (data.success) {
                            if (window.showToast) window.showToast(`Record #${itemId} deleted.`, 'success');
                            // Animate removal or refetch
                            const card = document.getElementById(`history-card-${itemId}`);
                            if (card) {
                                card.style.opacity = '0';
                                card.style.transform = 'scale(0.9)';
                                setTimeout(() => fetchHistory(), 200);
                            } else {
                                fetchHistory();
                            }
                        } else {
                            throw new Error(data.message || 'Delete failed');
                        }
                    } catch (err) {
                        console.error('Delete error:', err);
                        if (window.showToast) window.showToast(err.message, 'error');
                    }
                }
            });
        }
    };
});
