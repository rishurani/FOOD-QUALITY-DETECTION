/**
 * FreshVision AI - Dashboard Analytics Script
 * Fetches real metrics from /api/dashboard/stats and renders Chart.js visualizations.
 */

document.addEventListener('DOMContentLoaded', () => {
    let pieChartInstance = null;
    let barChartInstance = null;

    const refreshBtn = document.getElementById('refreshDashboardBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            loadDashboardData();
            if (window.showToast) window.showToast('Refreshing dashboard metrics...');
        });
    }

    loadDashboardData();

    async function loadDashboardData() {
        try {
            const response = await fetch('/api/dashboard/stats');
            if (!response.ok) throw new Error('Failed to fetch dashboard statistics');

            const stats = await response.json();
            updateStatCards(stats);
            updateCharts(stats);
            updateRecentTable(stats);
        } catch (err) {
            console.error('Error loading dashboard data:', err);
            if (window.showToast) window.showToast('Error loading live dashboard statistics', 'error');
        }
    }

    function updateStatCards(stats) {
        const total = stats.total_analyses || 0;
        const fresh = stats.fresh_count || 0;
        const spoiled = stats.spoiled_count || 0;
        const uncertain = stats.uncertain_count || 0;

        document.getElementById('statTotal').textContent = total.toLocaleString();
        document.getElementById('statFresh').textContent = fresh.toLocaleString();
        document.getElementById('statSpoiled').textContent = spoiled.toLocaleString();
        document.getElementById('statUncertain').textContent = uncertain.toLocaleString();

        document.getElementById('statAvgConf').textContent = `Avg Confidence: ${stats.average_confidence_percentage || '0.0%'}`;
        document.getElementById('statFreshPct').textContent = `${stats.fresh_percentage || 0}% of total`;
        document.getElementById('statSpoiledPct').textContent = `${stats.spoiled_percentage || 0}% of total`;
    }

    function updateCharts(stats) {
        const total = stats.total_analyses || 0;
        const fresh = stats.fresh_count || 0;
        const spoiled = stats.spoiled_count || 0;
        const uncertain = stats.uncertain_count || 0;

        // 1. Prediction Distribution (Doughnut / Pie Chart)
        const pieCtx = document.getElementById('distributionPieChart').getContext('2d');
        if (pieChartInstance) pieChartInstance.destroy();

        const pieData = total > 0 
            ? [fresh, spoiled, uncertain]
            : [0, 0, 0];

        pieChartInstance = new Chart(pieCtx, {
            type: 'doughnut',
            data: {
                labels: ['Fresh Quality', 'Spoiled Quality', 'Uncertain / Borderline'],
                datasets: [{
                    data: pieData,
                    backgroundColor: ['#10B981', '#EF4444', '#F59E0B'],
                    borderWidth: 2,
                    borderColor: '#FFFFFF',
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { boxWidth: 12, padding: 16, font: { family: 'Inter', size: 12 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                const val = ctx.raw || 0;
                                const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
                                return ` ${ctx.label}: ${val} (${pct}%)`;
                            }
                        }
                    }
                },
                cutout: '65%'
            }
        });

        // 2. Daily Inspection Activity (Bar Chart)
        const barCtx = document.getElementById('activityBarChart').getContext('2d');
        if (barChartInstance) barChartInstance.destroy();

        const daily = stats.daily_activity || [];
        const labels = daily.map(d => d.date);
        const freshCounts = daily.map(d => d.fresh);
        const spoiledCounts = daily.map(d => d.spoiled);

        barChartInstance = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Fresh Produce',
                        data: freshCounts,
                        backgroundColor: '#10B981',
                        borderRadius: 6
                    },
                    {
                        label: 'Spoiled Produce',
                        data: spoiledCounts,
                        backgroundColor: '#EF4444',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        grid: { display: false }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        ticks: { stepSize: 1, precision: 0 }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { boxWidth: 12, padding: 16, font: { family: 'Inter', size: 12 } }
                    }
                }
            }
        });
    }

    function updateRecentTable(stats) {
        const tbody = document.getElementById('recentTableBody');
        const emptyState = document.getElementById('dashboardEmptyState');
        const recent = stats.recent_analyses || [];

        if (!tbody) return;

        if (recent.length === 0) {
            tbody.innerHTML = '';
            if (emptyState) emptyState.style.display = 'block';
            return;
        }

        if (emptyState) emptyState.style.display = 'none';

        tbody.innerHTML = recent.map(item => {
            let badgeClass = 'badge-fresh';
            let labelText = item.predicted_label;

            if (item.is_uncertain) {
                badgeClass = 'badge-uncertain';
                labelText = 'Uncertain';
            } else if (item.predicted_label === 'Spoiled') {
                badgeClass = 'badge-spoiled';
            }

            return `
                <tr>
                    <td>
                        <img src="${item.image_url}" alt="${item.image_filename}" class="table-img-thumb"
                             onerror="this.src='https://placehold.co/48x48?text=Image';">
                    </td>
                    <td>
                        <div style="font-weight: 600; color: var(--text-primary); font-size: 0.85rem;">
                            #${item.id} - ${item.image_filename.substring(0, 16)}...
                        </div>
                    </td>
                    <td>
                        <span class="badge ${badgeClass}">${labelText}</span>
                    </td>
                    <td>
                        <strong style="color: var(--primary);">${item.confidence_percentage}</strong>
                    </td>
                    <td>
                        <span style="font-size: 0.8rem; color: var(--text-muted); font-family: monospace;">
                            ${item.model_version || 'v1.0.0'}
                        </span>
                    </td>
                    <td>
                        <span style="font-size: 0.825rem; color: var(--text-secondary);">
                            ${item.created_at_formatted || item.created_at}
                        </span>
                    </td>
                    <td style="text-align: right;">
                        <a href="/history?search=${item.image_filename}" class="btn btn-secondary btn-sm" title="Inspect Record">
                            View
                        </a>
                    </td>
                </tr>
            `;
        }).join('');
    }
});
