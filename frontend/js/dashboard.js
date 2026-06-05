let segmentChart = null;
let clvChart = null;

async function loadDashboard() {
    const dataset = document.getElementById("datasetSelect").value;

    // Update time
    document.getElementById("lastUpdated").textContent =
        "Last updated: " + new Date().toLocaleTimeString();

    // Load stats
    try {
        const stats = await getStats(dataset);
        document.getElementById("totalCustomers").textContent =
            stats.total_customers?.toLocaleString() || "0";
        document.getElementById("avgClv").textContent =
            formatCurrency(stats.avg_clv || 0);
        document.getElementById("maxClv").textContent =
            formatCurrency(stats.max_clv || 0);
        document.getElementById("totalRevenue").textContent =
            formatCurrency(stats.total_revenue || 0);
    } catch (e) {
        console.error("Stats error:", e);
    }

    // Load segment summary for charts
    try {
        const summaryData = await getSummary(dataset);
        const segments    = summaryData.summary || [];

        const labels   = segments.map(s => s._id || "Unknown");
        const counts   = segments.map(s => s.count || 0);
        const avgClvs  = segments.map(s => parseFloat((s.avg_clv || 0).toFixed(2)));

        const colors = [
            "#7c6af7", "#34d399", "#fbbf24", "#ef4444"
        ];

        // Segment Donut Chart
       if (segmentChart) segmentChart.destroy();
const ctx1 = document.getElementById("segmentChart").getContext("2d");
segmentChart = new Chart(ctx1, {
    type: "doughnut",
    data: {
        labels: labels,
        datasets: [{
            data: counts,
            backgroundColor: colors,
            borderWidth: 0,
            hoverOffset: 8
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: "bottom",
                labels: { 
                    color: "#9ca3af", 
                    padding: 16,
                    generateLabels: function(chart) {
                        const data = chart.data;
                        const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                        return data.labels.map((label, i) => ({
                            text: `${label}: ${((data.datasets[0].data[i] / total) * 100).toFixed(1)}%`,
                            fillStyle: data.datasets[0].backgroundColor[i],
                            index: i
                        }));
                    }
                }
            },
            datalabels: {
                color: "#fff",
                font: { weight: "bold", size: 13 },
                formatter: function(value, context) {
                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                    const pct = ((value / total) * 100).toFixed(1);
                    return pct + "%";
                },
                display: function(context) {
                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                    const pct = (context.dataset.data[context.dataIndex] / total) * 100;
                    return pct > 3;
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const pct = ((context.parsed / total) * 100).toFixed(1);
                        return ` ${context.label}: ${context.parsed} (${pct}%)`;
                    }
                }
            }
        }
    },
    plugins: [ChartDataLabels]
});

        // CLV Bar Chart
        if (clvChart) clvChart.destroy();
        const ctx2 = document.getElementById("clvChart").getContext("2d");
        clvChart = new Chart(ctx2, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Avg CLV ($)",
                    data: avgClvs,
                    backgroundColor: colors,
                    borderRadius: 8,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        ticks: { color: "#9ca3af" },
                        grid:  { color: "#2a2d3e" }
                    },
                    y: {
                        ticks: { color: "#9ca3af" },
                        grid:  { color: "#2a2d3e" }
                    }
                }
            }
        });

    } catch (e) {
        console.error("Summary error:", e);
    }

    // Load top customers table
    try {
        const topData  = await getTopCustomers(dataset);
        const customers = topData.customers || [];
        const tbody    = document.getElementById("topTableBody");
        tbody.innerHTML = "";

        customers.forEach((c, i) => {
            const badge = getBadgeClass(c.segment);
            tbody.innerHTML += `
                <tr>
                    <td>${i + 1}</td>
                    <td>${c.customer_id}</td>
                    <td style="color:#7c6af7;font-weight:700">
                        ${formatCurrency(c.predicted_clv)}
                    </td>
                    <td><span class="badge ${badge}">${c.segment}</span></td>
                    <td>${c.frequency}</td>
                    <td>${formatCurrency(c.monetary_value)}</td>
                </tr>
            `;
        });

        if (customers.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align:center;color:#6b7280;padding:32px">
                        No customers yet — upload a CSV or make predictions first!
                    </td>
                </tr>
            `;
        }

    } catch (e) {
        console.error("Top customers error:", e);
    }
}

// Load on page ready
window.addEventListener("load", loadDashboard);