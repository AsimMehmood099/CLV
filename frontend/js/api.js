const API_BASE = "http://127.0.0.1:8000";

async function getStats(dataset) {
    try {
        const res = await fetch(`${API_BASE}/stats/${dataset}`);
        return await res.json();
    } catch (e) {
        console.error("getStats error:", e);
        return {};
    }
}

async function getSummary(dataset) {
    try {
        const res = await fetch(`${API_BASE}/summary/${dataset}`);
        return await res.json();
    } catch (e) {
        console.error("getSummary error:", e);
        return {};
    }
}

async function getTopCustomers(dataset) {
    try {
        const res = await fetch(`${API_BASE}/customers/${dataset}/top`);
        return await res.json();
    } catch (e) {
        console.error("getTopCustomers error:", e);
        return {};
    }
}

async function getAllCustomers(dataset) {
    try {
        const res = await fetch(`${API_BASE}/customers/${dataset}`);
        return await res.json();
    } catch (e) {
        console.error("getAllCustomers error:", e);
        return {};
    }
}

async function predictCustomer(data) {
    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        return await res.json();
    } catch (e) {
        console.error("predictCustomer error:", e);
        return { error: e.message };
    }
}

async function uploadCSV(dataset, file) {
    try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`${API_BASE}/upload/predict/${dataset}`, {
            method: "POST",
            body: formData
        });
        return await res.json();
    } catch (e) {
        console.error("uploadCSV error:", e);
        return { error: e.message };
    }
}

function getBadgeClass(segment) {
    if (!segment) return "badge-inactive";
    const s = segment.toLowerCase();
    if (s.includes("gold"))     return "badge-gold";
    if (s.includes("loyal"))    return "badge-loyal";
    if (s.includes("requires")) return "badge-requires";
    return "badge-inactive";
}

function formatCurrency(val) {
    return "$" + parseFloat(val).toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}