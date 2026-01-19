const API_BASE_URL = "http://3.36.185.146:8000";

const ApiClient = {
    getStats: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/stats`);
            if (!response.ok) throw new Error("Network response was not ok");
            return await response.json();
        } catch (error) {
            console.error("Error fetching stats:", error);
            return null;
        }
    },
    getDefects: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/defects`);
            if (!response.ok) throw new Error("Network response was not ok");
            return await response.json();
        } catch (error) {
            console.error("Error fetching defects aggregation:", error);
            return null;
        }
    },
    getLatest: async (limit = 10) => {
        try {
            const response = await fetch(`${API_BASE_URL}/latest?limit=${limit}`);
            if (!response.ok) throw new Error("Network response was not ok");
            return await response.json();
        } catch (error) {
            console.error("Error fetching latest logs:", error);
            return null;
        }
    },
};

const DashboardUpdater = {
    charts: {},

    init: function () {
        this.initCharts();
        this.startPolling();
    },

    initCharts: function () {
        // Initialize Defect Trends Chart (Line)
        const ctxTrends = document.getElementById("chartDefectTrends").getContext("2d");

        // Generate 00:00 to 23:00 labels
        this.trendLabels = Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`);
        this.trendData = Array(24).fill(null); // start as null/empty

        this.charts.trends = new Chart(ctxTrends, {
            type: "line",
            data: {
                labels: this.trendLabels,
                datasets: [
                    {
                        borderColor: "#6bd098",
                        backgroundColor: "rgba(107, 208, 152, 0.3)",
                        pointRadius: 4,
                        pointHoverRadius: 4,
                        borderWidth: 3,
                        label: "Total Defects",
                        data: this.trendData,
                        fill: true,
                        spanGaps: true // Connect points if there are gaps, or set false to see dots
                    },
                ],
            },
            options: {
                legend: { display: false },
                tooltips: { enabled: true },
                scales: {
                    yAxes: [
                        {
                            ticks: {
                                fontColor: "#9f9f9f",
                                beginAtZero: true,
                                maxTicksLimit: 5,
                            },
                            gridLines: {
                                drawBorder: false,
                                zeroLineColor: "#ccc",
                                color: "rgba(255,255,255,0.05)",
                            },
                        },
                    ],
                    xAxes: [
                        {
                            // barPercentage: 1.6,
                            gridLines: {
                                drawBorder: false,
                                color: "rgba(255,255,255,0.1)",
                                zeroLineColor: "transparent",
                                display: false,
                            },
                            ticks: { padding: 20, fontColor: "#9f9f9f" },
                        },
                    ],
                },
            },
        });

        // Initialize Defect Types Chart (Pie)
        const ctxTypes = document.getElementById("chartEmail").getContext("2d");
        this.charts.types = new Chart(ctxTypes, {
            type: "pie",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Defects",
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        backgroundColor: [
                            "#51cbce",
                            "#fbc658",
                            "#ef8157",
                            "#6bd098",
                            "#51bcda",
                            "#e3e3e3",
                        ],
                        borderWidth: 0,
                        data: [],
                    },
                ],
            },
            options: {
                legend: { display: false },
                pieceLabel: {
                    render: "percentage",
                    fontColor: ["white"],
                    precision: 2,
                },
                tooltips: { enabled: true },
                scales: {
                    yAxes: [
                        {
                            ticks: { display: false },
                            gridLines: {
                                drawBorder: false,
                                zeroLineColor: "transparent",
                                color: "rgba(255,255,255,0.05)",
                            },
                        },
                    ],
                    xAxes: [
                        {
                            barPercentage: 1.6,
                            gridLines: {
                                drawBorder: false,
                                color: "rgba(255,255,255,0.1)",
                                zeroLineColor: "transparent",
                            },
                            ticks: { display: false },
                        },
                    ],
                },
            },
        });

        // Initialize Confidence Chart (Line)
        const ctxConfidence = document.getElementById("chartConfidence").getContext("2d");
        this.charts.confidence = new Chart(ctxConfidence, {
            type: "line",
            hover: false,
            data: {
                labels: [],
                datasets: [
                    {
                        data: [],
                        fill: false,
                        borderColor: "#fbc658",
                        backgroundColor: "transparent",
                        pointBorderColor: "#fbc658",
                        pointRadius: 4,
                        pointHoverRadius: 4,
                        pointBorderWidth: 8,
                        label: "Defect Confidence",
                    },
                ],
            },
            options: {
                legend: { display: false, position: "top" },
                layout: {
                    padding: { top: 15, right: 15, left: 10, bottom: 10 }
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            // suggestedMax: 100 // Scale is 0-100% or 0-1? If user sees 0.92, it's 0-1. 
                            // Wait, user image shows 0.92. So range is 0-1.
                            // BE careful. If I set max 100 it will flatline at bottom.
                            // Let's use suggestedMax 1.05 or just padding.
                            // Actually, let's just stick to padding.
                            // suggestedMax: 1.0 
                        },
                        gridLines: {
                            drawBorder: false,
                            zeroLineColor: "transparent",
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }]
                }
            },
        });
    },

    startPolling: function () {
        this.updateData();
        setInterval(() => this.updateData(), 1000); // Poll every 1 second
    },

    updateData: async function ({ forceChartUpdate = false } = {}) {
        const stats = await ApiClient.getStats();
        if (stats) {
            // Update Cards (Real-time)
            // Backend sends: total_inspections, normal_count, defect_items, total_defects
            document.getElementById("total-count").innerText = stats.total_inspections || 0;
            document.getElementById("normal-count").innerText = stats.normal_count || 0;
            document.getElementById("defect-count").innerText = stats.defect_items || 0;

            // Update Trend Chart (Total Inspections or Total Defects? Using Total Inspections for activity)
            // Or usually "Total Defects"? The label says "Total Defects".
            // If label is "Total Defects", we should use stats.total_defects.
            // Let's check trend label... it says "Total Defects" in initCharts.
            // But previous code passed stats.total_count.
            // If the user wants "Defect Trends", it should be defects count.
            // But usually this chart shows Hourly Output if it's "Activity".
            // Let's stick to what's labeled: "Total Defects" -> stats.total_defects.
            this.updateTrendChart(stats.total_defects || 0, forceChartUpdate);
        }

        const defects = await ApiClient.getDefects();
        if (defects) {
            this.updateTypesChart(defects);
        }

        if (stats && stats.recent_defects) {
            this.updateConfidenceChart(stats.recent_defects);
        } else {
            const latest = await ApiClient.getLatest(10);
            if (latest) {
                const defectsOnly = latest.filter(item => item.result === 'defect');
                this.updateConfidenceChart(defectsOnly.length > 0 ? defectsOnly : latest);
            }
        }
    },

    // Trend Data Management
    trendData: [],
    trendLabels: [],
    // lastTrendHour: null, // Removed as we use direct index mapping now

    updateTrendChart: function (totalDefects, forceUpdate) {
        const now = new Date();
        const currentHour = now.getHours(); // 0-23

        // Update the data point for the current hour
        // Since it's cumulative, we just overwrite the current hour's value with the latest total count
        // We don't touch other hours (they remain null or their previous values)

        // Check if value changed to avoid unnecessary redraws? 
        // Chart.js handles updates efficiently, but let's just update.

        if (this.trendData[currentHour] !== totalDefects) {
            this.trendData[currentHour] = totalDefects;
            this.charts.trends.data.datasets[0].data = this.trendData;
            this.charts.trends.update();

            this.saveTrendData();
        }
    },

    saveTrendData: function () {
        const today = new Date().toDateString(); // e.g., "Sun Jan 18 2026"
        const payload = {
            date: today,
            data: this.trendData
        };
        localStorage.setItem('defectTrends', JSON.stringify(payload));
    },

    loadTrendData: function () {
        const today = new Date().toDateString();
        const stored = localStorage.getItem('defectTrends');

        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                if (parsed.date === today) {
                    this.trendData = parsed.data;
                    return;
                }
            } catch (e) {
                console.error("Failed to parse stored trend data", e);
            }
        }

        // Default / Reset if new day or no data
        this.trendData = Array(24).fill(null);
    },

    // pushTrendData removed as we use fixed 24h array  },

    updateTypesChart: function (defectsData) {
        // defectsData is like { "scratch": 5, "dent": 2 }
        const labels = Object.keys(defectsData);
        const data = Object.values(defectsData);

        this.charts.types.data.labels = labels;
        this.charts.types.data.datasets[0].data = data;
        this.charts.types.update();
    },

    updateConfidenceChart: function (items) {
        // items is array of objects with 'confidence'
        // Take last 10
        const recent = items.slice(0, 10);
        const data = recent.map(item => item.confidence);
        const labels = recent.map((_, i) => `Defect ${i + 1}`); // Simple labels

        this.charts.confidence.data.labels = labels;
        this.charts.confidence.data.datasets[0].data = data;
        this.charts.confidence.update();
    }
};

$(document).ready(function () {
    DashboardUpdater.init();
});
