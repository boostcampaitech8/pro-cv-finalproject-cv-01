// 로컬 테스트 시: "http://localhost:8000"
// 배포 시: "/api"
const API_BASE_URL = "/api";

const ApiClient = {
    // 세션 API
    getSessions: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/sessions/`);
            if (!response.ok) throw new Error("Network response was not ok");
            return await response.json();
        } catch (error) {
            console.error("Error fetching sessions:", error);
            return null;
        }
    },

    // 통계 API (session_id 지원)
    getStats: async (sessionId = null) => {
        try {
            let url = `${API_BASE_URL}/stats`;
            if (sessionId) {
                url += `?session_id=${sessionId}`;
            }
            const response = await fetch(url);
            if (!response.ok) throw new Error("Network response was not ok");
            return await response.json();
        } catch (error) {
            console.error("Error fetching stats:", error);
            return null;
        }
    },

    // 결함 집계 API (session_id 지원)
    getDefects: async (sessionId = null) => {
        try {
            let url = `${API_BASE_URL}/defects`;
            if (sessionId) {
                url += `?session_id=${sessionId}`;
            }
            const response = await fetch(url);
            if (!response.ok) throw new Error("Network response was not ok");
            return await response.json();
        } catch (error) {
            console.error("Error fetching defects aggregation:", error);
            return null;
        }
    },

    // 최근 로그 API (session_id 지원)
    getLatest: async (limit = 10, sessionId = null) => {
        try {
            let url = `${API_BASE_URL}/latest?limit=${limit}`;
            if (sessionId) {
                url += `&session_id=${sessionId}`;
            }
            const response = await fetch(url);
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
    selectedSessionId: null,  // 선택된 세션 ID (null = 전체)

    init: function () {
        this.initCharts();
        this.initSessionSelector();
        this.startPolling();
    },

    // 세션 선택기 초기화
    initSessionSelector: function () {
        const selector = document.getElementById("session-select");
        if (!selector) return;

        // 세션 변경 이벤트
        selector.addEventListener("change", (e) => {
            const value = e.target.value;
            this.selectedSessionId = value ? parseInt(value) : null;
            console.log("Session selected:", this.selectedSessionId);

            // 트렌드 차트 리셋 (세션 변경 시)
            this.trendData = Array(24).fill(null);
            this.charts.trends.data.datasets[0].data = this.trendData;
            this.charts.trends.update();

            // 즉시 데이터 업데이트
            this.updateData({ forceChartUpdate: true });
        });

        // 초기 세션 목록 로드
        this.loadSessions();
    },

    // 세션 목록 로드
    loadSessions: async function () {
        const selector = document.getElementById("session-select");
        if (!selector) return;

        const result = await ApiClient.getSessions();
        if (!result || !result.sessions) return;

        // 기존 옵션 제거 (전체 옵션 유지)
        while (selector.options.length > 1) {
            selector.remove(1);
        }

        // 세션 목록 추가
        result.sessions.forEach((session) => {
            const option = document.createElement("option");
            option.value = session.id;

            // 시간 포맷팅
            const startTime = this.formatDateTime(session.started_at);
            const endTime = session.ended_at ? this.formatDateTime(session.ended_at) : "진행중";
            option.textContent = `#${session.id} (${startTime} ~ ${endTime})`;

            selector.appendChild(option);
        });
    },

    // 날짜/시간 포맷팅
    formatDateTime: function (isoString) {
        if (!isoString) return "";
        const date = new Date(isoString);
        const month = String(date.getMonth() + 1).padStart(2, "0");
        const day = String(date.getDate()).padStart(2, "0");
        const hours = String(date.getHours()).padStart(2, "0");
        const minutes = String(date.getMinutes()).padStart(2, "0");
        return `${month}/${day} ${hours}:${minutes}`;
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
        // 세션 목록도 주기적으로 갱신 (5초마다)
        setInterval(() => this.loadSessions(), 5000);
        setInterval(() => this.updateData(), 1000); // Poll every 1 second
    },

    updateData: async function ({ forceChartUpdate = false } = {}) {
        const sessionId = this.selectedSessionId;

        const stats = await ApiClient.getStats(sessionId);
        if (stats) {
            // Update Cards (Real-time)
            // Backend sends: total_inspections, normal_count, defect_items, total_defects, defect_rate
            document.getElementById("total-count").innerText = stats.total_inspections || 0;
            document.getElementById("normal-count").innerText = stats.normal_count || 0;
            document.getElementById("defect-count").innerText = stats.defect_items || 0;

            // Format defect rate logic removed as requested by user (reverted to Model Used)
            // const rate = stats.defect_rate !== undefined ? stats.defect_rate.toFixed(2) : "0.00";
            // document.getElementById("defect-rate").innerText = `${rate}%`;

            // Update Trend Chart (Total Defects)
            this.updateTrendChart(stats.total_defects || 0, forceChartUpdate);
        }

        const defects = await ApiClient.getDefects(sessionId);
        if (defects) {
            this.updateTypesChart(defects);
        }

        // Always fetch latest logs as stats.recent_defects is deprecated in new API
        const latest = await ApiClient.getLatest(10, sessionId);
        if (latest) {
            // Filter only defects for confidence chart, or show latest if no defects found?
            // Usually we want to track confidence of actual defects.
            const defectsOnly = latest.filter(item => item.result === 'defect');
            // If there are no recent defects, show generic latest items (which might be normal)
            // or just show empty if we strictly want Defects.
            // Let's fallback to latest if defectsOnly is empty but typically we want confidence of defects.
            // If latest has no defects, defectsOnly is empty.
            this.updateConfidenceChart(defectsOnly.length > 0 ? defectsOnly : []);
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
        const data = recent.map(item => {
            if (item.detections && item.detections.length > 0) {
                // Use the confidence of the first detection or max
                return item.detections[0].confidence;
            }
            return 0;
        });
        const labels = recent.map((_, i) => `Defect ${i + 1}`); // Simple labels

        this.charts.confidence.data.labels = labels;
        this.charts.confidence.data.datasets[0].data = data;
        this.charts.confidence.update();
    }
};

$(document).ready(function () {
    DashboardUpdater.init();
});
