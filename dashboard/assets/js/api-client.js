// API Configuration
// Use proxy path for production, or direct URL for local testing
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

    // Defect type별 고정 색상 (범례와 일치)
    defectTypeColors: {
        "missing_hole": "#51cbce",     // primary (cyan)
        "mouse_bite": "#fbc658",       // warning (yellow)
        "open_circuit": "#ef8157",     // danger (orange/red)
        "short": "#6bd098",            // success (green)
        "spur": "#51bcda",             // info (blue)
        "spurious_copper": "#e3e3e3"   // gray
    },

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
        // Initialize Session Trends Chart (Line) - 세션별 통계
        const ctxTrends = document.getElementById("chartDefectTrends").getContext("2d");

        this.charts.trends = new Chart(ctxTrends, {
            type: "line",
            data: {
                labels: [],  // 세션 라벨 (Session #1, #2, ...)
                datasets: [
                    {
                        borderColor: "#ef8157",  // 빨간색 (결함) - 앞에 표시
                        backgroundColor: "rgba(239, 129, 87, 0.3)",
                        pointRadius: 0,  // 동그라미 제거
                        pointHoverRadius: 0,
                        borderWidth: 3,
                        label: "Defects",
                        data: [],
                        fill: true,
                        spanGaps: false
                    },
                    {
                        borderColor: "#6bd098",  // 초록색 (총 검사)
                        backgroundColor: "rgba(107, 208, 152, 0.3)",
                        pointRadius: 0,  // 동그라미 제거
                        pointHoverRadius: 0,
                        borderWidth: 3,
                        label: "Total Inspections",
                        data: [],
                        fill: true,
                        spanGaps: false
                    },
                ],
            },
            options: {
                legend: { display: false },  // 범례 숨김
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
                        backgroundColor: [],  // 동적으로 할당
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

        // Initialize Confidence Chart (Line) - 개선된 디자인
        const ctxConfidence = document.getElementById("chartConfidence").getContext("2d");

        // 그라디언트 생성
        const gradientStroke = ctxConfidence.createLinearGradient(0, 230, 0, 50);
        gradientStroke.addColorStop(1, 'rgba(251, 198, 88, 0.3)');
        gradientStroke.addColorStop(0.4, 'rgba(251, 198, 88, 0.1)');
        gradientStroke.addColorStop(0, 'rgba(251, 198, 88, 0)');

        this.charts.confidence = new Chart(ctxConfidence, {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    {
                        data: [],
                        fill: true,
                        borderColor: "#fbc658",  // 골든 옐로우
                        backgroundColor: gradientStroke,
                        pointBackgroundColor: "#fbc658",
                        pointBorderColor: "#fff",
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointBorderWidth: 3,
                        borderWidth: 3,
                        label: "Avg Confidence",
                        tension: 0.4,  // 부드러운 곡선
                    },
                ],
            },
            options: {
                legend: { display: false },
                layout: {
                    padding: { top: 15, right: 15, left: 15, bottom: 10 }
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            fontColor: "#9f9f9f",
                            beginAtZero: false,
                            min: 0,
                            max: 1,
                            maxTicksLimit: 5,
                            callback: function (value) {
                                return (value * 100).toFixed(0) + '%';  // 퍼센트로 표시
                            }
                        },
                        gridLines: {
                            drawBorder: false,
                            zeroLineColor: "transparent",
                            color: 'rgba(0,0,0,0.05)'
                        }
                    }],
                    xAxes: [{
                        ticks: {
                            fontColor: "#9f9f9f",
                            padding: 10
                        },
                        gridLines: {
                            drawBorder: false,
                            display: false
                        }
                    }]
                },
                tooltips: {
                    enabled: true,
                    callbacks: {
                        label: function (tooltipItem) {
                            return 'Confidence: ' + (tooltipItem.yLabel * 100).toFixed(1) + '%';
                        }
                    }
                }
            },
        });
    },

    startPolling: function () {
        this.updateData();
        this.updateSessionChart();  // 초기 세션 차트 로드
        // 세션 목록도 주기적으로 갱신 (5초마다)
        setInterval(() => this.loadSessions(), 5000);
        setInterval(() => this.updateSessionChart(), 5000);  // 세션 차트도 5초마다
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

            // Update Session Trends Chart
            this.updateSessionChart();
        }

        const defects = await ApiClient.getDefects(sessionId);
        if (defects) {
            this.updateTypesChart(defects);
        }

        // Always fetch latest logs (세션별 필터링)
        const latest = await ApiClient.getLatest(50, sessionId);  // 더 많이 가져와서 defect만 필터링
        if (latest) {
            // defect만 필터링
            const defectsOnly = latest.filter(item => item.result === 'defect');
            this.updateConfidenceChart(defectsOnly.slice(0, 10));  // 최대 10개
        }
    },

    // Session Statistics Chart Update
    updateSessionChart: async function () {
        try {
            // 최근 10개 세션 가져오기
            const sessionsData = await ApiClient.getSessions();
            if (!sessionsData || !sessionsData.sessions) return;

            const sessions = sessionsData.sessions.slice(0, 10).reverse(); // 최근 10개, 오래된 순

            const labels = [];
            const inspections = [];
            const defects = [];

            // 각 세션의 통계 가져오기
            for (const session of sessions) {
                const stats = await ApiClient.getStats(session.id);
                if (stats) {
                    labels.push(`#${session.id}`);
                    inspections.push(stats.total_inspections || 0);
                    defects.push(stats.defect_items || 0);
                }
            }

            // 차트 업데이트 - datasets 순서 변경됨 (defects가 [0], inspections가 [1])
            this.charts.trends.data.labels = labels;
            this.charts.trends.data.datasets[0].data = defects;  // 빨간색 (앞)
            this.charts.trends.data.datasets[1].data = inspections;  // 초록색 (뒤)
            this.charts.trends.update();
        } catch (error) {
            console.error('Error updating session chart:', error);
        }
    },

    updateTypesChart: function (defectsData) {
        // defectsData is like { "missing_hole": 5, "mouse_bite": 2 }
        const labels = Object.keys(defectsData);
        const data = Object.values(defectsData);

        // 각 defect type에 맞는 색상 할당
        const colors = labels.map(defectType => {
            return this.defectTypeColors[defectType] || "#cccccc";  // 기본 회색
        });

        this.charts.types.data.labels = labels;
        this.charts.types.data.datasets[0].data = data;
        this.charts.types.data.datasets[0].backgroundColor = colors;
        this.charts.types.update();
    },

    updateConfidenceChart: function (items) {
        if (!items || items.length === 0) {
            // 데이터가 없으면 차트 비우기
            this.charts.confidence.data.labels = [];
            this.charts.confidence.data.datasets[0].data = [];
            this.charts.confidence.update();
            return;
        }

        // 이미지당 평균 confidence 계산
        const chartData = items.map(item => {
            if (item.detections && item.detections.length > 0) {
                // 여러 detection이 있는 경우 평균 계산
                const totalConfidence = item.detections.reduce(
                    (sum, det) => sum + det.confidence, 0
                );
                const avgConfidence = totalConfidence / item.detections.length;
                return {
                    imageId: item.image_id,
                    confidence: avgConfidence,
                    detectionCount: item.detections.length
                };
            }
            return null;
        }).filter(item => item !== null);

        // 차트 업데이트
        const labels = chartData.map((item, i) => `#${i + 1}`);
        const data = chartData.map(item => item.confidence);

        this.charts.confidence.data.labels = labels;
        this.charts.confidence.data.datasets[0].data = data;
        this.charts.confidence.update();
    }
};

$(document).ready(function () {
    DashboardUpdater.init();
});
