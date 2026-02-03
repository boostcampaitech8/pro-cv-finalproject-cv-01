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
            if (sessionId) url += `&session_id=${sessionId}`;
            const response = await fetch(url);
            if (!response.ok) throw new Error("Network response was not ok");
            return await response.json();
        } catch (error) {
            console.error("Error fetching latest logs:", error);
            return null;
        }
    },
    // 모니터링 상세 API
    getHealth: async (sessionId = null) => {
        try {
            let url = `${API_BASE_URL}/monitoring/health`;
            if (sessionId) url += `?session_id=${sessionId}`;
            const response = await fetch(url);
            return await response.json();
        } catch (error) {
            console.error("Error fetching health:", error);
            return null;
        }
    },
    // 알림 요약 API (배너용)
    getAlerts: async (sessionId = null) => {
        try {
            let url = `${API_BASE_URL}/monitoring/alerts`;
            if (sessionId) url += `?session_id=${sessionId}`;
            const response = await fetch(url);
            return await response.json();
        } catch (error) {
            console.error("Error fetching alerts:", error);
            return null;
        }
    }
};

const DashboardUpdater = {
    charts: {},
    selectedSessionId: null,  // 선택된 세션 ID (null = 전체)
    dismissedAlertSession: undefined, // 알림을 끈 세션 ID 저장

    // Defect type별 고정 색상 (범례와 일치)
    defectTypeColors: {
        "Missing Hole": "#51cbce",     // primary (cyan)
        "Mouse Bite": "#fbc658",       // warning (yellow)
        "Open Circuit": "#ef8157",     // danger (orange/red)
        "Short": "#6bd098",            // success (green)
        "Spur": "#51bcda",             // info (blue)
        "Spurious Copper": "#e3e3e3"   // gray
    },

    init: function () {
        // 페이지 종류 확인 (Health 페이지인지 체크)
        this.isHealthPage = window.location.pathname.includes('health.html');

        // 세션 ID 복원
        const savedSession = sessionStorage.getItem("selectedSessionId");
        if (savedSession !== null) {
            this.selectedSessionId = savedSession === "null" ? null : parseInt(savedSession);
        }

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

            // 두 저장소 모두 업데이트 (호환성 유지)
            sessionStorage.setItem("selectedSessionId", this.selectedSessionId);
            if (this.selectedSessionId) {
                localStorage.setItem("dashboard_selected_session", this.selectedSessionId);
            } else {
                localStorage.removeItem("dashboard_selected_session");
            }

            this.dismissedAlertSession = undefined; // 세션 변경 시 알림 초기화
            console.log("Session selected:", this.selectedSessionId);

            // 트렌드 차트 리셋 (세션 변경 시)
            if (this.charts.trends) {
                this.trendData = Array(24).fill(null);
                this.charts.trends.data.datasets[0].data = this.trendData;
                this.charts.trends.update();
            }

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

            // 저장된 세션과 일치하면 선택 상태로
            if (this.selectedSessionId === session.id) {
                option.selected = true;
            }

            // 시간 포맷팅
            const startTime = this.formatDateTime(session.started_at);
            const endTime = session.ended_at ? this.formatDateTime(session.ended_at) : "진행중";
            option.textContent = `#${session.id} (${startTime} ~ ${endTime})`;

            selector.appendChild(option);
        });

        // "전체" 옵션 선택 처리 + 유효성 검증 (dev 로직 통합)
        if (this.selectedSessionId === null) {
            selector.options[0].selected = true;
        } else {
            // 저장된 세션이 실제 목록에 존재하는지 검증
            const exists = Array.from(selector.options).some(opt => opt.value == this.selectedSessionId);
            if (!exists) {
                // 목록에 없는 세션이면 "전체"로 복구하고 저장소 정리
                selector.options[0].selected = true;
                this.selectedSessionId = null;
                localStorage.removeItem("dashboard_selected_session");
                sessionStorage.removeItem("selectedSessionId");
                console.log("Invalid session removed, reset to All Sessions");
            }
        }
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
        const elTrends = document.getElementById("chartDefectTrends");
        if (elTrends) {
            const ctxTrends = elTrends.getContext("2d");
            this.charts.trends = new Chart(ctxTrends, {
                type: "line",
                data: {
                    labels: [], datasets: [
                        { borderColor: "#ef8157", backgroundColor: "rgba(239, 129, 87, 0.3)", pointRadius: 0, borderWidth: 3, label: "Defects", data: [], fill: true },
                        { borderColor: "#6bd098", backgroundColor: "rgba(107, 208, 152, 0.3)", pointRadius: 0, borderWidth: 3, label: "Total Inspections", data: [], fill: true }
                    ],
                },
                options: {
                    legend: { display: false },
                    scales: {
                        yAxes: [{ ticks: { fontColor: "#9f9f9f", beginAtZero: true, maxTicksLimit: 5 }, gridLines: { drawBorder: false, color: "rgba(255,255,255,0.05)" } }],
                        xAxes: [{ gridLines: { display: false }, ticks: { padding: 20, fontColor: "#9f9f9f" } }]
                    }
                }
            });
        }

        // Initialize Defect Types Chart (Pie)
        const elEmail = document.getElementById("chartEmail");
        if (elEmail) {
            const ctxTypes = elEmail.getContext("2d");
            this.charts.types = new Chart(ctxTypes, {
                type: "pie",
                data: { labels: [], datasets: [{ label: "Defects", backgroundColor: [], borderWidth: 0, data: [] }] },
                options: { legend: { display: false }, pieceLabel: { render: "percentage", fontColor: ["white"] }, tooltips: { enabled: true } }
            });
        }

        // Initialize Confidence Chart (Line)
        const elConfidence = document.getElementById("chartConfidence");
        if (elConfidence) {
            const ctxConfidence = elConfidence.getContext("2d");
            const gradientStroke = ctxConfidence.createLinearGradient(0, 230, 0, 50);
            gradientStroke.addColorStop(1, 'rgba(251, 198, 88, 0.3)'); gradientStroke.addColorStop(0, 'rgba(251, 198, 88, 0)');
            this.charts.confidence = new Chart(ctxConfidence, {
                type: "line",
                data: { labels: [], datasets: [{ data: [], fill: true, borderColor: "#fbc658", backgroundColor: gradientStroke, pointRadius: 6, borderWidth: 3, label: "Avg Confidence", tension: 0.4 }] },
                options: {
                    legend: { display: false },
                    scales: {
                        yAxes: [{ ticks: { fontColor: "#9f9f9f", beginAtZero: false, min: 0, max: 1, callback: (v) => (v * 100).toFixed(0) + '%' } }],
                        xAxes: [{ ticks: { fontColor: "#9f9f9f" }, gridLines: { display: false } }]
                    }
                }
            });
        }

        // Health Page Charts
        const elConfDist = document.getElementById("chartConfidenceDist");
        if (elConfDist) {
            this.charts.confDist = new Chart(elConfDist.getContext("2d"), {
                type: "doughnut",
                data: {
                    labels: ["High (>=90%)", "Med (80~90%)", "Low (70~80%)", "Critical (<70%)"],
                    datasets: [{
                        backgroundColor: ["#6bd098", "#fbc658", "#ef8157", "#c0c0c0"],
                        data: [0, 0, 0, 0]
                    }]
                },
                options: { legend: { position: 'bottom' } }
            });
        }

        const elTypeDist = document.getElementById("chartDefectTypesDist");
        if (elTypeDist) {
            this.charts.typeDist = new Chart(elTypeDist.getContext("2d"), {
                type: "horizontalBar",
                data: { labels: [], datasets: [{ label: "Count", backgroundColor: "#51bcda", data: [] }] },
                options: {
                    legend: { display: false },
                    scales: { xAxes: [{ ticks: { beginAtZero: true } }] }
                }
            });
        }
    },

    startPolling: function () {
        this.updateData();
        this.updateSessionChart();  // 초기 세션 차트 로드

        // Health 페이지가 아닐 때만 초기 배너 로드 (Health는 updateData 내부에서 처리)
        if (!this.isHealthPage) {
            this.updateAlertBanner();
        }

        // 세션 목록도 주기적으로 갱신 (5초마다)
        setInterval(() => this.loadSessions(), 5000);
        setInterval(() => this.updateSessionChart(), 5000);  // 세션 차트도 5초마다

        // 실시간 업데이트
        setInterval(() => {
            this.updateData();
            // Health 페이지가 아닐 때만 배너 별도 업데이트
            if (!this.isHealthPage) {
                this.updateAlertBanner();
            }
        }, 1000);
    },

    // 공통 알림 배너 업데이트
    updateAlertBanner: async function () {
        const container = document.getElementById("alert-banner-container");
        if (!container) return;

        // 현재 세션에서 이미 알림을 껐다면 표시하지 않음
        if (this.dismissedAlertSession === this.selectedSessionId) return;

        // Health 페이지에서는 이미 로드된 healthData 사용 (일관성 보장)
        let alerts = [];
        if (this.isHealthPage && this.cachedHealthData && this.cachedHealthData.alerts) {
            alerts = this.cachedHealthData.alerts;
        } else {
            // Dashboard 페이지에서는 별도 API 호출
            const data = await ApiClient.getAlerts(this.selectedSessionId);
            if (data && data.alerts) {
                alerts = data.alerts;
            }
        }

        if (alerts.length === 0) {
            container.innerHTML = "";
            return;
        }

        // 가장 심각한 알림 하나만 표시
        const mainAlert = alerts[0];
        const levelClass = mainAlert.level === 'critical' ? 'alert-danger' : 'alert-warning';
        const icon = mainAlert.level === 'critical' ? 'nc-bell-55' : 'nc-alert-circle-i';

        container.innerHTML = `
            <div class="alert ${levelClass} alert-dismissible fade show" role="alert" style="margin-bottom: 0; padding-left: 50px;">
                <i class="nc-icon ${icon}" style="position: absolute; left: 15px; top: 50%; transform: translateY(-50%); font-size: 20px;"></i>
                <span>
                    <b>[${mainAlert.level.toUpperCase()}]</b> ${mainAlert.message}
                </span>
                <button type="button" class="close" data-dismiss="alert" aria-label="Close" onclick="DashboardUpdater.dismissAlert()">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
        `;
    },

    // 알림 끄기 처리
    dismissAlert: function () {
        this.dismissedAlertSession = this.selectedSessionId;
        console.log("Alert dismissed for session:", this.dismissedAlertSession);
    },

    updateData: async function ({ forceChartUpdate = false } = {}) {
        const sessionId = this.selectedSessionId;

        // 대시보드 메인 카드 업데이트
        if (document.getElementById("total-count")) {
            const stats = await ApiClient.getStats(sessionId);
            if (stats) {
                document.getElementById("total-count").innerText = stats.total_inspections || 0;
                document.getElementById("normal-count").innerText = stats.normal_count || 0;
                document.getElementById("defect-count").innerText = stats.defect_items || 0;
                this.updateSessionChart();
            }
        }

        // Health 페이지인 경우 상세 지표 업데이트
        if (this.isHealthPage) {
            this.updateHealthMetrics(sessionId);
        }

        const defects = await ApiClient.getDefects(sessionId);
        if (defects && this.charts.types) {
            this.updateTypesChart(defects);
        }

        // Always fetch latest logs (세션별 필터링)
        const latest = await ApiClient.getLatest(50, sessionId);
        if (latest && this.charts.confidence) {
            const defectsOnly = latest.filter(item => item.result === 'defect');
            this.updateConfidenceChart(defectsOnly.slice(0, 10));
        }
    },

    // Health 페이지 전용 업데이트 로직
    updateHealthMetrics: async function (sessionId) {
        const data = await ApiClient.getHealth(sessionId);
        if (!data) return;

        // Health 데이터 캐시 (배너와 일관성 유지)
        this.cachedHealthData = data;

        // 상단 배지 및 상태
        const statusBadge = document.getElementById("system-status-badge");
        if (statusBadge) {
            statusBadge.innerText = data.status.toUpperCase();
            statusBadge.className = `badge badge-${data.status === 'healthy' ? 'success' : (data.status === 'warning' ? 'warning' : 'danger')}`;
        }

        const timestamp = document.getElementById("health-timestamp");
        if (timestamp) timestamp.innerText = `Last updated: ${new Date(data.timestamp).toLocaleTimeString()}`;

        // 수치 업데이트
        if (document.getElementById("h-defect-rate")) document.getElementById("h-defect-rate").innerText = `${data.defect_rate.toFixed(1)}%`;
        if (document.getElementById("h-avg-confidence")) {
            const conf = data.defect_confidence_stats ? data.defect_confidence_stats.avg_confidence : 0;
            document.getElementById("h-avg-confidence").innerText = conf.toFixed(2);
        }
        if (document.getElementById("h-low-conf-ratio")) document.getElementById("h-low-conf-ratio").innerText = `${data.low_confidence_ratio.toFixed(1)}%`;
        if (document.getElementById("h-avg-defects")) document.getElementById("h-avg-defects").innerText = data.avg_defects_per_item.toFixed(1);

        // 알림 테이블
        const tableBody = document.getElementById("alerts-table-body");
        if (tableBody) {
            if (data.alerts.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="3" class="text-center">No active alerts</td></tr>';
            } else {
                tableBody.innerHTML = data.alerts.map(alert => `
                    <tr>
                        <td><span class="badge badge-${alert.level === 'critical' ? 'danger' : 'warning'}">${alert.level.toUpperCase()}</span></td>
                        <td>${alert.message}</td>
                        <td>${alert.action}</td>
                    </tr>
                `).join('');
            }
        }

        // 결함 타입 신뢰도 리스트
        const typeList = document.getElementById("defect-type-list");
        if (typeList && data.defect_type_stats) {
            typeList.innerHTML = data.defect_type_stats.map(type => `
                <li class="mb-2">
                    <div class="d-flex justify-content-between">
                        <span>${type.defect_type} (${type.count}건)</span>
                        <span class="text-primary font-weight-bold">${(type.avg_confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar" role="progressbar" style="width: ${type.avg_confidence * 100}%"></div>
                    </div>
                </li>
            `).join('');
        }

        // 신뢰도 분포 차트 업데이트
        if (this.charts.confDist && data.defect_confidence_stats) {
            const dist = data.defect_confidence_stats.distribution;
            this.charts.confDist.data.datasets[0].data = [dist.high, dist.medium, dist.low, dist.very_low];
            this.charts.confDist.update();
        }

        // 결함 타입 분포 차트 업데이트
        if (this.charts.typeDist && data.defect_type_stats) {
            const labels = data.defect_type_stats.map(t => t.defect_type);
            const counts = data.defect_type_stats.map(t => t.count);
            this.charts.typeDist.data.labels = labels;
            this.charts.typeDist.data.datasets[0].data = counts;
            this.charts.typeDist.update();
        }

        // Health 데이터 로드 후 배너 업데이트 (일관성 보장)
        this.updateAlertBanner();
    },

    // Session Statistics Chart Update
    updateSessionChart: async function () {
        // Health 페이지에는 trends 차트가 없으므로 체크
        if (!this.charts.trends) return;

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
