import aiosqlite
import json
from typing import List, Dict, Optional
from datetime import datetime
from schemas.schemas import (
    DetectRequest,
    StatsResponse,
    InspectionLogResponse,
    HealthResponse,
    SessionInfo,
    DefectConfidenceStats,
    ConfidenceDistribution,
    DefectTypeStat,
    AlertInfo
)
from config.settings import DB_PATH, ALERT_THRESHOLDS
from fastapi import HTTPException


async def init_db():
    """
    Initialize database and create tables if they don't exist.
    Called on server startup.
    Schema Update: 1 row per image. 'detections' column stores JSON list of defects.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        # 세션 테이블 생성
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT
            )
        """)

        # 검사 로그 테이블 생성
        await db.execute("""
            CREATE TABLE IF NOT EXISTS inspection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_id TEXT NOT NULL,
                result TEXT NOT NULL,
                detections TEXT,
                image_path TEXT,
                session_id INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # 기존 테이블에 session_id 컬럼이 없으면 추가 (마이그레이션)
        try:
            await db.execute("ALTER TABLE inspection_logs ADD COLUMN session_id INTEGER")
        except Exception:
            # 컬럼이 이미 존재하면 무시
            pass

        # 피드백 테이블 생성
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                correct_label TEXT,
                comment TEXT,
                created_at TEXT,
                created_by TEXT,
                status TEXT DEFAULT 'pending', -- pending, resolved
                FOREIGN KEY (log_id) REFERENCES inspection_logs (id)
            )
        """)

        # 피드백 테이블 인덱스 생성
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_log_id ON feedback(log_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at)"
        )

        await db.commit()


async def close_db():
    """
    Close database connections.
    Called on server shutdown.
    """
    # aiosqlite는 context manager 사용하므로 별도 정리 불필요
    # 이 함수는 향후 확장을 위해 존재
    pass


async def add_inspection_log(data: DetectRequest, image_path: Optional[str] = None) -> List[int]:
    """
    Save inspection result as a single row.
    'detections' list is serialized to JSON string.
    Returns: List containing the single inserted row ID.
    """
    async with aiosqlite.connect(DB_PATH) as db:

        # Serialize detections list to JSON string
        # data.detections is a list of Pydantic models, so we dump them to dicts
        detections_json = json.dumps([d.model_dump() for d in data.detections])

        result_status = "defect" if len(data.detections) > 0 else "normal"

        cursor = await db.execute(
            """
            INSERT INTO inspection_logs (timestamp, image_id, result, detections, image_path, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                data.timestamp,
                data.image_id,
                result_status,
                detections_json,
                image_path,
                data.session_id
            )
        )
        await db.commit()

        return [cursor.lastrowid]


async def get_stats(session_id: Optional[int] = None) -> StatsResponse:
    """
    Calculate statistics from the logs.
    Now requires parsing 'detections' column to count total defects.
    session_id가 제공되면 해당 세션의 통계만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 세션 필터 조건
        session_filter = ""
        session_params = ()
        if session_id is not None:
            session_filter = " WHERE session_id = ?"
            session_params = (session_id,)

        # 1. Total Inspections (Total Rows)
        cursor = await db.execute(
            f"SELECT COUNT(*) as cnt FROM inspection_logs{session_filter}",
            session_params
        )
        row = await cursor.fetchone()
        total_inspections = row["cnt"]

        if total_inspections == 0:
            return StatsResponse(
                total_inspections=0,
                normal_count=0,
                defect_items=0,
                total_defects=0,
                defect_rate=0.0,
                avg_defects_per_item=0.0,
                avg_fps=0.0,
                last_defect=None
            )

        # 2. Defect Items (Rows where result='defect')
        defect_filter = " WHERE result='defect'" if not session_filter else " WHERE result='defect' AND session_id = ?"
        defect_params = () if not session_filter else (session_id,)
        cursor = await db.execute(
            f"SELECT COUNT(*) as cnt FROM inspection_logs{defect_filter}",
            defect_params
        )
        row = await cursor.fetchone()
        defect_items = row["cnt"]

        # 3. Total Defects (Sum of length of detections list in each defect row)
        # SQLite JSON extension might not be enabled, so we fetch and sum in Python.
        cursor = await db.execute(
            f"SELECT detections FROM inspection_logs{defect_filter}",
            defect_params
        )
        rows = await cursor.fetchall()

        total_defects = 0
        for r in rows:
            try:
                dets = json.loads(r["detections"])
                total_defects += len(dets)
            except:
                pass

        normal_count = total_inspections - defect_items
        defect_rate = (defect_items / total_inspections) * 100
        avg_defects_per_item = total_defects / defect_items if defect_items > 0 else 0.0
        avg_fps = 0.0

        # 4. Last Defect Log
        cursor = await db.execute(
            f"SELECT * FROM inspection_logs{defect_filter} ORDER BY id DESC LIMIT 1",
            defect_params
        )
        row = await cursor.fetchone()

        last_defect_log = None
        if row:
            # Map DB row to InspectionLogResponse
            last_defect_log = InspectionLogResponse(
                id=row["id"],
                timestamp=row["timestamp"],
                image_id=row["image_id"],
                result=row["result"],
                detections=json.loads(row["detections"]) if row["detections"] else [],
                image_path=row["image_path"],
                session_id=row["session_id"]
            )

        return StatsResponse(
            total_inspections=total_inspections,
            normal_count=normal_count,
            defect_items=defect_items,
            total_defects=total_defects,
            defect_rate=round(defect_rate, 2),
            avg_defects_per_item=round(avg_defects_per_item, 2),
            avg_fps=avg_fps,
            last_defect=last_defect_log
        )


async def get_recent_logs(limit: int = 10, session_id: Optional[int] = None) -> List[Dict]:
    """
    Returns the N most recent logs.
    session_id가 제공되면 해당 세션의 로그만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if session_id is not None:
            cursor = await db.execute(
                "SELECT * FROM inspection_logs WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM inspection_logs ORDER BY id DESC LIMIT ?",
                (limit,)
            )
        rows = await cursor.fetchall()

        result = []
        for row in rows:
            record = dict(row)
            if record["detections"]:
                record["detections"] = json.loads(record["detections"])
            else:
                record["detections"] = []
            result.append(record)

        return result


async def get_defect_logs(session_id: Optional[int] = None) -> List[Dict]:
    """
    Returns all logs that are defects.
    session_id가 제공되면 해당 세션의 결함 로그만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if session_id is not None:
            cursor = await db.execute(
                "SELECT * FROM inspection_logs WHERE result = 'defect' AND session_id = ? ORDER BY id DESC",
                (session_id,)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM inspection_logs WHERE result = 'defect' ORDER BY id DESC"
            )
        rows = await cursor.fetchall()

        result = []
        for row in rows:
            record = dict(row)
            if record["detections"]:
                record["detections"] = json.loads(record["detections"])
            else:
                record["detections"] = []
            result.append(record)

        return result


# ===== 세션 관리 함수 =====

async def create_session() -> Dict:
    """
    새 세션을 생성하고 ID와 시작 시간을 반환.
    """
    from datetime import datetime
    started_at = datetime.now().isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO sessions (started_at) VALUES (?)",
            (started_at,)
        )
        await db.commit()
        session_id = cursor.lastrowid

    return {"id": session_id, "started_at": started_at, "ended_at": None}


async def end_session(session_id: int) -> Optional[Dict]:
    """
    세션을 종료하고 ended_at을 설정.
    """
    from datetime import datetime
    ended_at = datetime.now().isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 세션 존재 여부 확인
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        # ended_at 업데이트
        await db.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (ended_at, session_id)
        )
        await db.commit()

        return {
            "id": session_id,
            "started_at": row["started_at"],
            "ended_at": ended_at
        }


async def get_sessions() -> List[Dict]:
    """
    모든 세션 목록을 반환 (최신순).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions ORDER BY id DESC"
        )
        rows = await cursor.fetchall()

        return [dict(row) for row in rows]


async def get_session(session_id: int) -> Optional[Dict]:
    """
    특정 세션 정보를 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()

        if row:
            return dict(row)
        return None


# ===== Health Monitoring 관련 함수 =====

async def resolve_session_filter(session_id: Optional[str]) -> Optional[int]:
    """
    세션 필터를 처리하여 실제 세션 ID를 반환.
    - None → None (전체 데이터)
    - "latest" → 진행 중인 세션 ID
    - 숫자 문자열 → 검증 후 int 반환
    """
    if session_id is None:
        return None

    if session_id == "latest":
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id FROM sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="진행 중인 세션이 없습니다")
            return row["id"]

    # 숫자 문자열 변환 및 검증
    try:
        sid = int(session_id)
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT id FROM sessions WHERE id = ?", (sid,))
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"세션 ID {sid}를 찾을 수 없습니다")
            return sid
    except ValueError:
        raise HTTPException(status_code=400, detail=f"잘못된 session_id 형식: {session_id}")


async def get_confidence_distribution(session_id: Optional[int]) -> ConfidenceDistribution:
    """
    결함의 신뢰도 분포를 계산.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        query = "SELECT detections FROM inspection_logs WHERE result='defect'"
        params = ()
        if session_id is not None:
            query += " AND session_id=?"
            params = (session_id,)

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        high = medium = low = very_low = 0

        for row in rows:
            if row["detections"]:
                try:
                    detections = json.loads(row["detections"])
                    for det in detections:
                        conf = det.get("confidence", 0)
                        if conf >= 0.9:
                            high += 1
                        elif conf >= 0.8:
                            medium += 1
                        elif conf >= 0.7:
                            low += 1
                        else:
                            very_low += 1
                except:
                    pass

        return ConfidenceDistribution(high=high, medium=medium, low=low, very_low=very_low)


async def get_defect_confidence_stats(session_id: Optional[int]) -> Optional[DefectConfidenceStats]:
    """
    결함 신뢰도 통계 계산 (평균, 최소, 최대, 분포).
    결함이 없으면 None 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        query = "SELECT detections FROM inspection_logs WHERE result='defect'"
        params = ()
        if session_id is not None:
            query += " AND session_id=?"
            params = (session_id,)

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        confidences = []

        for row in rows:
            if row["detections"]:
                try:
                    detections = json.loads(row["detections"])
                    for det in detections:
                        conf = det.get("confidence", 0)
                        if conf > 0:
                            confidences.append(conf)
                except:
                    pass

        if not confidences:
            return None

        distribution = await get_confidence_distribution(session_id)

        return DefectConfidenceStats(
            avg_confidence=round(sum(confidences) / len(confidences), 3),
            min_confidence=round(min(confidences), 3),
            max_confidence=round(max(confidences), 3),
            distribution=distribution
        )


async def get_defect_type_stats(session_id: Optional[int]) -> List[DefectTypeStat]:
    """
    결함 타입별 통계 (개수, 평균 신뢰도).
    개수 내림차순 정렬.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        query = "SELECT detections FROM inspection_logs WHERE result='defect'"
        params = ()
        if session_id is not None:
            query += " AND session_id=?"
            params = (session_id,)

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        # 타입별 집계
        type_data = {}

        for row in rows:
            if row["detections"]:
                try:
                    detections = json.loads(row["detections"])
                    for det in detections:
                        defect_type = det.get("defect_type", "unknown")
                        conf = det.get("confidence", 0)

                        if defect_type not in type_data:
                            type_data[defect_type] = {"count": 0, "total_conf": 0.0}

                        type_data[defect_type]["count"] += 1
                        type_data[defect_type]["total_conf"] += conf
                except:
                    pass

        # DefectTypeStat 리스트로 변환 및 정렬
        result = []
        for defect_type, data in type_data.items():
            avg_conf = data["total_conf"] / data["count"] if data["count"] > 0 else 0.0
            result.append(DefectTypeStat(
                defect_type=defect_type,
                count=data["count"],
                avg_confidence=round(avg_conf, 3)
            ))

        # 개수 내림차순 정렬
        result.sort(key=lambda x: x.count, reverse=True)

        return result


async def get_session_info(session_id: Optional[int]) -> SessionInfo:
    """
    세션 정보 및 진행 시간 계산.
    """
    if session_id is None:
        return SessionInfo(
            id=None,
            started_at=None,
            ended_at=None,
            duration_seconds=None,
            is_active=False
        )

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"세션 ID {session_id}를 찾을 수 없습니다")

        started_at = row["started_at"]
        ended_at = row["ended_at"]
        is_active = ended_at is None

        # duration 계산
        if started_at:
            start_time = datetime.fromisoformat(started_at)
            end_time = datetime.fromisoformat(ended_at) if ended_at else datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
        else:
            duration_seconds = None

        return SessionInfo(
            id=session_id,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=round(duration_seconds, 1) if duration_seconds else None,
            is_active=is_active
        )


def generate_alerts(
    defect_rate: float,
    avg_defects_per_item: float,
    defect_confidence_stats: Optional[DefectConfidenceStats]
) -> List[AlertInfo]:
    """
    임계값 기반 알림 생성.
    """
    alerts = []

    # 1. 불량률 체크
    if defect_rate >= ALERT_THRESHOLDS["defect_rate_critical"]:
        alerts.append(AlertInfo(
            level="critical",
            message=f"불량률이 {defect_rate:.1f}%로 매우 높습니다",
            value=defect_rate,
            threshold=ALERT_THRESHOLDS["defect_rate_critical"],
            action="생산 라인 점검 및 원인 분석 필요"
        ))
    elif defect_rate >= ALERT_THRESHOLDS["defect_rate_warning"]:
        alerts.append(AlertInfo(
            level="warning",
            message=f"불량률이 {defect_rate:.1f}%로 높습니다",
            value=defect_rate,
            threshold=ALERT_THRESHOLDS["defect_rate_warning"],
            action="품질 모니터링 강화 권장"
        ))

    # 2. 평균 신뢰도 체크 (결함이 있는 경우만)
    if defect_confidence_stats:
        avg_conf = defect_confidence_stats.avg_confidence

        if avg_conf < ALERT_THRESHOLDS["avg_confidence_critical"]:
            alerts.append(AlertInfo(
                level="critical",
                message=f"평균 신뢰도가 {avg_conf:.2f}로 매우 낮습니다",
                value=avg_conf,
                threshold=ALERT_THRESHOLDS["avg_confidence_critical"],
                action="모델 재학습 또는 임계값 조정 필요"
            ))
        elif avg_conf < ALERT_THRESHOLDS["avg_confidence_warning"]:
            alerts.append(AlertInfo(
                level="warning",
                message=f"평균 신뢰도가 {avg_conf:.2f}로 낮습니다",
                value=avg_conf,
                threshold=ALERT_THRESHOLDS["avg_confidence_warning"],
                action="검출 정확도 모니터링 권장"
            ))

        # 3. 낮은 신뢰도 비율 체크
        dist = defect_confidence_stats.distribution
        total_detections = dist.high + dist.medium + dist.low + dist.very_low

        if total_detections > 0:
            low_ratio = ((dist.low + dist.very_low) / total_detections) * 100

            if low_ratio >= ALERT_THRESHOLDS["low_confidence_ratio_critical"]:
                alerts.append(AlertInfo(
                    level="critical",
                    message=f"낮은 신뢰도 비율이 {low_ratio:.1f}%로 매우 높습니다",
                    value=low_ratio,
                    threshold=ALERT_THRESHOLDS["low_confidence_ratio_critical"],
                    action="모델 성능 점검 및 재학습 검토"
                ))
            elif low_ratio >= ALERT_THRESHOLDS["low_confidence_ratio_warning"]:
                alerts.append(AlertInfo(
                    level="warning",
                    message=f"낮은 신뢰도 비율이 {low_ratio:.1f}%로 높습니다",
                    value=low_ratio,
                    threshold=ALERT_THRESHOLDS["low_confidence_ratio_warning"],
                    action="신뢰도 낮은 검출 건 검토 권장"
                ))

    # 4. PCB당 평균 결함 개수 체크
    if avg_defects_per_item >= ALERT_THRESHOLDS["avg_defects_per_item_critical"]:
        alerts.append(AlertInfo(
            level="critical",
            message=f"불량 PCB당 평균 결함 개수가 {avg_defects_per_item:.1f}개로 매우 많습니다",
            value=avg_defects_per_item,
            threshold=ALERT_THRESHOLDS["avg_defects_per_item_critical"],
            action="공정 품질 긴급 점검 필요"
        ))
    elif avg_defects_per_item >= ALERT_THRESHOLDS["avg_defects_per_item_warning"]:
        alerts.append(AlertInfo(
            level="warning",
            message=f"불량 PCB당 평균 결함 개수가 {avg_defects_per_item:.1f}개로 많습니다",
            value=avg_defects_per_item,
            threshold=ALERT_THRESHOLDS["avg_defects_per_item_warning"],
            action="공정 개선 검토 권장"
        ))

    return alerts


def determine_system_status(alerts: List[AlertInfo]) -> str:
    """
    알림 목록에 따라 시스템 상태 결정.
    """
    if not alerts:
        return "healthy"

    if any(a.level == "critical" for a in alerts):
        return "critical"

    if any(a.level == "warning" for a in alerts):
        return "warning"

    return "healthy"


async def get_health(session_id: Optional[str]) -> HealthResponse:
    """
    시스템 건강 상태 모니터링 통합 함수.
    """
    # 세션 필터 처리
    resolved_session_id = await resolve_session_filter(session_id)

    # 세션 정보 조회
    session_info = await get_session_info(resolved_session_id)

    # 기본 통계 조회
    stats = await get_stats(resolved_session_id)

    # 결함 신뢰도 통계
    defect_confidence_stats = await get_defect_confidence_stats(resolved_session_id)

    # 결함 타입별 통계
    defect_type_stats = await get_defect_type_stats(resolved_session_id)

    # 알림 생성
    alerts = generate_alerts(
        defect_rate=stats.defect_rate,
        avg_defects_per_item=stats.avg_defects_per_item,
        defect_confidence_stats=defect_confidence_stats
    )

    # 시스템 상태 결정
    status = determine_system_status(alerts)

    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        session_info=session_info,
        total_inspections=stats.total_inspections,
        normal_count=stats.normal_count,
        defect_count=stats.defect_items,
        defect_rate=stats.defect_rate,
        total_defects=stats.total_defects,
        avg_defects_per_item=stats.avg_defects_per_item,
        defect_confidence_stats=defect_confidence_stats,
        defect_type_stats=defect_type_stats,
        alerts=alerts
    )


# ===== 피드백 관련 함수 =====

async def log_exists(log_id: int) -> bool:
    """
    log_id가 inspection_logs에 존재하는지 확인
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT id FROM inspection_logs WHERE id = ?", (log_id,)
        )
        row = await cursor.fetchone()
        return row is not None


async def add_feedback(
    log_id: int,
    feedback_type: str,
    correct_label: Optional[str] = None,
    comment: Optional[str] = None,
    created_by: Optional[str] = None
) -> Dict:
    """
    피드백 저장

    Args:
        log_id: 검사 로그 ID
        feedback_type: 피드백 종류 (false_positive, false_negative, label_correction)
        correct_label: 올바른 라벨 (label_correction 시 필수)
        comment: 추가 설명
        created_by: 작성자

    Returns:
        저장된 피드백 데이터

    Raises:
        HTTPException: log_id가 존재하지 않을 때
    """
    # log_id 존재 여부 검증
    if not await log_exists(log_id):
        raise HTTPException(status_code=404, detail=f"Inspection log {log_id} not found")

    # 피드백 저장
    created_at = datetime.now().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO feedback
            (log_id, feedback_type, correct_label, comment, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (log_id, feedback_type, correct_label, comment, created_at, created_by)
        )
        await db.commit()
        return {
            "id": cursor.lastrowid,
            "log_id": log_id,
            "feedback_type": feedback_type,
            "correct_label": correct_label,
            "comment": comment,
            "created_at": created_at,
            "created_by": created_by,
        }

async def get_feedback_queue() -> List[Dict]:
    """
    처리되지 않은(resolved=0) 피드백 목록 조회
    사용자가 신고한 건들을 라벨링 툴에 보여주기 위함
    """
    query = """
    SELECT 
        f.id as feedback_id,
        f.log_id,
        f.feedback_type,
        f.comment,
        f.created_at,
        l.image_path,   -- S3 키 (예: raw/...)
        l.detections    -- 원본 AI 예측 결과 (JSON)
    FROM feedback f
    JOIN inspection_logs l ON f.log_id = l.id
    WHERE f.status != 'resolved' OR f.status IS NULL
    ORDER BY f.created_at DESC
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(query)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

async def resolve_feedback(feedback_id: int):
    """
    피드백 상태를 'resolved'로 변경 (재라벨링 완료 시)
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE feedback SET status = 'resolved' WHERE id = ?",
            (feedback_id,)
        )
        await db.commit()


async def get_feedback_by_log_id(log_id: int) -> List[Dict]:
    """
    특정 검사에 대한 피드백 조회 (향후 확장용)

    Args:
        log_id: 검사 로그 ID

    Returns:
        피드백 목록 (최신순)
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM feedback WHERE log_id = ? ORDER BY created_at DESC",
            (log_id,)
        )
        return [dict(row) for row in await cursor.fetchall()]


async def get_feedback_stats() -> Dict:
    """
    피드백 통계 집계

    Returns:
        {
            "total_feedback": int,
            "by_type": {"false_positive": int, "false_negative": int, "label_correction": int},
            "by_defect_type": [
                {
                    "defect_type": str,
                    "false_positive": int,
                    "false_negative": int,
                    "label_correction": int
                }
            ],
            "recent_feedback_count": int
        }
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 1. 전체 피드백 개수
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM feedback")
        row = await cursor.fetchone()
        total_feedback = row["cnt"]

        # 2. 피드백 종류별 집계
        cursor = await db.execute("""
            SELECT feedback_type, COUNT(*) as cnt
            FROM feedback
            GROUP BY feedback_type
        """)
        rows = await cursor.fetchall()

        by_type = {
            "false_positive": 0,
            "false_negative": 0,
            "label_correction": 0
        }
        for row in rows:
            by_type[row["feedback_type"]] = row["cnt"]

        # 3. 결함 타입별 × 피드백 종류별 교차 집계
        # inspection_logs와 JOIN, detections JSON 파싱
        cursor = await db.execute("""
            SELECT il.detections, f.feedback_type
            FROM feedback f
            INNER JOIN inspection_logs il ON f.log_id = il.id
            WHERE il.result = 'defect'
        """)
        rows = await cursor.fetchall()

        # 결함 타입별 집계
        defect_type_stats = {}

        for row in rows:
            if row["detections"]:
                try:
                    detections = json.loads(row["detections"])
                    for det in detections:
                        defect_type = det.get("defect_type", "unknown")
                        feedback_type = row["feedback_type"]

                        if defect_type not in defect_type_stats:
                            defect_type_stats[defect_type] = {
                                "defect_type": defect_type,
                                "false_positive": 0,
                                "false_negative": 0,
                                "label_correction": 0
                            }

                        defect_type_stats[defect_type][feedback_type] += 1
                except:
                    pass

        # 리스트로 변환 (개수 내림차순 정렬)
        by_defect_type = list(defect_type_stats.values())
        by_defect_type.sort(
            key=lambda x: x["false_positive"] + x["false_negative"] + x["label_correction"],
            reverse=True
        )

        # 4. 최근 24시간 피드백 개수
        cursor = await db.execute("""
            SELECT COUNT(*) as cnt
            FROM feedback
            WHERE created_at >= datetime('now', '-1 day')
        """)
        row = await cursor.fetchone()
        recent_feedback_count = row["cnt"]

        return {
            "total_feedback": total_feedback,
            "by_type": by_type,
            "by_defect_type": by_defect_type,
            "recent_feedback_count": recent_feedback_count
        }
