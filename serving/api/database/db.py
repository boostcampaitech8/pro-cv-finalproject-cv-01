import aiosqlite
import json
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta

# 한국 표준시 (KST, UTC+9)
KST = timezone(timedelta(hours=9))
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
                ended_at TEXT,
                mlops_version TEXT,
                yolo_version TEXT
            )
        """)

        # 기존 sessions 테이블 마이그레이션 (필요시 컬럼 추가)
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN mlops_version TEXT")
            await db.execute("ALTER TABLE sessions ADD COLUMN yolo_version TEXT")
        except aiosqlite.OperationalError:
            pass

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
                target_bbox TEXT,
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

        # 기존 테이블에 target_bbox 컬럼이 없으면 추가 (마이그레이션)
        try:
            await db.execute("ALTER TABLE feedback ADD COLUMN target_bbox TEXT")
        except Exception:
            # 컬럼이 이미 존재하면 무시
            pass

        # 기존 테이블에 camera_id 컬럼이 없으면 추가 (마이그레이션)
        try:
            await db.execute("ALTER TABLE inspection_logs ADD COLUMN camera_id TEXT DEFAULT 'cam_1'")
        except Exception:
            pass

        # 검증 완료 표시 컬럼 추가 (마이그레이션)
        try:
            await db.execute("ALTER TABLE inspection_logs ADD COLUMN is_verified BOOLEAN DEFAULT FALSE")
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE inspection_logs ADD COLUMN verified_at TEXT")
        except Exception:
            pass

        try:
            await db.execute("ALTER TABLE inspection_logs ADD COLUMN verified_by TEXT")
        except Exception:
            pass

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
            INSERT INTO inspection_logs (timestamp, image_id, result, detections, image_path, session_id, camera_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.timestamp,
                data.image_id,
                result_status,
                detections_json,
                image_path,
                data.session_id,
                data.camera_id or "cam_1"
            )
        )
        await db.commit()

        return [cursor.lastrowid]


async def get_stats(session_id: Optional[int] = None, camera_id: Optional[str] = None) -> StatsResponse:
    """
    Calculate statistics from the logs.
    Now requires parsing 'detections' column to count total defects.
    session_id가 제공되면 해당 세션의 통계만 반환.
    camera_id가 제공되면 해당 카메라의 통계만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 필터 조건 동적 구성
        conditions = []
        params = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if camera_id is not None:
            conditions.append("camera_id = ?")
            params.append(camera_id)

        session_filter = ""
        if conditions:
            session_filter = " WHERE " + " AND ".join(conditions)
        session_params = tuple(params)

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
        defect_conditions = ["result='defect'"] + conditions[:]
        defect_filter = " WHERE " + " AND ".join(defect_conditions)
        defect_params = session_params
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
                session_id=row["session_id"],
                camera_id=row["camera_id"]
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


async def get_recent_logs(limit: int = 10, session_id: Optional[int] = None, camera_id: Optional[str] = None) -> List[Dict]:
    """
    Returns the N most recent logs.
    session_id가 제공되면 해당 세션의 로그만 반환.
    camera_id가 제공되면 해당 카메라의 로그만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        conditions = []
        params = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if camera_id is not None:
            conditions.append("camera_id = ?")
            params.append(camera_id)

        query = "SELECT * FROM inspection_logs"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor = await db.execute(query, tuple(params))
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


async def get_defect_logs(session_id: Optional[int] = None, camera_id: Optional[str] = None) -> List[Dict]:
    """
    Returns all logs that are defects.
    session_id가 제공되면 해당 세션의 결함 로그만 반환.
    camera_id가 제공되면 해당 카메라의 결함 로그만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        conditions = ["result = 'defect'"]
        params = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if camera_id is not None:
            conditions.append("camera_id = ?")
            params.append(camera_id)

        query = "SELECT * FROM inspection_logs WHERE " + " AND ".join(conditions) + " ORDER BY id DESC"
        cursor = await db.execute(query, tuple(params))
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

async def create_session(model_name: Optional[str] = None) -> Dict:
    """
    새 세션을 생성하고 ID와 시작 시간을 반환.
    """
    from datetime import datetime
    started_at = datetime.now(KST).isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO sessions (started_at, model_name) VALUES (?, ?)",
            (started_at, model_name)
        )
        await db.commit()
        session_id = cursor.lastrowid

    return {"id": session_id, "started_at": started_at, "ended_at": None, "model_name": model_name}


async def end_session(session_id: int) -> Optional[Dict]:
    """
    세션을 종료하고 ended_at을 설정.
    """
    from datetime import datetime
    ended_at = datetime.now(KST).isoformat()

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
            "ended_at": ended_at,
            "model_name": row["model_name"]
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

        high = mid = low = 0

        for row in rows:
            if row["detections"]:
                try:
                    detections = json.loads(row["detections"])
                    for det in detections:
                        conf = det.get("confidence", 0)
                        if conf >= 0.8:
                            high += 1
                        elif conf >= 0.5:
                            mid += 1
                        else:
                            low += 1
                except:
                    pass

        return ConfidenceDistribution(high=high, mid=mid, low=low)


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
            end_time = datetime.fromisoformat(ended_at) if ended_at else datetime.now(KST)
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
        total_detections = dist.high + dist.mid + dist.low

        if total_detections > 0:
            # mid + low (< 80%)를 기준으로 저신뢰도 비율 계산
            low_ratio = ((dist.mid + dist.low) / total_detections) * 100

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

    # 심각도 순으로 정렬 (critical → warning)
    # 프론트엔드에서 alerts[0]을 배너에 표시하므로 가장 심각한 알림이 먼저 와야 함
    severity_order = {"critical": 0, "warning": 1}
    alerts.sort(key=lambda x: severity_order.get(x.level, 2))

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

    # 저신뢰도 비율 계산
    low_ratio = 0.0
    if defect_confidence_stats:
        dist = defect_confidence_stats.distribution
        total_detections = dist.high + dist.mid + dist.low
        if total_detections > 0:
            # 기존에는 < 80%를 저신뢰도로 정의했으므로 mid + low를 사용
            low_ratio = ((dist.mid + dist.low) / total_detections) * 100

    # 활성 모델 버전 추출 (session_info가 존재할 경우)
    active_model = None
    if session_info:
        active_model = session_info.get("model_name")

    return HealthResponse(
        status=status,
        timestamp=datetime.now(KST).isoformat(),
        session_info=session_info,
        total_inspections=stats.total_inspections,
        normal_count=stats.normal_count,
        defect_count=stats.defect_items,
        defect_rate=stats.defect_rate,
        total_defects=stats.total_defects,
        avg_defects_per_item=stats.avg_defects_per_item,
        low_confidence_ratio=round(low_ratio, 2),
        defect_confidence_stats=defect_confidence_stats,
        defect_type_stats=defect_type_stats,
        alerts=alerts,
        active_model=active_model
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
    created_by: Optional[str] = None,
    target_bbox: Optional[List[int]] = None
) -> Dict:
    """
    피드백 저장

    Args:
        log_id: 검사 로그 ID
        feedback_type: 피드백 종류 (false_positive, false_negative, tp_wrong_class)
        correct_label: 올바른 라벨 (tp_wrong_class 시 필수)
        comment: 추가 설명
        created_by: 작성자
        target_bbox: 대상 bbox [x1, y1, x2, y2] (false_positive, tp_wrong_class 시 필수)

    Returns:
        저장된 피드백 데이터

    Raises:
        HTTPException: log_id가 존재하지 않을 때
    """
    # log_id 존재 여부 검증
    if not await log_exists(log_id):
        raise HTTPException(status_code=404, detail=f"Inspection log {log_id} not found")

    # target_bbox를 JSON 문자열로 직렬화
    target_bbox_json = json.dumps(target_bbox) if target_bbox else None

    # 피드백 저장
    created_at = datetime.now(KST).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO feedback
            (log_id, feedback_type, correct_label, comment, created_at, created_by, target_bbox)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (log_id, feedback_type, correct_label, comment, created_at, created_by, target_bbox_json)
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
            "target_bbox": target_bbox
        }

async def get_feedback_queue(session_id: Optional[int] = None) -> List[Dict]:
    """
    처리되지 않은(resolved=0) 피드백 목록 조회
    사용자가 신고한 건들을 라벨링 툴에 보여주기 위함

    Args:
        session_id: 세션 필터 (None이면 전체, 숫자면 해당 세션만)
    """
    query = """
    SELECT
        f.id as feedback_id,
        f.log_id,
        f.feedback_type,
        f.comment,
        f.created_at,
        f.target_bbox,
        l.image_path,   -- S3 키 (예: raw/...)
        l.detections    -- 원본 AI 예측 결과 (JSON)
    FROM feedback f
    JOIN inspection_logs l ON f.log_id = l.id
    WHERE (f.status != 'resolved' OR f.status IS NULL)
    """
    params = ()
    if session_id is not None:
        query += " AND l.session_id = ?"
        params = (session_id,)
    query += " ORDER BY f.created_at DESC"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(query, params)
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


def bbox_equals(bbox1: List, bbox2: List, tolerance: int = 2) -> bool:
    """
    두 bbox가 동일한지 확인 (오차 허용)

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        tolerance: 허용 오차 (픽셀)

    Returns:
        동일 여부
    """
    if not bbox1 or not bbox2:
        return False
    if len(bbox1) != 4 or len(bbox2) != 4:
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(bbox1, bbox2))


async def get_feedback_stats(session_id: Optional[int] = None) -> Dict:
    """
    피드백 통계 조회 (bbox 기반 정확도 분석)

    Args:
        session_id: 세션 필터 (None이면 전체, 숫자면 해당 세션만)

    새로운 응답 구조:
    - image_stats: 전체 이미지 + 검증 진행률
    - bbox_stats: 검증된 defect의 bbox별 정확도 (is_verified=true만)
    - feedback_stats: 피드백 타입별 집계
    - class_confusion: 클래스 혼동 패턴 (is_verified=true, FN 제외)

    Returns:
        {
            "image_stats": {...},
            "bbox_stats": {...},
            "feedback_stats": {...},
            "class_confusion": [...]
        }
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 세션 필터 조건
        session_filter = " WHERE session_id = ?" if session_id is not None else ""
        session_params = (session_id,) if session_id is not None else ()

        # ===== 1. image_stats 집계 (전체 이미지) =====
        cursor = await db.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result = 'defect' THEN 1 ELSE 0 END) as defect_count,
                SUM(CASE WHEN result = 'normal' THEN 1 ELSE 0 END) as normal_count,
                SUM(CASE WHEN is_verified = 1 THEN 1 ELSE 0 END) as verified_count,
                SUM(CASE WHEN is_verified = 1 AND result = 'defect' THEN 1 ELSE 0 END) as verified_defect,
                SUM(CASE WHEN is_verified = 1 AND result = 'normal' THEN 1 ELSE 0 END) as verified_normal
            FROM inspection_logs
            {session_filter}
        """, session_params)
        row = await cursor.fetchone()

        total = row["total"] or 0
        defect_count = row["defect_count"] or 0
        normal_count = row["normal_count"] or 0
        verified_count = row["verified_count"] or 0
        verified_defect = row["verified_defect"] or 0
        verified_normal = row["verified_normal"] or 0

        image_stats = {
            "total": total,
            "by_result": {"defect": defect_count, "normal": normal_count},
            "verified": verified_count,
            "unverified": total - verified_count,
            "verification_rate": round(verified_count / total * 100, 1) if total > 0 else 0.0,
            "verified_by_result": {"defect": verified_defect, "normal": verified_normal}
        }

        # ===== 2. bbox_stats 집계 (검증된 defect만) =====
        # 검증된 defect 로그 조회 (세션 필터 적용)
        verified_query = "SELECT id, detections FROM inspection_logs WHERE result = 'defect' AND is_verified = 1"
        if session_id is not None:
            verified_query += " AND session_id = ?"
        cursor = await db.execute(verified_query, session_params)
        verified_defect_logs = await cursor.fetchall()

        # 전체 피드백 조회 (target_bbox 포함, 세션 필터 시 JOIN)
        if session_id is not None:
            cursor = await db.execute("""
                SELECT f.id, f.log_id, f.feedback_type, f.target_bbox, f.correct_label
                FROM feedback f
                JOIN inspection_logs l ON f.log_id = l.id
                WHERE l.session_id = ?
            """, (session_id,))
        else:
            cursor = await db.execute("""
                SELECT id, log_id, feedback_type, target_bbox, correct_label
                FROM feedback
            """)
        all_feedbacks = await cursor.fetchall()

        # 피드백을 log_id별로 그룹화
        feedbacks_by_log = {}
        for fb in all_feedbacks:
            log_id = fb["log_id"]
            if log_id not in feedbacks_by_log:
                feedbacks_by_log[log_id] = []

            target_bbox = None
            if fb["target_bbox"]:
                try:
                    target_bbox = json.loads(fb["target_bbox"])
                except:
                    pass

            feedbacks_by_log[log_id].append({
                "id": fb["id"],
                "feedback_type": fb["feedback_type"],
                "target_bbox": target_bbox,
                "correct_label": fb["correct_label"]
            })

        # bbox별 정답/오답 판정
        bbox_stats = {
            "total": 0,
            "correct": 0,
            "false_positive": 0,
            "wrong_class": 0,
            "by_defect_type": {}
        }

        # class_confusion 집계용 (from_class -> to_class -> count)
        confusion_counter = {}

        for log in verified_defect_logs:
            log_id = log["id"]
            detections = []
            if log["detections"]:
                try:
                    detections = json.loads(log["detections"])
                except:
                    pass

            log_feedbacks = feedbacks_by_log.get(log_id, [])

            for det in detections:
                det_bbox = det.get("bbox")
                defect_type = det.get("defect_type", "unknown")

                bbox_stats["total"] += 1

                # 결함 타입별 통계 초기화
                if defect_type not in bbox_stats["by_defect_type"]:
                    bbox_stats["by_defect_type"][defect_type] = {
                        "total": 0, "correct": 0, "fp": 0, "wrong": 0
                    }
                bbox_stats["by_defect_type"][defect_type]["total"] += 1

                # 해당 bbox에 대한 피드백 찾기
                feedback = None
                for fb in log_feedbacks:
                    fb_bbox = fb.get("target_bbox")
                    if bbox_equals(det_bbox, fb_bbox):
                        feedback = fb
                        break

                if feedback is None:
                    # 피드백 없음 → 정답 (암묵적 TP)
                    bbox_stats["correct"] += 1
                    bbox_stats["by_defect_type"][defect_type]["correct"] += 1

                elif feedback["feedback_type"] == "false_positive":
                    # 오탐
                    bbox_stats["false_positive"] += 1
                    bbox_stats["by_defect_type"][defect_type]["fp"] += 1

                elif feedback["feedback_type"] == "tp_wrong_class":
                    # 클래스 오류
                    bbox_stats["wrong_class"] += 1
                    bbox_stats["by_defect_type"][defect_type]["wrong"] += 1

                    # class_confusion 집계
                    from_class = defect_type
                    to_class = feedback.get("correct_label", "unknown")
                    key = (from_class, to_class)
                    confusion_counter[key] = confusion_counter.get(key, 0) + 1

        # 정확도 계산
        if bbox_stats["total"] > 0:
            bbox_stats["accuracy_rate"] = round(
                bbox_stats["correct"] / bbox_stats["total"] * 100, 1
            )
        else:
            bbox_stats["accuracy_rate"] = 0.0

        for dt_stats in bbox_stats["by_defect_type"].values():
            if dt_stats["total"] > 0:
                dt_stats["accuracy"] = round(
                    dt_stats["correct"] / dt_stats["total"] * 100, 1
                )
            else:
                dt_stats["accuracy"] = 0.0

        # ===== 3. feedback_stats 집계 (세션 필터 적용) =====
        if session_id is not None:
            cursor = await db.execute("""
                SELECT f.feedback_type, COUNT(*) as cnt
                FROM feedback f
                JOIN inspection_logs l ON f.log_id = l.id
                WHERE l.session_id = ?
                GROUP BY f.feedback_type
            """, (session_id,))
        else:
            cursor = await db.execute("""
                SELECT feedback_type, COUNT(*) as cnt
                FROM feedback
                GROUP BY feedback_type
            """)
        rows = await cursor.fetchall()

        feedback_stats = {
            "total": 0,
            "false_positive": 0,
            "tp_wrong_class": 0,
            "false_negative": 0
        }
        for row in rows:
            fb_type = row["feedback_type"]
            cnt = row["cnt"]
            feedback_stats["total"] += cnt
            if fb_type in feedback_stats:
                feedback_stats[fb_type] = cnt

        # ===== 4. class_confusion 리스트 생성 =====
        class_confusion = [
            {"from_class": k[0], "to_class": k[1], "count": v}
            for k, v in confusion_counter.items()
        ]
        # count 내림차순 정렬
        class_confusion.sort(key=lambda x: x["count"], reverse=True)

        return {
            "image_stats": image_stats,
            "bbox_stats": bbox_stats,
            "feedback_stats": feedback_stats,
            "class_confusion": class_confusion
        }


async def add_bulk_feedback(
    log_id: int,
    feedbacks: List[Dict],
    created_by: Optional[str] = None
) -> Dict:
    """
    다중 bbox 피드백 저장 (트랜잭션 원자성 보장)

    Args:
        log_id: 검사 로그 ID
        feedbacks: 피드백 목록 (FeedbackItem.model_dump() 결과)
        created_by: 작성자

    Returns:
        {
            "log_id": int,
            "feedback_ids": List[int],
            "created_count": int,
            "created_at": str
        }

    Raises:
        HTTPException 404: log_id 없음
        HTTPException 500: 트랜잭션 실패
    """
    # 1. log_id 존재 검증
    if not await log_exists(log_id):
        raise HTTPException(status_code=404, detail=f"Inspection log {log_id} not found")

    created_at = datetime.now(KST).isoformat()
    feedback_ids = []

    async with aiosqlite.connect(DB_PATH) as db:
        try:
            # 2. BEGIN TRANSACTION (원자성)
            await db.execute("BEGIN TRANSACTION")

            # 3. Loop: N개 INSERT
            for feedback_item in feedbacks:
                target_bbox_json = json.dumps(feedback_item["target_bbox"]) \
                    if feedback_item.get("target_bbox") else None

                cursor = await db.execute(
                    """INSERT INTO feedback
                    (log_id, feedback_type, correct_label, comment, created_at, created_by, target_bbox)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        log_id,
                        feedback_item["feedback_type"],
                        feedback_item.get("correct_label"),
                        feedback_item.get("comment"),
                        created_at,
                        created_by,
                        target_bbox_json
                    )
                )
                feedback_ids.append(cursor.lastrowid)

            # 4. COMMIT (성공)
            await db.commit()

            return {
                "log_id": log_id,
                "feedback_ids": feedback_ids,
                "created_count": len(feedback_ids),
                "created_at": created_at
            }

        except Exception as e:
            # 5. ROLLBACK (실패)
            await db.execute("ROLLBACK")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create bulk feedback: {str(e)}"
            )


async def mark_as_verified(log_id: int, verified_by: Optional[str] = None) -> None:
    """
    검사 로그를 검증 완료 표시

    Args:
        log_id: 검사 로그 ID
        verified_by: 검증자 (선택)
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE inspection_logs
               SET is_verified = 1,
                   verified_at = ?,
                   verified_by = ?
               WHERE id = ?""",
            (datetime.now(KST).isoformat(), verified_by, log_id)
        )
        await db.commit()


async def get_inspection_log(log_id: int) -> Optional[Dict]:
    """
    특정 검사 로그 조회

    Args:
        log_id: 검사 로그 ID

    Returns:
        검사 로그 데이터 (detections는 JSON 문자열 그대로)
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM inspection_logs WHERE id = ?",
            (log_id,)
        )
        row = await cursor.fetchone()

        if row:
            return dict(row)
        return None
