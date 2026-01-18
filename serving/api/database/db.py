import aiosqlite
import json
from typing import List, Dict, Optional
from schemas.schemas import DetectRequest, DefectInfo, StatsResponse
from config.settings import DB_PATH


async def init_db():
    """
    Initialize database and create tables if they don't exist.
    Called on server startup.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS inspection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_id TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                defect_type TEXT,
                bbox TEXT,
                image_path TEXT
            )
        """)
        await db.commit()


async def add_inspection_log(data: DetectRequest, image_path: Optional[str] = None) -> List[int]:
    """
    1개 요청에서 여러 결함을 각각 row로 저장.
    detections가 비어있으면 (정상) 1개 row만 저장.
    Returns: 저장된 row ID 리스트
    """
    async with aiosqlite.connect(DB_PATH) as db:
        inserted_ids = []

        if len(data.detections) == 0:
            # 정상인 경우: 1개 row만 저장
            cursor = await db.execute(
                """
                INSERT INTO inspection_logs (timestamp, image_id, result, confidence, defect_type, bbox, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (data.timestamp, data.image_id, "normal", 1.0, None, None, None)
            )
            inserted_ids.append(cursor.lastrowid)
        else:
            # 불량인 경우: 각 결함마다 row 저장
            for i, detection in enumerate(data.detections):
                cursor = await db.execute(
                    """
                    INSERT INTO inspection_logs (timestamp, image_id, result, confidence, defect_type, bbox, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data.timestamp,
                        data.image_id,
                        "defect",
                        detection.confidence,
                        detection.defect_type,
                        json.dumps(detection.bbox),
                        image_path if i == 0 else None  # 첫번째 결함에만 이미지 경로
                    )
                )
                inserted_ids.append(cursor.lastrowid)

        await db.commit()
        return inserted_ids


async def get_stats() -> StatsResponse:
    """
    새로운 통계 계산:
    - total_inspections: DISTINCT image_id 개수
    - defect_items: DISTINCT image_id WHERE result='defect'
    - total_defects: COUNT(*) WHERE result='defect'
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 총 검사 수 (DISTINCT image_id)
        cursor = await db.execute("SELECT COUNT(DISTINCT image_id) as cnt FROM inspection_logs")
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

        # 불량 PCB 개수 (DISTINCT image_id WHERE result='defect')
        cursor = await db.execute(
            "SELECT COUNT(DISTINCT image_id) as cnt FROM inspection_logs WHERE result='defect'"
        )
        row = await cursor.fetchone()
        defect_items = row["cnt"]

        # 총 결함 개수
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM inspection_logs WHERE result='defect'")
        row = await cursor.fetchone()
        total_defects = row["cnt"]

        normal_count = total_inspections - defect_items
        defect_rate = (defect_items / total_inspections) * 100
        avg_defects_per_item = total_defects / defect_items if defect_items > 0 else 0.0

        # TODO: Calculate avg_fps based on timestamp intervals
        avg_fps = 0.0

        # 가장 최근 불량
        cursor = await db.execute(
            "SELECT * FROM inspection_logs WHERE result='defect' ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()

        last_defect_info = None
        if row:
            last_defect_info = DefectInfo(
                timestamp=row["timestamp"],
                image_id=row["image_id"],
                result=row["result"],
                confidence=row["confidence"],
                defect_type=row["defect_type"],
                bbox=json.loads(row["bbox"]) if row["bbox"] else None,
                image_path=row["image_path"]
            )

        return StatsResponse(
            total_inspections=total_inspections,
            normal_count=normal_count,
            defect_items=defect_items,
            total_defects=total_defects,
            defect_rate=round(defect_rate, 2),
            avg_defects_per_item=round(avg_defects_per_item, 2),
            avg_fps=avg_fps,
            last_defect=last_defect_info
        )


async def get_recent_logs(limit: int = 10) -> List[Dict]:
    """
    Returns the N most recent logs.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM inspection_logs ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = await cursor.fetchall()

        result = []
        for row in rows:
            record = dict(row)
            if record["bbox"]:
                record["bbox"] = json.loads(record["bbox"])
            result.append(record)

        return result


async def get_defect_logs() -> List[Dict]:
    """
    Returns all logs that are defects.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM inspection_logs WHERE result = 'defect' ORDER BY id DESC"
        )
        rows = await cursor.fetchall()

        result = []
        for row in rows:
            record = dict(row)
            if record["bbox"]:
                record["bbox"] = json.loads(record["bbox"])
            result.append(record)

        return result
