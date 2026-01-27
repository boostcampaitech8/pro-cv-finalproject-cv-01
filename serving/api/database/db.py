import aiosqlite
import json
from typing import List, Dict, Optional
from schemas.schemas import DetectRequest, StatsResponse, InspectionLogResponse
from config.settings import DB_PATH


async def init_db():
    """
    Initialize database and create tables if they don't exist.
    Called on server startup.
    Schema Update: 1 row per image. 'detections' column stores JSON list of defects.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS inspection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_id TEXT NOT NULL,
                result TEXT NOT NULL,
                detections TEXT,
                image_path TEXT
            )
        """)
        await db.commit()


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
            INSERT INTO inspection_logs (timestamp, image_id, result, detections, image_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                data.timestamp,
                data.image_id,
                result_status,
                detections_json,
                image_path
            )
        )
        await db.commit()
        
        return [cursor.lastrowid]


async def get_stats() -> StatsResponse:
    """
    Calculate statistics from the logs.
    Now requires parsing 'detections' column to count total defects.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 1. Total Inspections (Total Rows)
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM inspection_logs")
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
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM inspection_logs WHERE result='defect'")
        row = await cursor.fetchone()
        defect_items = row["cnt"]

        # 3. Total Defects (Sum of length of detections list in each defect row)
        # SQLite JSON extension might not be enabled, so we fetch and sum in Python.
        cursor = await db.execute("SELECT detections FROM inspection_logs WHERE result='defect'")
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
            "SELECT * FROM inspection_logs WHERE result='defect' ORDER BY id DESC LIMIT 1"
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
            last_defect=last_defect_log
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
            if record["detections"]:
                record["detections"] = json.loads(record["detections"])
            else:
                record["detections"] = []
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
            if record["detections"]:
                record["detections"] = json.loads(record["detections"])
            else:
                record["detections"] = []
            result.append(record)

        return result
