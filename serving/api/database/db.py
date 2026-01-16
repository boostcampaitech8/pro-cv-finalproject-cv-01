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


async def add_inspection_log(data: DetectRequest, image_path: Optional[str] = None) -> int:
    """
    Saves the inspection result to SQLite.
    Returns the new record ID.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO inspection_logs (timestamp, image_id, result, confidence, defect_type, bbox, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.timestamp,
                data.image_id,
                data.result,
                data.confidence,
                data.defect_type,
                json.dumps(data.bbox) if data.bbox else None,
                image_path
            )
        )
        await db.commit()
        return cursor.lastrowid


async def get_stats() -> StatsResponse:
    """
    Calculates statistics from SQLite.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Total count
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM inspection_logs")
        row = await cursor.fetchone()
        total_count = row["cnt"]

        if total_count == 0:
            return StatsResponse(
                total_count=0,
                normal_count=0,
                defect_count=0,
                defect_rate=0.0,
                avg_fps=0.0,
                last_defect=None
            )

        # Defect count
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM inspection_logs WHERE result = 'defect'")
        row = await cursor.fetchone()
        defect_count = row["cnt"]

        normal_count = total_count - defect_count
        defect_rate = (defect_count / total_count) * 100

        # TODO: Calculate avg_fps based on timestamp intervals
        avg_fps = 0.0

        # Last defect
        last_defect_info = None
        cursor = await db.execute(
            "SELECT * FROM inspection_logs WHERE result = 'defect' ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()

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
            total_count=total_count,
            normal_count=normal_count,
            defect_count=defect_count,
            defect_rate=round(defect_rate, 2),
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
