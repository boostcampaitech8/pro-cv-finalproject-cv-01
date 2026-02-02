import asyncio
import aiosqlite
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db import DB_PATH

async def dump_db():
    print(f"Reading DB from: {DB_PATH}")
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT id, result, detections FROM inspection_logs")
        rows = await cursor.fetchall()
        
        print(f"Total Rows: {len(rows)}")
        
        unknown_count = 0
        valid_count = 0
        
        for row in rows:
            print(f"ID: {row['id']}, Result: {row['result']}")
            dets = row['detections']
            print(f"  Raw Detections: {dets}")
            
            if not dets:
                continue
                
            try:
                parsed_dets = json.loads(dets)
                for d in parsed_dets:
                    dtype = d.get('defect_type')
                    print(f"    - Defect Type: {dtype}")
                    if not dtype:
                        unknown_count += 1
                    else:
                        valid_count += 1
            except Exception as e:
                print(f"    [Error Parsing JSON]: {e}")
                
        print(f"\nSummary: Valid Defects: {valid_count}, Unknown Defects: {unknown_count}")

if __name__ == "__main__":
    asyncio.run(dump_db())
