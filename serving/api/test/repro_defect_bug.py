import asyncio
import aiosqlite
import json
from datetime import datetime
import sys
import os

# Add parent directory to path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db import DB_PATH
from routers.stats import get_defect_aggregation

async def reproduce_bug():
    print(f"Connecting to DB at: {DB_PATH}")
    
    # 1. Insert Dummy Data with known defect types
    async with aiosqlite.connect(DB_PATH) as db:
        # Clear existing logs to be sure (optional, but good for clean test)
        # await db.execute("DELETE FROM inspection_logs")
        
        timestamp = datetime.now().isoformat()
        image_id = "BUG_REPRO_TEST"
        result = "defect"
        
        # Two defects: 1 scratch, 1 dent
        detections = [
            {"defect_type": "scratch", "confidence": 0.9, "bbox": [0,0,10,10]},
            {"defect_type": "dent", "confidence": 0.8, "bbox": [20,20,30,30]}
        ]
        detections_json = json.dumps(detections)
        
        await db.execute(
            """
            INSERT INTO inspection_logs (timestamp, image_id, result, detections, image_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, image_id, result, detections_json, "dummy_path")
        )
        await db.commit()
    
    print("Inserted dummy data with 1 'scratch' and 1 'dent'.")

    # 2. Call the aggregation function directly (simulating API call)
    try:
        aggregation = await get_defect_aggregation()
        print("\n--- Aggregation Result ---")
        print(json.dumps(aggregation, indent=2))
        
        # 3. Validation
        scratch_count = aggregation.get("scratch", 0)
        dent_count = aggregation.get("dent", 0)
        
        if scratch_count >= 1 and dent_count >= 1:
            print("\n[SUCCESS] 'scratch' and 'dent' found found in aggregation.")
        else:
            print("\n[FAIL] 'scratch' or 'dent' NOT found (or count is 0).")
            print("Likely bug: 'unknown' might be high or counts are missing.")
            
    except Exception as e:
        print(f"\n[ERROR] Function call failed: {e}")

if __name__ == "__main__":
    asyncio.run(reproduce_bug())
