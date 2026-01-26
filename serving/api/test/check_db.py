import sqlite3
import json
from pathlib import Path

# DB 경로 설정
DB_PATH = Path(r"c:\Users\admin\Downloads\Final Project\pro-cv-finalproject-cv-01\serving\api\data\inspection.db")

def check_db():
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 테이블 목록 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"--- Tables in Database ---")
        for table in tables:
            print(f"Table: {table['name']}")
        print("\n")

        # inspection_logs 테이블 데이터 확인
        cursor.execute("SELECT * FROM inspection_logs ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()

        print(f"--- Recent 10 Logs in 'inspection_logs' ---")
        if not rows:
            print("No data found in inspection_logs table.")
        else:
            for row in rows:
                record = dict(row)
                if 'detections' in record and record['detections']:
                    try:
                        record['detections'] = json.loads(record['detections'])
                    except:
                        pass
                print(record)
        
        # 총 레코드 수 확인
        cursor.execute("SELECT count(*) as count FROM inspection_logs")
        count = cursor.fetchone()['count']
        print(f"\nTotal record count: {count}")

        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_db()
