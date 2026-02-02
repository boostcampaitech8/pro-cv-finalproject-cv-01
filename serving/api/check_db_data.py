import sqlite3
import json
import os

DB_PATH = "/home/ubuntu/pro-cv-finalproject-cv-01/serving/api/data/inspection.db"

def list_data():
    if not os.path.exists(DB_PATH):
        print(f"Database file not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {[t['name'] for t in tables]}")

        cursor.execute("SELECT COUNT(*) as count FROM inspection_logs")
        count = cursor.fetchone()['count']
        print(f"Total rows in inspection_logs: {count}")

        if count > 0:
            cursor.execute("SELECT * FROM inspection_logs ORDER BY id DESC LIMIT 20")
            rows = cursor.fetchall()
            
            print(f"\n{'ID':<5} | {'Timestamp':<25} | {'Image ID':<15} | {'Result':<10}")
            print("-" * 65)
            for row in rows:
                print(f"{row['id']:<5} | {row['timestamp']:<25} | {row['image_id']:<15} | {row['result']:<10}")
        else:
            print("No data found in inspection_logs table.")
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    list_data()
