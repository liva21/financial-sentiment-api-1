import sqlite3
import os
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "data/monitoring.db")

def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

get_connection = _get_conn

def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            text        TEXT    NOT NULL,
            sentiment   TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            latency_ms  REAL    NOT NULL,
            endpoint    TEXT    NOT NULL,
            batch_size  INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()

def log_request(text, sentiment, confidence, latency_ms, endpoint="/predict", batch_size=1):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO requests
            (timestamp, text, sentiment, confidence, latency_ms, endpoint, batch_size)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        text[:500],
        sentiment,
        confidence,
        latency_ms,
        endpoint,
        batch_size,
    ))
    conn.commit()
    conn.close()

def get_stats():
    conn = _get_conn()

    total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    if total == 0:
        conn.close()
        return {"total": 0}

    dist = conn.execute("""
        SELECT sentiment, COUNT(*) as count
        FROM requests GROUP BY sentiment
    """).fetchall()

    hourly = conn.execute("""
        SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) as hour,
               COUNT(*) as count
        FROM requests
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY hour ORDER BY hour
    """).fetchall()

    latency = conn.execute("""
        SELECT endpoint,
               ROUND(AVG(latency_ms), 2) as avg_ms,
               ROUND(MIN(latency_ms), 2) as min_ms,
               ROUND(MAX(latency_ms), 2) as max_ms
        FROM requests GROUP BY endpoint
    """).fetchall()

    recent = conn.execute("""
        SELECT timestamp, text, sentiment, confidence, latency_ms
        FROM requests ORDER BY id DESC LIMIT 10
    """).fetchall()

    conn.close()

    return {
        "total"       : total,
        "distribution": [dict(r) for r in dist],
        "hourly"      : [dict(r) for r in hourly],
        "latency"     : [dict(r) for r in latency],
        "recent"      : [dict(r) for r in recent],
    }
