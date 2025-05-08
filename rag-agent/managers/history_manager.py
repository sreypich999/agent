import sqlite3
import json
from datetime import datetime

class HistoryManager:
    def __init__(self):
        self.conn = sqlite3.connect("./data/history.db")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                question TEXT,
                answer TEXT,
                sources TEXT,
                validated BOOLEAN
            )
        """)

    def add_entry(self, question, answer, sources):
        self.conn.execute(
            "INSERT INTO history (timestamp, question, answer, sources, validated) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(), question, answer, json.dumps(sources), True)
        )
        self.conn.commit()

    def get_history(self, limit=10):
        cur = self.conn.execute(
            "SELECT timestamp, question, answer, sources FROM history ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        return [
            {"timestamp": t, "question": q, "answer": a, "sources": json.loads(s)}
            for t, q, a, s in cur.fetchall()
        ]
