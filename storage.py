"""SQLite persistence for historical imports, live snapshots, and final-result tracking."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

DB_PATH = Path("data/live_football_edge.sqlite3")


def connect(db_path: str | Path = DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str | Path = DB_PATH):
    with connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS historical_matches (
            date TEXT NOT NULL, home_team TEXT NOT NULL, away_team TEXT NOT NULL,
            home_goals INTEGER NOT NULL, away_goals INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'csv',
            UNIQUE(date, home_team, away_team)
        );
        CREATE TABLE IF NOT EXISTS live_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL, fixture_id TEXT NOT NULL,
            home_team TEXT NOT NULL, away_team TEXT NOT NULL,
            minute INTEGER, home_score INTEGER, away_score INTEGER,
            status TEXT, payload_json TEXT
        );
        CREATE TABLE IF NOT EXISTS final_results (
            fixture_id TEXT PRIMARY KEY, finished_at TEXT,
            home_team TEXT NOT NULL, away_team TEXT NOT NULL,
            home_goals INTEGER, away_goals INTEGER, result TEXT,
            source TEXT NOT NULL DEFAULT 'api'
        );
        """)


def save_history(frame: pd.DataFrame, db_path: str | Path = DB_PATH) -> int:
    rows = [
        (r.date.isoformat(), str(r.home_team), str(r.away_team),
         int(r.home_goals), int(r.away_goals), "csv")
        for r in frame.itertuples(index=False)
    ]
    with connect(db_path) as conn:
        conn.executemany("""
        INSERT INTO historical_matches(date, home_team, away_team, home_goals, away_goals, source)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(date, home_team, away_team) DO UPDATE SET
          home_goals=excluded.home_goals, away_goals=excluded.away_goals, source=excluded.source
        """, rows)
    return len(rows)


def load_history(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        frame = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_goals, away_goals FROM historical_matches ORDER BY date",
            conn
        )
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
        frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
        frame = frame.dropna(subset=["date", "home_goals", "away_goals"])
    return frame


def save_snapshot(fixture: dict[str, Any], payload_json: str = "",
                  db_path: str | Path = DB_PATH):
    import datetime
    with connect(db_path) as conn:
        conn.execute("""
        INSERT INTO live_snapshots(captured_at, fixture_id, home_team, away_team, minute,
                                   home_score, away_score, status, payload_json)
        VALUES(?,?,?,?,?,?,?,?,?)
        """, (
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            str(fixture.get("id", "")), str(fixture.get("home_name", "")),
            str(fixture.get("away_name", "")), fixture.get("minute"),
            fixture.get("home_score"), fixture.get("away_score"),
            str(fixture.get("status", "")), payload_json
        ))


def save_final_result(fixture_id: str, home_team: str, away_team: str,
                      home_goals: int, away_goals: int, finished_at: str | None = None,
                      db_path: str | Path = DB_PATH):
    result = "1" if home_goals > away_goals else "X" if home_goals == away_goals else "2"
    with connect(db_path) as conn:
        conn.execute("""
        INSERT INTO final_results(fixture_id, finished_at, home_team, away_team, home_goals, away_goals, result)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(fixture_id) DO UPDATE SET finished_at=excluded.finished_at,
          home_goals=excluded.home_goals, away_goals=excluded.away_goals, result=excluded.result
        """, (str(fixture_id), finished_at, home_team, away_team,
              int(home_goals), int(away_goals), result))


def recent_snapshots(limit: int = 100, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM live_snapshots ORDER BY id DESC LIMIT ?",
            conn, params=(int(limit),)