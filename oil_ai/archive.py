"""ARCHIVE — a tiny SQLite store for analysed headlines (Stage 2 groundwork).

Every run that analyses news saves the headline text, the day it was
published, the driver it was tagged with, and the AI's impact score. Over
time this becomes the labelled historical news dataset that the "missing
link" news-fused model (Model C) needs — without it, news can only ever be
shown side-by-side, never fused into training.

Schema (single table):
    headlines(id, saved_at, day, pub_raw, driver, title, direction,
              score, horizon, uncertainty, model)
"""

import os
import sqlite3
from datetime import datetime, timezone

from .config import ARCHIVE_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_at    TEXT NOT NULL,
    day         TEXT NOT NULL,      -- grouped calendar day, e.g. 'Tue 13 Oct'
    pub_raw     TEXT,
    driver      TEXT NOT NULL,
    title       TEXT NOT NULL,
    direction   TEXT,
    score       INTEGER,
    horizon     TEXT,
    uncertainty TEXT,
    model       TEXT
);
"""


def connect(db_path=ARCHIVE_DB):
    """Open (and create) the archive database with its schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def save(analyses, model=None, db_path=ARCHIVE_DB):
    """Store one run's analysed headlines. Returns the number of rows saved."""
    if not analyses:
        return 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = [(now, a.get("day", ""), a.get("published", ""), a.get("driver", "other"),
             a.get("headline", ""), a.get("impact_direction", ""),
             int(a.get("impact_score", 0)), str(a.get("expected_horizon", "")),
             str(a.get("uncertainty", "")), model or "") for a in analyses]
    conn = connect(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO headlines(saved_at, day, pub_raw, driver, title, direction,"
                " score, horizon, uncertainty, model) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    finally:
        conn.close()
    return len(rows)


def stats(db_path=ARCHIVE_DB, limit_days=30):
    """Two useful summaries for the archive: score per published day and
    score per driver. Returns (rows_by_day, rows_by_driver)."""
    if not os.path.exists(db_path):
        return [], []
    conn = connect(db_path)
    try:
        by_day = conn.execute(
            "SELECT day, COUNT(*), AVG(score) FROM headlines"
            " GROUP BY day ORDER BY COUNT(*) DESC, day LIMIT ?", (limit_days,)).fetchall()
        by_driver = conn.execute(
            "SELECT driver, COUNT(*), AVG(score) FROM headlines"
            " GROUP BY driver ORDER BY ABS(AVG(score)) DESC").fetchall()
    finally:
        conn.close()
    return by_day, by_driver


def print_stats(db_path=ARCHIVE_DB):
    """CLI-friendly printout of what the archive has collected so far."""
    by_day, by_driver = stats(db_path)
    if not by_day:
        print("The news archive is empty — run the program once with news analysis.")
        return
    print(f"{'=' * 62}\n  NEWS ARCHIVE — labelled history for the future news-fused model")
    print(f"{'=' * 62}")
    total = sum(n for _d, n, _s in by_day)
    print(f"  {total} saved headlines. Average score per published day:")
    for day, n, avg in by_day[:10]:
        print(f"   {day:<14} {n:>3} headlines   average score {avg:+7.1f}")
    print("\n  Average score per driver so far:")
    for driver, n, avg in by_driver:
        print(f"   {driver:<22} {n:>3} headlines   average score {avg:+7.1f}")
