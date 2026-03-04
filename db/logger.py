"""SQLite conversation + latency logger for AryaGPT."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).parent.parent / "aryagpt.db"))


def _get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist."""
    conn = _get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            ts          REAL NOT NULL,
            ts_human    TEXT NOT NULL,
            user_msg    TEXT NOT NULL,
            bot_msg     TEXT NOT NULL,
            provider    TEXT,
            model       TEXT,
            sources     TEXT,
            retrieval_ms REAL DEFAULT 0,
            llm_ms       REAL DEFAULT 0,
            total_ms     REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS evals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              REAL NOT NULL,
            session_id      TEXT,
            query           TEXT,
            retrieval_ms    REAL,
            llm_ms          REAL,
            total_ms        REAL,
            provider        TEXT,
            model           TEXT,
            chunks_returned INTEGER,
            answer_length   INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conv_ts      ON conversations(ts);
        CREATE INDEX IF NOT EXISTS idx_evals_ts     ON evals(ts);
    """)
    conn.commit()
    conn.close()


def log_conversation(
    session_id: str,
    user_msg: str,
    bot_msg: str,
    provider: str = "",
    model: str = "",
    sources: Optional[List[str]] = None,
    retrieval_ms: float = 0.0,
    llm_ms: float = 0.0,
    total_ms: float = 0.0,
    db_path: str = DB_PATH,
) -> None:
    conn = _get_conn(db_path)
    now = time.time()
    conn.execute(
        """INSERT INTO conversations
           (session_id, ts, ts_human, user_msg, bot_msg, provider, model, sources,
            retrieval_ms, llm_ms, total_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            now,
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)),
            user_msg,
            bot_msg,
            provider,
            model,
            json.dumps(sources or []),
            retrieval_ms,
            llm_ms,
            total_ms,
        ),
    )
    conn.commit()
    conn.close()


def log_eval(
    session_id: str,
    query: str,
    retrieval_ms: float,
    llm_ms: float,
    total_ms: float,
    provider: str,
    model: str,
    chunks_returned: int,
    answer_length: int,
    db_path: str = DB_PATH,
) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        """INSERT INTO evals
           (ts, session_id, query, retrieval_ms, llm_ms, total_ms,
            provider, model, chunks_returned, answer_length)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            time.time(),
            session_id,
            query,
            retrieval_ms,
            llm_ms,
            total_ms,
            provider,
            model,
            chunks_returned,
            answer_length,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_conversations(limit: int = 50, db_path: str = DB_PATH) -> List[Dict]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM conversations ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_eval_metrics(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Compute P50/P95 latency, counts, and per-provider breakdown."""
    conn = _get_conn(db_path)

    rows = conn.execute(
        "SELECT total_ms, retrieval_ms, llm_ms, provider, model FROM evals ORDER BY ts"
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "total_queries": 0,
            "p50_total_ms": 0,
            "p95_total_ms": 0,
            "p50_retrieval_ms": 0,
            "p95_retrieval_ms": 0,
            "p50_llm_ms": 0,
            "p95_llm_ms": 0,
            "avg_total_ms": 0,
            "provider_breakdown": {},
        }

    total_ms_vals = sorted([r["total_ms"] for r in rows])
    retr_ms_vals = sorted([r["retrieval_ms"] for r in rows])
    llm_ms_vals = sorted([r["llm_ms"] for r in rows])

    def percentile(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]

    provider_breakdown: Dict[str, Dict] = {}
    for r in rows:
        prov = r["provider"] or "unknown"
        if prov not in provider_breakdown:
            provider_breakdown[prov] = {"count": 0, "total_ms_sum": 0.0}
        provider_breakdown[prov]["count"] += 1
        provider_breakdown[prov]["total_ms_sum"] += r["total_ms"] or 0

    for prov in provider_breakdown:
        cnt = provider_breakdown[prov]["count"]
        provider_breakdown[prov]["avg_total_ms"] = (
            provider_breakdown[prov]["total_ms_sum"] / cnt if cnt else 0
        )

    return {
        "total_queries": len(rows),
        "p50_total_ms": percentile(total_ms_vals, 50),
        "p95_total_ms": percentile(total_ms_vals, 95),
        "p50_retrieval_ms": percentile(retr_ms_vals, 50),
        "p95_retrieval_ms": percentile(retr_ms_vals, 95),
        "p50_llm_ms": percentile(llm_ms_vals, 50),
        "p95_llm_ms": percentile(llm_ms_vals, 95),
        "avg_total_ms": sum(total_ms_vals) / len(total_ms_vals),
        "provider_breakdown": provider_breakdown,
    }
