"""Call history, number lists and operator feedback, in SQLite (standard library, one file).

Phone numbers are never stored in clear: a number is kept as a salted SHA-256 key (to match the
blacklist and count repeat callers) and a masked form for display ("0912***678"). Audio of flagged calls
is kept for review as a WAV file (see api.py, VG_RECORD); the table only holds whether it exists.
"""
import hashlib
import os
import secrets
import sqlite3
import threading
import time

from .prefilter import normalize

LEVEL_RANK = {None: -1, "LOW": 0, "MEDIUM": 1, "HIGH": 2}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS calls (
  id TEXT PRIMARY KEY, started_at REAL, ended_at REAL, status TEXT,
  caller_key TEXT, caller_masked TEXT, category TEXT, channel TEXT, agent TEXT, recording_ref TEXT,
  duration_s REAL, n_windows INTEGER DEFAULT 0, score REAL, spoof_percent REAL,
  risk_level TEXT, max_level TEXT, screened_by TEXT, recording INTEGER DEFAULT 0,
  action TEXT, action_at REAL, verdict TEXT, verdict_note TEXT, verdict_by TEXT, verdict_at REAL);
CREATE INDEX IF NOT EXISTS calls_started ON calls(started_at);
CREATE INDEX IF NOT EXISTS calls_caller ON calls(caller_key);
CREATE TABLE IF NOT EXISTS windows (call_id TEXT, start REAL, end REAL, score REAL, level TEXT, latency_ms REAL);
CREATE INDEX IF NOT EXISTS windows_call ON windows(call_id);
CREATE TABLE IF NOT EXISTS numbers (
  key TEXT, list TEXT, masked TEXT, added_at REAL, source TEXT, note TEXT, PRIMARY KEY (key, list));
"""

CALL_FIELDS = ("id", "started_at", "ended_at", "status", "caller_masked", "category", "channel", "agent",
               "recording_ref", "duration_s", "n_windows", "score", "spoof_percent", "risk_level", "max_level",
               "screened_by", "recording", "action", "action_at", "verdict", "verdict_note", "verdict_by", "verdict_at")


def mask(number):
    d = normalize(number)
    if not d:
        return "số ẩn" if (number or "").strip() else "không có số"
    return d[:4] + "***" + d[-3:] if len(d) > 7 else d[:2] + "***"


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            cols = {r["name"] for r in self.db.execute("PRAGMA table_info(calls)")}
            if "recording" not in cols:                                   # database from an older version
                self.db.execute("ALTER TABLE calls ADD COLUMN recording INTEGER DEFAULT 0")
            row = self.db.execute("SELECT v FROM meta WHERE k='salt'").fetchone()
            if row is None:
                self.salt = os.environ.get("VG_SALT") or secrets.token_hex(16)
                self.db.execute("INSERT INTO meta VALUES ('salt', ?)", (self.salt,))
                self.db.commit()
            else:
                self.salt = row["v"]

    # ---------- numbers ----------
    def key(self, number):
        d = normalize(number)
        return hashlib.sha256((self.salt + d).encode()).hexdigest()[:32] if d else None

    def add_number(self, number=None, lst="blacklist", source="manual", note="", key=None, masked=None):
        key = key or self.key(number)
        if not key:
            raise ValueError("số điện thoại không hợp lệ")
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO numbers VALUES (?,?,?,?,?,?)",
                            (key, lst, masked or mask(number), time.time(), source, note))
            self.db.commit()
        return key

    def remove_number(self, key, lst):
        with self.lock:
            n = self.db.execute("DELETE FROM numbers WHERE key=? AND list=?", (key, lst)).rowcount
            self.db.commit()
        return n

    def lists_of(self, key):
        if not key:
            return set()
        with self.lock:
            return {r["list"] for r in self.db.execute("SELECT list FROM numbers WHERE key=?", (key,))}

    def numbers(self, lst):
        with self.lock:
            rows = self.db.execute(
                "SELECT n.key, n.masked, n.added_at, n.source, n.note, "
                "(SELECT COUNT(*) FROM calls c WHERE c.caller_key=n.key) AS calls, "
                "(SELECT MAX(started_at) FROM calls c WHERE c.caller_key=n.key) AS last_call "
                "FROM numbers n WHERE n.list=? ORDER BY n.added_at DESC", (lst,)).fetchall()
        return [dict(r) for r in rows]

    def count_numbers(self):
        with self.lock:
            return {r["list"]: r["n"] for r in self.db.execute("SELECT list, COUNT(*) n FROM numbers GROUP BY list")}

    def suggestions(self, min_high=2, days=30):
        """Numbers with at least min_high calls that reached HIGH in the last days, not yet blacklisted."""
        since = time.time() - days * 86400
        with self.lock:
            rows = self.db.execute(
                "SELECT caller_key AS key, MAX(caller_masked) AS masked, COUNT(*) AS high_calls, "
                "MAX(started_at) AS last_call, SUM(verdict='fraud') AS fraud, SUM(verdict='genuine') AS genuine "
                "FROM calls WHERE caller_key IS NOT NULL AND max_level='HIGH' AND started_at>=? "
                "AND caller_key NOT IN (SELECT key FROM numbers WHERE list='blacklist') "
                "GROUP BY caller_key HAVING COUNT(*)>=? ORDER BY high_calls DESC", (since, min_high)).fetchall()
        return [dict(r) for r in rows]

    # ---------- calls ----------
    def start_call(self, call_id, number, category, channel, agent="", recording_ref=""):
        with self.lock:
            self.db.execute("INSERT INTO calls (id, started_at, status, caller_key, caller_masked, category, channel, "
                            "agent, recording_ref) VALUES (?,?,?,?,?,?,?,?,?)",
                            (call_id, time.time(), "active", self.key(number), mask(number), category, channel,
                             agent, recording_ref))
            self.db.commit()

    def update_call(self, call_id, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.lock:
            self.db.execute(f"UPDATE calls SET {cols} WHERE id=?", (*fields.values(), call_id))
            self.db.commit()

    def add_windows(self, call_id, rows):
        with self.lock:
            self.db.executemany("INSERT INTO windows VALUES (?,?,?,?,?,?)",
                                [(call_id, r["start"], r["end"], r["score"], r["level"], r.get("latency_ms"))
                                 for r in rows])
            self.db.commit()

    def get_call(self, call_id):
        with self.lock:
            row = self.db.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
            if row is None:
                return None
            wins = self.db.execute("SELECT start, end, score, level FROM windows WHERE call_id=? ORDER BY start",
                                   (call_id,)).fetchall()
            same = self.db.execute("SELECT COUNT(*) n, SUM(max_level='HIGH') high FROM calls WHERE caller_key=? "
                                   "AND caller_key IS NOT NULL", (row["caller_key"],)).fetchone()
        out = {k: row[k] for k in CALL_FIELDS}
        out["windows"] = [dict(w) for w in wins]
        out["caller_history"] = dict(calls=same["n"] or 0, high=same["high"] or 0)
        out["caller_lists"] = sorted(self.lists_of(row["caller_key"]))
        return out

    def caller_key_of(self, call_id):
        with self.lock:
            row = self.db.execute("SELECT caller_key, caller_masked FROM calls WHERE id=?", (call_id,)).fetchone()
        return (row["caller_key"], row["caller_masked"]) if row else (None, None)

    def list_calls(self, status=None, min_level=None, unreviewed=False, limit=100, since=None):
        q, args = "SELECT " + ", ".join(CALL_FIELDS) + " FROM calls WHERE 1=1", []
        if status:
            q += " AND status=?"
            args.append(status)
        if min_level:
            q += " AND max_level IN (" + ",".join("?" * len([l for l, r in LEVEL_RANK.items() if l and r >= LEVEL_RANK[min_level]])) + ")"
            args += [l for l, r in LEVEL_RANK.items() if l and r >= LEVEL_RANK[min_level]]
        if unreviewed:
            q += " AND verdict IS NULL"
        if since:
            q += " AND started_at>=?"
            args.append(since)
        q += " ORDER BY started_at DESC LIMIT ?"
        args.append(int(limit))
        with self.lock:
            return [dict(r) for r in self.db.execute(q, args)]

    def stats(self, since):
        with self.lock:
            r = self.db.execute(
                "SELECT COUNT(*) total, SUM(status='active') active, SUM(max_level='HIGH') high, "
                "SUM(max_level='MEDIUM') medium, SUM(max_level='LOW') low, SUM(screened_by='prefilter') blocked, "
                "SUM(verdict='fraud') fraud, SUM(verdict='genuine') genuine, "
                "SUM(verdict='genuine' AND max_level IN ('HIGH','MEDIUM')) false_alarm, "
                "SUM(max_level IN ('HIGH','MEDIUM') AND verdict IS NULL) pending, "
                "SUM(action='hold') held, SUM(action='transfer') transferred "
                "FROM calls WHERE started_at>=?", (since,)).fetchone()
            lat = [x[0] for x in self.db.execute(
                "SELECT latency_ms FROM windows w JOIN calls c ON c.id=w.call_id WHERE c.started_at>=? "
                "AND latency_ms IS NOT NULL ORDER BY latency_ms", (since,))]
            hours = [dict(x) for x in self.db.execute(
                "SELECT CAST((started_at-?)/3600 AS INT) h, COUNT(*) n, SUM(max_level='HIGH') high, "
                "SUM(max_level='MEDIUM') medium FROM calls WHERE started_at>=? GROUP BY h ORDER BY h",
                (since, since))]
        out = {k: (r[k] or 0) for k in r.keys()}
        out["latency_p95_ms"] = round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else None
        out["by_hour"] = hours
        return out
