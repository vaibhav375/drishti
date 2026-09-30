"""Local SQLite store for TLEs and SATCAT metadata (F0.1).

Design notes
------------
* Dedupe is enforced at the schema level: UNIQUE(norad_id, epoch).
  Ingesting the same TLE twice is a no-op (INSERT OR IGNORE).
* Everything is plain SQLite — single file, no server, laptop-friendly.
* Epochs are stored as ISO-8601 UTC strings so they sort correctly
  and stay human-readable in any SQLite browser.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS tle (
    id          INTEGER PRIMARY KEY,
    norad_id    INTEGER NOT NULL,
    epoch       TEXT    NOT NULL,          -- ISO-8601 UTC
    line1       TEXT    NOT NULL,
    line2       TEXT    NOT NULL,
    source      TEXT    NOT NULL,          -- 'celestrak' | 'spacetrack'
    ingested_at TEXT    NOT NULL,
    UNIQUE (norad_id, epoch)
);
CREATE INDEX IF NOT EXISTS idx_tle_norad_epoch ON tle (norad_id, epoch);

CREATE TABLE IF NOT EXISTS satcat (
    norad_id     INTEGER PRIMARY KEY,
    intl_desig   TEXT,
    name         TEXT,
    country      TEXT,
    launch_date  TEXT,
    decay_date   TEXT,
    object_type  TEXT,     -- PAY / R/B / DEB / UNK
    rcs_m2       REAL,     -- radar cross-section, m^2 (Celestrak 'RCS' column;
                            -- ~47% coverage live, confirmed 2026-07-15 — NULL
                            -- when unpublished, never fabricated)
    period_min   REAL,
    inclination  REAL,
    apogee_km    REAL,
    perigee_km   REAL,
    updated_at   TEXT
);

-- Catalog snapshots power breakup detection (F1.3): a burst of new
-- norad_ids sharing a regime between snapshots = candidate fragmentation.
CREATE TABLE IF NOT EXISTS catalog_snapshot (
    snapshot_at TEXT NOT NULL,
    norad_id    INTEGER NOT NULL,
    PRIMARY KEY (snapshot_at, norad_id)
);

-- Watch agent: the analyst's watchlist, each run's findings, and the
-- per-object state snapshot used to diff one cycle against the next.
CREATE TABLE IF NOT EXISTS watch_list (
    norad_id INTEGER PRIMARY KEY,
    added_at TEXT NOT NULL,
    note     TEXT
);
CREATE TABLE IF NOT EXISTS watch_cycle (
    cycle_id   INTEGER PRIMARY KEY,
    run_at     TEXT NOT NULL,
    refreshed  INTEGER NOT NULL,        -- 1 if TLEs were pulled fresh this cycle
    n_findings INTEGER NOT NULL,
    brief      TEXT
);
CREATE TABLE IF NOT EXISTS watch_finding (
    cycle_id     INTEGER NOT NULL,
    norad_id     INTEGER,
    kind         TEXT NOT NULL,
    significance TEXT NOT NULL,          -- priority | elevated | routine
    summary      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_watch_finding_cycle ON watch_finding (cycle_id);
CREATE TABLE IF NOT EXISTS watch_object_state (
    norad_id            INTEGER PRIMARY KEY,   -- latest state, overwritten each cycle
    cycle_id            INTEGER NOT NULL,
    threat_score        REAL,
    threat_band         TEXT,
    last_maneuver_epoch TEXT,
    n_maneuvers         INTEGER,
    rpo_label           TEXT,
    rpo_min_range_km    REAL,
    decay_reentry_iso   TEXT,
    disposal_status     TEXT
);
"""


@dataclass(frozen=True)
class TleRecord:
    norad_id: int
    epoch: str  # ISO-8601 UTC
    line1: str
    line2: str
    source: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _migrate(conn: sqlite3.Connection) -> None:
    """Lightweight ALTER-TABLE migrations for existing local DBs.

    CREATE TABLE IF NOT EXISTS never adds columns to an already-created
    table, so a schema change (like rcs_class -> rcs_m2, 2026-07-15: the
    live Celestrak SATCAT column is 'RCS', a numeric m^2 value, not the
    legacy SMALL/MEDIUM/LARGE category the old column assumed) needs an
    explicit migration or every local DB stays stuck on the old shape.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(satcat)")}
    if "rcs_m2" not in cols:
        conn.execute("ALTER TABLE satcat ADD COLUMN rcs_m2 REAL")


_INITIALIZED: set[str] = set()


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # schema + migrations once per DB per process — running them on every
        # connection made catalogue-wide loops ~10x slower (found 2026-09-30:
        # 4.9 of a dossier's 5.4 s was re-running SCHEMA 16k times)
        key = str(Path(db_path).resolve())
        if key not in _INITIALIZED:
            conn.executescript(SCHEMA)
            _migrate(conn)
            _INITIALIZED.add(key)
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_tles(records: Iterable[TleRecord], db_path: Path = DB_PATH) -> int:
    """Insert TLE records; duplicates on (norad_id, epoch) are ignored.

    Returns the number of NEW rows actually inserted.
    """
    now = _utcnow_iso()
    inserted = 0
    with connect(db_path) as conn:
        for r in records:
            cur = conn.execute(
                "INSERT OR IGNORE INTO tle "
                "(norad_id, epoch, line1, line2, source, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (r.norad_id, r.epoch, r.line1, r.line2, r.source, now),
            )
            inserted += cur.rowcount
    return inserted


def get_tles(
    norad_id: int,
    start: Optional[str] = None,
    end: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> list[TleRecord]:
    """All stored TLEs for an object, chronological, optionally windowed."""
    q = "SELECT norad_id, epoch, line1, line2, source FROM tle WHERE norad_id = ?"
    args: list = [norad_id]
    if start:
        q += " AND epoch >= ?"
        args.append(start)
    if end:
        q += " AND epoch <= ?"
        args.append(end)
    q += " ORDER BY epoch ASC"
    with connect(db_path) as conn:
        rows = conn.execute(q, args).fetchall()
    return [TleRecord(**dict(row)) for row in rows]


def get_latest_tle(norad_id: int, db_path: Path = DB_PATH) -> Optional[TleRecord]:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT norad_id, epoch, line1, line2, source FROM tle "
            "WHERE norad_id = ? ORDER BY epoch DESC LIMIT 1",
            (norad_id,),
        ).fetchone()
    return TleRecord(**dict(row)) if row else None


def get_latest_tles_many(norad_ids: Iterable[int], db_path: Path = DB_PATH) -> dict[int, TleRecord]:
    """Latest TLE per object for many objects in ONE query (bulk form of
    get_latest_tle for catalogue-wide loops). Missing objects are absent."""
    wanted = set(norad_ids)
    if not wanted:
        return {}
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT t.norad_id, t.epoch, t.line1, t.line2, t.source FROM tle t "
            "JOIN (SELECT norad_id, MAX(epoch) AS e FROM tle GROUP BY norad_id) m "
            "ON m.norad_id = t.norad_id AND m.e = t.epoch"
        ).fetchall()
    return {r["norad_id"]: TleRecord(**dict(r)) for r in rows if r["norad_id"] in wanted}


def list_objects(db_path: Path = DB_PATH) -> list[dict]:
    """Objects present in the TLE store, with names when SATCAT has them."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT t.norad_id,
                   COALESCE(s.name, 'NORAD ' || t.norad_id) AS name,
                   COUNT(*) AS n_tles,
                   MAX(t.epoch) AS latest_epoch
            FROM tle t LEFT JOIN satcat s ON s.norad_id = t.norad_id
            GROUP BY t.norad_id ORDER BY name
            """
        ).fetchall()
    return [dict(r) for r in rows]


def record_catalog_snapshot(norad_ids: Iterable[int], db_path: Path = DB_PATH) -> str:
    """Record which objects exist in the catalog right now (for F1.3)."""
    ts = _utcnow_iso()
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO catalog_snapshot (snapshot_at, norad_id) VALUES (?, ?)",
            [(ts, nid) for nid in norad_ids],
        )
    return ts


def list_snapshot_timestamps(db_path: Path = DB_PATH) -> list[str]:
    """All recorded snapshot times, chronological."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT snapshot_at FROM catalog_snapshot ORDER BY snapshot_at"
        ).fetchall()
    return [r[0] for r in rows]


def get_snapshot_ids(snapshot_at: str, db_path: Path = DB_PATH) -> set[int]:
    """norad_ids present in one recorded snapshot."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT norad_id FROM catalog_snapshot WHERE snapshot_at = ?", (snapshot_at,)
        ).fetchall()
    return {r["norad_id"] for r in rows}


def list_current_objects(source: str = "celestrak", db_path: Path = DB_PATH) -> list[dict]:
    """norad_ids with a live-sourced TLE (default: Celestrak, no auth) and
    their latest epoch — the "current catalog" pool, as opposed to
    historical archive entries kept around for event validation."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT norad_id, MAX(epoch) AS latest_epoch FROM tle "
            "WHERE source = ? GROUP BY norad_id",
            (source,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_active_satcat_norad_ids(db_path: Path = DB_PATH) -> list[int]:
    """norad_ids currently in SATCAT with no decay_date — "up there right
    now", the universe catalog_snapshot should track for F1.3."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT norad_id FROM satcat WHERE decay_date IS NULL"
        ).fetchall()
    return [r["norad_id"] for r in rows]


def list_high_value_asset_ids(exclude_norad_id: Optional[int] = None, db_path: Path = DB_PATH) -> list[int]:
    """Live-tracked objects SATCAT marks as PAY — the data-driven "high
    value asset" pool shared by F1.4 (score/threat.py) and F3.2
    (detect/graph.py). Not exhaustive: only covers whatever Celestrak
    groups have been ingest-celestrak'd (stations/geo/gps-ops by
    convention). Expand by ingesting more groups.
    """
    current = list_current_objects(db_path=db_path)
    candidate_ids = [c["norad_id"] for c in current if c["norad_id"] != exclude_norad_id]
    satcat = get_satcat_many(candidate_ids, db_path=db_path)
    return [nid for nid in candidate_ids if satcat.get(nid, {}).get("object_type") == "PAY"]


def get_satcat_by_launch_prefix(prefix: str, db_path: Path = DB_PATH) -> list[dict]:
    """All catalogued objects sharing an 8-char launch designator (YYYY-NNN).

    intl_desig is fixed-width YYYY-NNN + variable-length piece letter(s)
    (e.g. '1982-092A', '1982-092AAH'), so the first 8 characters are always
    exactly the launch designator — no separate join table needed.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM satcat WHERE substr(intl_desig, 1, 8) = ? ORDER BY intl_desig",
            (prefix,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_satcat_many(norad_ids: Iterable[int], db_path: Path = DB_PATH) -> dict[int, dict]:
    """SATCAT rows for a specific set of norad_ids, keyed by norad_id."""
    ids = list(norad_ids)
    if not ids:
        return {}
    result: dict[int, dict] = {}
    # SQLite has a variable limit (~999); batch to stay under it.
    batch_size = 900
    with connect(db_path) as conn:
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT * FROM satcat WHERE norad_id IN ({placeholders})", chunk
            ).fetchall()
            for r in rows:
                result[r["norad_id"]] = dict(r)
    return result


def upsert_satcat(rows: Iterable[dict], db_path: Path = DB_PATH) -> int:
    now = _utcnow_iso()
    n = 0
    with connect(db_path) as conn:
        for r in rows:
            conn.execute(
                """
                INSERT INTO satcat (norad_id, intl_desig, name, country,
                                    launch_date, decay_date, object_type,
                                    rcs_m2, period_min, inclination,
                                    apogee_km, perigee_km, updated_at)
                VALUES (:norad_id, :intl_desig, :name, :country,
                        :launch_date, :decay_date, :object_type,
                        :rcs_m2, :period_min, :inclination,
                        :apogee_km, :perigee_km, :updated_at)
                ON CONFLICT(norad_id) DO UPDATE SET
                    name=excluded.name, country=excluded.country,
                    decay_date=excluded.decay_date,
                    object_type=excluded.object_type,
                    rcs_m2=excluded.rcs_m2,
                    period_min=excluded.period_min,
                    inclination=excluded.inclination,
                    apogee_km=excluded.apogee_km,
                    perigee_km=excluded.perigee_km,
                    updated_at=excluded.updated_at
                """,
                {**r, "updated_at": now},
            )
            n += 1
    return n


# ---------------------------------------------------------------------------
# Watch agent persistence
# ---------------------------------------------------------------------------

def add_to_watchlist(norad_id: int, note: str = "", db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO watch_list (norad_id, added_at, note) VALUES (?, ?, ?) "
            "ON CONFLICT(norad_id) DO UPDATE SET note=excluded.note",
            (norad_id, _utcnow_iso(), note),
        )


def remove_from_watchlist(norad_id: int, db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM watch_list WHERE norad_id = ?", (norad_id,))


def get_watchlist(db_path: Path = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM watch_list ORDER BY added_at").fetchall()
    return [dict(r) for r in rows]


def is_watched(norad_id: int, db_path: Path = DB_PATH) -> bool:
    with connect(db_path) as conn:
        return conn.execute("SELECT 1 FROM watch_list WHERE norad_id = ?",
                            (norad_id,)).fetchone() is not None


def get_last_object_states(db_path: Path = DB_PATH) -> dict[int, dict]:
    """The most-recent per-object state, keyed by norad_id — the baseline a
    new cycle diffs against."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM watch_object_state").fetchall()
    return {r["norad_id"]: dict(r) for r in rows}


def save_watch_cycle(run_at: str, refreshed: bool, brief: str,
                     findings: list[dict], states: list[dict],
                     db_path: Path = DB_PATH) -> int:
    """Persist a completed cycle: the cycle row, its findings, and the
    updated per-object state snapshots (overwriting the prior baseline)."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO watch_cycle (run_at, refreshed, n_findings, brief) VALUES (?, ?, ?, ?)",
            (run_at, int(refreshed), len(findings), brief),
        )
        cycle_id = cur.lastrowid
        for f in findings:
            conn.execute(
                "INSERT INTO watch_finding (cycle_id, norad_id, kind, significance, summary) "
                "VALUES (?, ?, ?, ?, ?)",
                (cycle_id, f.get("norad_id"), f["kind"], f["significance"], f["summary"]),
            )
        for s in states:
            conn.execute(
                "INSERT INTO watch_object_state (norad_id, cycle_id, threat_score, threat_band, "
                "last_maneuver_epoch, n_maneuvers, rpo_label, rpo_min_range_km, decay_reentry_iso, "
                "disposal_status) VALUES (:norad_id, :cycle_id, :threat_score, :threat_band, "
                ":last_maneuver_epoch, :n_maneuvers, :rpo_label, :rpo_min_range_km, "
                ":decay_reentry_iso, :disposal_status) "
                "ON CONFLICT(norad_id) DO UPDATE SET cycle_id=excluded.cycle_id, "
                "threat_score=excluded.threat_score, threat_band=excluded.threat_band, "
                "last_maneuver_epoch=excluded.last_maneuver_epoch, n_maneuvers=excluded.n_maneuvers, "
                "rpo_label=excluded.rpo_label, rpo_min_range_km=excluded.rpo_min_range_km, "
                "decay_reentry_iso=excluded.decay_reentry_iso, disposal_status=excluded.disposal_status",
                {**s, "cycle_id": cycle_id},
            )
    return cycle_id


def get_watch_cycles(limit: int = 20, db_path: Path = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM watch_cycle ORDER BY cycle_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_cycle_findings(cycle_id: int, db_path: Path = DB_PATH) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM watch_finding WHERE cycle_id = ?", (cycle_id,)
        ).fetchall()
    return [dict(r) for r in rows]
