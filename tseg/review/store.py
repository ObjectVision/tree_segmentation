# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""SQLite annotation store -- the single source of truth for the loop.

Every candidate the pipeline produces lands here, gets a verdict from you, and
is read back by the COCO exporter. Both modes (tile and pand) and hand-drawn
corrections share one table so the exporter does not care where a row came from.

Two design points that matter for training quality:

  * Rejects are kept, not deleted. A rejected detection is a hard negative and
    is worth as much as a positive -- deleting it throws away the signal that
    stops the model making the same mistake next round.
  * The holdout is assigned once, deterministically, at insert time, and the
    review UI never serves holdout rows. If the loop could relabel its own
    validation set the metric would drift with the training data and the
    numbers would stop meaning anything.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tseg.records import Feature

PENDING, ACCEPT, REJECT, RELABEL = "pending", "accept", "reject", "relabel"
VERDICTS = (PENDING, ACCEPT, REJECT, RELABEL)

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    round        INTEGER NOT NULL DEFAULT 0,
    source       TEXT    NOT NULL DEFAULT 'tile',
    tile_key     TEXT,
    uid          TEXT    UNIQUE,
    backend      TEXT,
    score        REAL,
    pred_label   TEXT,
    verdict      TEXT    NOT NULL DEFAULT 'pending',
    true_label   TEXT,
    is_holdout   INTEGER NOT NULL DEFAULT 0,
    chip_path    TEXT,
    geom_wkt     TEXT,
    bbox_wkt     TEXT,
    rect_wkt     TEXT,
    circle_wkt   TEXT,
    feature_json TEXT,
    created_at   TEXT,
    reviewed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_verdict  ON candidates(verdict);
CREATE INDEX IF NOT EXISTS idx_round    ON candidates(round);
CREATE INDEX IF NOT EXISTS idx_holdout  ON candidates(is_holdout);
CREATE INDEX IF NOT EXISTS idx_tile     ON candidates(tile_key);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    def __init__(self, path, holdout_fraction: float = 0.2, seed: int = 1337):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.holdout_fraction = holdout_fraction
        self.seed = seed
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------- holdout
    def _holdout_flag(self, uid: str) -> int:
        """Stable hash-based split. Deterministic in uid, so re-inserting the
        same candidate can never move it between train and holdout."""
        h = hashlib.sha1(f"{self.seed}:{uid}".encode()).digest()
        bucket = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
        return int(bucket < self.holdout_fraction)

    @staticmethod
    def uid_for(feature: Feature, source: str) -> str:
        ident = (feature.props or {}).get("identificatie")
        if ident:
            return f"pand:{ident}"
        return (f"{source}:{feature.tile_key}:"
                f"{feature.cx:.2f}:{feature.cy:.2f}:{feature.label}")

    # --------------------------------------------------------------- insert
    def add(self, features, round_no: int = 0, source: str = "tile") -> int:
        rows = []
        for f in features:
            uid = self.uid_for(f, source)
            rows.append((
                round_no, source, f.tile_key, uid, f.backend, float(f.score),
                f.label, PENDING, None, self._holdout_flag(uid),
                (f.props or {}).get("chip"),
                f.geometry.wkt if f.geometry is not None else None,
                f.bbox.wkt if f.bbox is not None else None,
                f.rect.wkt if f.rect is not None else None,
                f.circle.wkt if f.circle is not None else None,
                json.dumps(f.to_json()), _now(),
            ))
        cur = self.db.executemany(
            """INSERT OR IGNORE INTO candidates
               (round, source, tile_key, uid, backend, score, pred_label,
                verdict, true_label, is_holdout, chip_path, geom_wkt, bbox_wkt,
                rect_wkt, circle_wkt, feature_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.db.commit()
        return cur.rowcount

    def add_manual(self, feature: Feature, true_label: str, round_no: int = 0,
                   chip_path: str | None = None) -> int:
        """A object the model missed, drawn by hand in the review UI.

        This is what makes the loop able to improve recall. Accept/reject alone
        can only suppress false positives; without hand-drawn positives the
        model never learns about the objects it failed to propose, and recall
        plateaus.
        """
        uid = f"manual:{feature.tile_key}:{feature.cx:.2f}:{feature.cy:.2f}"
        cur = self.db.execute(
            """INSERT OR IGNORE INTO candidates
               (round, source, tile_key, uid, backend, score, pred_label,
                verdict, true_label, is_holdout, chip_path, geom_wkt, bbox_wkt,
                rect_wkt, circle_wkt, feature_json, created_at, reviewed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (round_no, "manual", feature.tile_key, uid, "human", 1.0,
             true_label, ACCEPT, true_label, self._holdout_flag(uid),
             chip_path,
             feature.geometry.wkt if feature.geometry is not None else None,
             feature.bbox.wkt if feature.bbox is not None else None,
             feature.rect.wkt if feature.rect is not None else None,
             feature.circle.wkt if feature.circle is not None else None,
             json.dumps(feature.to_json()), _now(), _now()),
        )
        self.db.commit()
        return cur.lastrowid

    # --------------------------------------------------------------- review
    def batch(self, limit: int = 24, sort: str = "uncertainty",
              round_no: int | None = None, source: str | None = None):
        """Next candidates to review. Holdout rows are never served."""
        where = ["verdict = ?", "is_holdout = 0"]
        args: list = [PENDING]
        if round_no is not None:
            where.append("round = ?")
            args.append(round_no)
        if source:
            where.append("source = ?")
            args.append(source)

        if sort == "uncertainty":
            # |score - 0.5| ascending. Reviewing the model confident hits
            # teaches it almost nothing; the boundary cases are where each
            # label buys the most.
            order = "ABS(score - 0.5) ASC"
        elif sort == "score":
            order = "score DESC"
        elif sort == "random":
            order = "RANDOM()"
        else:
            raise ValueError(
                "sort must be uncertainty | score | random, got " + repr(sort)
            )

        args.append(limit)
        return list(self.db.execute(
            f"SELECT * FROM candidates WHERE {' AND '.join(where)} "
            f"ORDER BY {order} LIMIT ?", args))

    def set_verdict(self, row_id: int, verdict: str, true_label=None):
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
        self.db.execute(
            "UPDATE candidates SET verdict=?, true_label=?, reviewed_at=? WHERE id=?",
            (verdict, true_label, _now(), row_id))
        self.db.commit()

    def set_verdicts(self, pairs):
        """pairs: iterable of (row_id, verdict, true_label_or_None)."""
        self.db.executemany(
            "UPDATE candidates SET verdict=?, true_label=?, reviewed_at=? WHERE id=?",
            [(v, t, _now(), i) for i, v, t in pairs])
        self.db.commit()

    def undo(self, row_id: int):
        self.db.execute(
            "UPDATE candidates SET verdict=?, true_label=NULL, reviewed_at=NULL "
            "WHERE id=?", (PENDING, row_id))
        self.db.commit()

    def get(self, row_id: int):
        return self.db.execute(
            "SELECT * FROM candidates WHERE id=?", (row_id,)).fetchone()

    # ---------------------------------------------------------------- stats
    def stats(self, round_no: int | None = None) -> dict:
        where, args = ("WHERE round = ?", [round_no]) if round_no is not None else ("", [])
        rows = self.db.execute(
            f"SELECT verdict, COUNT(*) n FROM candidates {where} GROUP BY verdict",
            args).fetchall()
        out = {v: 0 for v in VERDICTS}
        out.update({r["verdict"]: r["n"] for r in rows})
        out["total"] = sum(out[v] for v in VERDICTS)
        out["reviewed"] = out["total"] - out[PENDING]
        out["holdout"] = self.db.execute(
            f"SELECT COUNT(*) n FROM candidates {where or 'WHERE 1=1'} "
            f"{'AND' if where else 'AND'} is_holdout = 1", args).fetchone()["n"]
        return out

    def rounds(self) -> list[int]:
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT round FROM candidates ORDER BY round")]

    def max_round(self) -> int:
        r = self.db.execute("SELECT MAX(round) m FROM candidates").fetchone()
        return int(r["m"] or 0)

    # --------------------------------------------------------------- export
    def labelled(self, holdout: bool | None = None, source: str | None = None):
        """Rows with a usable training label.

        Accepts and relabels are positives; rejects are hard negatives and come
        through with true_label NULL so the exporter can emit an image with no
        annotation for them.
        """
        where = ["verdict != ?"]
        args: list = [PENDING]
        if holdout is not None:
            where.append("is_holdout = ?")
            args.append(int(holdout))
        if source:
            where.append("source = ?")
            args.append(source)
        return list(self.db.execute(
            f"SELECT * FROM candidates WHERE {' AND '.join(where)} ORDER BY id",
            args))

    @staticmethod
    def to_feature(row) -> Feature:
        return Feature.from_json(json.loads(row["feature_json"]))

    @staticmethod
    def training_label(row):
        """Effective label for a reviewed row, or None if it is a negative."""
        if row["verdict"] == REJECT:
            return None
        return row["true_label"] or row["pred_label"]

    # ----------------------------------------------------------------- meta
    def set_meta(self, key: str, value):
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
        self.db.commit()

    def get_meta(self, key, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default
