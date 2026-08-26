# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Find more of a rare class by similarity, not by text prompt.

The hard part of the riet task is not classifying a chip you are looking at.
It is *finding* thatch at all: it is a low single-digit percentage of Dutch
building stock and geographically clumped, so paging through panden in tile
order means reviewing hundreds of negatives per positive.

Open-vocabulary prompting is the usual answer, but the models that do it well
are not open source. Nearest-neighbour retrieval on frozen DINOv2 features is,
and for this problem it is the better tool anyway: thatch is a texture with no
crisp English name, and a handful of confirmed examples describes it far more
precisely than the word "thatch" ever will.

    tseg mine --out output/riet --like 0363100012061959 --limit 200
    tseg mine --out output/riet --label riet --limit 200    # all confirmed

Ranked candidates are promoted to the front of the review queue by raising
their score toward 0.5, which is where the uncertainty-first ordering in
ReviewStore.batch() looks first.
"""

from __future__ import annotations

import numpy as np

from tseg.review.store import ACCEPT, RELABEL, ReviewStore


def _load_chips(rows):
    """Read chip images for rows that have one on disk."""
    from pathlib import Path

    from PIL import Image

    chips, kept = [], []
    for row in rows:
        path = row["chip_path"]
        if not path or not Path(path).exists():
            continue
        chips.append(np.asarray(Image.open(path).convert("RGB")))
        kept.append(row)
    return chips, kept


def seeds(store: ReviewStore, label: str | None = None, uids=None):
    """Confirmed positives to search from: accepted or relabelled rows."""
    sql = ("SELECT * FROM candidates WHERE verdict IN (?, ?) "
           "AND chip_path IS NOT NULL")
    args = [ACCEPT, RELABEL]
    if label:
        sql += " AND COALESCE(true_label, pred_label) = ?"
        args.append(label)
    rows = list(store.db.execute(sql, args))

    if uids:
        wanted = set(uids)
        rows = [r for r in rows
                if r["uid"] in wanted
                or (r["uid"] or "").split(":")[-1] in wanted]
    return rows


def rank(store: ReviewStore, backend, label: str | None = None,
         like=None, limit: int = 200, promote: bool = True):
    """Rank unreviewed candidates by similarity to confirmed positives.

    Returns [(row, similarity)] sorted best first.
    """
    seed_rows = seeds(store, label=label, uids=like)
    if not seed_rows:
        raise SystemExit(
            "no confirmed positives to search from -- accept at least one "
            "example in 'tseg review' first, or pass --like with a pand id"
        )

    pending = list(store.db.execute(
        "SELECT * FROM candidates WHERE verdict = 'pending' AND is_holdout = 0 "
        "AND chip_path IS NOT NULL"))
    if not pending:
        return []

    seed_chips, seed_rows = _load_chips(seed_rows)
    cand_chips, pending = _load_chips(pending)
    if not seed_chips or not cand_chips:
        return []

    seed_feats = _normalise(backend.embed(seed_chips))
    cand_feats = _normalise(backend.embed(cand_chips))

    # Max similarity to any seed, not the mean: with a handful of examples the
    # centroid smears across sub-types (reed vs. combed wheat, hipped vs.
    # gabled) and matches none of them well.
    sims = (cand_feats @ seed_feats.T).max(axis=1)

    order = np.argsort(-sims)[:limit]
    ranked = [(pending[int(i)], float(sims[int(i)])) for i in order]

    if promote:
        # Nudge toward 0.5 so uncertainty-first ordering surfaces these next.
        # The stored score is a model output, so it is deliberately not
        # overwritten -- only moved, and only for rows we are recommending.
        store.db.executemany(
            "UPDATE candidates SET score = ? WHERE id = ?",
            [(0.5 + (1 - s) * 0.05, row["id"]) for row, s in ranked])
        store.db.commit()

    print(f"ranked {len(ranked)} candidate(s) against {len(seed_chips)} seed(s); "
          f"similarity {ranked[-1][1]:.3f}..{ranked[0][1]:.3f}"
          + (" (promoted in the review queue)" if promote else ""))
    return ranked


def _normalise(x: np.ndarray) -> np.ndarray:
    """L2-normalise so a dot product is cosine similarity."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n
