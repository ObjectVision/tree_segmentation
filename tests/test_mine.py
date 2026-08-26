# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Similarity mining: the open-source way to hunt a rare class.

The backend is stubbed. What matters here is the ranking maths and the queue
promotion, not DINOv2 itself.
"""

import numpy as np
import pytest
from PIL import Image

from tseg.review.mine import _normalise, rank, seeds
from tseg.review.store import ACCEPT, ReviewStore
from tests.conftest import make_feature


class StubBackend:
    """Embeds a chip as its mean RGB, so 'similar' means 'similar colour'."""

    def embed(self, chips):
        return np.array([[c[..., 0].mean(), c[..., 1].mean(), c[..., 2].mean()]
                         for c in chips], dtype=np.float64)


def _chip(tmp_path, name, colour):
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :] = colour
    path = tmp_path / f"{name}.jpg"
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture
def store(tmp_path):
    s = ReviewStore(tmp_path / "r.db", holdout_fraction=0.0, seed=1)

    # One confirmed positive, three lookalikes, three obvious negatives.
    plan = [("seed", (200, 40, 40))]
    plan += [(f"like{i}", (195 + i, 45, 42)) for i in range(3)]
    plan += [(f"other{i}", (20, 30, 220)) for i in range(3)]

    for i, (name, colour) in enumerate(plan):
        f = make_feature(i, score=0.9)
        f.props = {"chip": _chip(tmp_path, name, colour)}
        s.add([f], round_no=0)

    rows = list(s.db.execute("SELECT id FROM candidates ORDER BY id"))
    s.set_verdict(rows[0]["id"], ACCEPT, "riet")
    yield s
    s.close()


def test_seeds_are_confirmed_positives_only(store):
    got = seeds(store, label="riet")
    assert len(got) == 1
    assert got[0]["verdict"] == ACCEPT


def test_lookalikes_outrank_obvious_negatives(store):
    ranked = rank(store, StubBackend(), label="riet", limit=10, promote=False)

    assert len(ranked) == 6                      # the seed is no longer pending
    top3 = {r["chip_path"] for r, _ in ranked[:3]}
    assert all("like" in p for p in top3), "lookalikes should rank first"
    assert ranked[0][1] > ranked[-1][1]


def test_promotion_moves_candidates_up_the_review_queue(store):
    """Ranking is useless if the queue does not then serve them."""
    before = [r["id"] for r in store.batch(limit=3)]
    rank(store, StubBackend(), label="riet", limit=3, promote=True)
    after = [r["id"] for r in store.batch(limit=3)]

    assert after != before
    for row in store.batch(limit=3):
        assert abs(row["score"] - 0.5) < 0.06    # nudged toward the sort pivot


def test_no_confirmed_positives_is_an_actionable_error(tmp_path):
    s = ReviewStore(tmp_path / "empty.db")
    with pytest.raises(SystemExit, match="accept at least one"):
        rank(s, StubBackend(), label="riet")
    s.close()


def test_holdout_is_never_mined(store):
    """Mining promotes rows into review, so it must respect the frozen split."""
    store.db.execute("UPDATE candidates SET is_holdout = 1 WHERE pred_label = 'tree'")
    store.db.commit()
    assert rank(store, StubBackend(), label="riet", promote=False) == []


def test_normalise_makes_dot_product_cosine():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    n = _normalise(x)
    assert n[0] @ n[0] == pytest.approx(1.0)
    assert not np.isnan(n[1]).any()             # zero vector must not divide by 0
