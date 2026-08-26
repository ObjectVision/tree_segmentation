# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""The review store: dedupe, frozen holdout, uncertainty ordering, negatives."""

import pytest

from tseg.review.store import ACCEPT, PENDING, REJECT, ReviewStore
from tests.conftest import make_feature


@pytest.fixture
def store(tmp_path, features):
    s = ReviewStore(tmp_path / "review.db", holdout_fraction=0.2, seed=1337)
    s.add(features, round_no=0)
    yield s
    s.close()


def test_reinsert_is_idempotent(store, features):
    """Re-running detect over the same tiles must not duplicate candidates."""
    assert store.add(features, round_no=0) == 0
    assert store.stats()["total"] == 50


def test_holdout_is_deterministic_in_uid(tmp_path, features):
    """The split is hashed from the uid, so a candidate can never migrate
    between train and validation across runs."""
    a = ReviewStore(tmp_path / "a.db", holdout_fraction=0.2, seed=1337)
    b = ReviewStore(tmp_path / "b.db", holdout_fraction=0.2, seed=1337)
    a.add(features)
    b.add(list(reversed(features)))

    ha = {r["uid"] for r in a.db.execute("SELECT uid FROM candidates WHERE is_holdout=1")}
    hb = {r["uid"] for r in b.db.execute("SELECT uid FROM candidates WHERE is_holdout=1")}

    assert ha == hb and len(ha) > 0
    a.close()
    b.close()


def test_review_never_serves_holdout_rows(store):
    """If the loop could relabel its own validation set the metric would drift
    with the training data."""
    for row in store.batch(limit=50):
        assert row["is_holdout"] == 0


def test_batch_is_uncertainty_first(store):
    scores = [r["score"] for r in store.batch(limit=5)]
    assert scores == sorted(scores, key=lambda s: abs(s - 0.5))


def test_reject_is_a_hard_negative_not_a_deletion(store):
    row = store.batch(limit=1)[0]
    store.set_verdict(row["id"], REJECT)

    assert store.get(row["id"]) is not None
    assert store.stats()["reject"] == 1
    assert store.training_label(store.get(row["id"])) is None


def test_accept_yields_the_predicted_label(store):
    row = store.batch(limit=1)[0]
    store.set_verdict(row["id"], ACCEPT, "tree")
    assert store.training_label(store.get(row["id"])) == "tree"


def test_undo_returns_a_row_to_pending(store):
    row = store.batch(limit=1)[0]
    store.set_verdict(row["id"], ACCEPT, "tree")
    store.undo(row["id"])
    assert store.get(row["id"])["verdict"] == PENDING


def test_invalid_verdict_is_rejected(store):
    row = store.batch(limit=1)[0]
    with pytest.raises(ValueError, match="verdict must be"):
        store.set_verdict(row["id"], "maybe")


def test_manual_positive_is_accepted_immediately(store):
    """Hand-drawn objects are the only route to better recall, so they arrive
    already accepted."""
    before = store.stats()["accept"]
    store.add_manual(make_feature(999), true_label="tree", round_no=0)
    assert store.stats()["accept"] == before + 1


def test_feature_survives_the_roundtrip(store):
    row = store.batch(limit=1)[0]
    f = store.to_feature(row)
    assert f.label == "tree"
    assert f.circle.area > 0
    assert f.rect is not None
