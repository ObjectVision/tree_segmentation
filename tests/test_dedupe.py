# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Cross-tile deduplication and tile ownership."""

from shapely.geometry import Point, box

from tseg.geometry.dedupe import iou, nms, owns
from tseg.records import Feature


def _f(geom, score):
    return Feature(label="tree", score=score, backend="test",
                   mask=geom, bbox=geom)


def test_nms_keeps_highest_scoring_of_an_overlapping_pair():
    a = _f(box(0, 0, 10, 10), 0.9)
    b = _f(box(1, 1, 11, 11), 0.5)     # ~68% IoU with a
    far = _f(box(50, 50, 60, 60), 0.7)

    kept = nms([a, b, far], 0.4)

    assert len(kept) == 2
    assert {f.score for f in kept} == {0.9, 0.7}


def test_nms_keeps_neighbours_that_merely_touch():
    a = _f(box(0, 0, 10, 10), 0.9)
    b = _f(box(10, 0, 20, 10), 0.8)     # shares an edge, zero overlap

    assert len(nms([a, b], 0.4)) == 2


def test_nms_ignores_empty_geometries():
    assert nms([Feature(label="x", score=1.0, backend="t")], 0.4) == []


def test_iou_of_identical_boxes_is_one():
    assert iou(box(0, 0, 1, 1), box(0, 0, 1, 1)) == 1.0
    assert iou(box(0, 0, 1, 1), box(5, 5, 6, 6)) == 0.0


def test_tile_ownership_is_half_open():
    """A centroid on a shared edge belongs to exactly one tile, so an object
    on a boundary cannot be written twice."""
    core = (0, 0, 10, 10)

    assert owns(Point(0, 0), core)
    assert owns(Point(9.99, 9.99), core)
    assert not owns(Point(10, 5), core)
    assert not owns(Point(5, 10), core)
