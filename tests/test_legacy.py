# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Reading the pre-tseg tile cache.

6816 tiles of finished CPU work over Limburg live in that format. It is read,
never recomputed, so the reader has to keep working.
"""

import json

import pytest

from tseg.legacy import iter_legacy_cache, legacy_keys, read_legacy_tile

LEGACY_TILE = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature",
         "geometry": {"type": "Polygon", "coordinates": [[
             [173000.0, 316000.0], [173010.0, 316000.0],
             [173010.0, 316010.0], [173000.0, 316010.0],
             [173000.0, 316000.0]]]},
         "properties": {"score": 0.73}},
    ],
}


@pytest.fixture
def legacy_dir(tmp_path):
    (tmp_path / "173000_316000.geojson").write_text(json.dumps(LEGACY_TILE))
    return tmp_path


def test_legacy_box_gains_rect_and_circle(legacy_dir):
    feats = read_legacy_tile(legacy_dir / "173000_316000.geojson")
    assert len(feats) == 1

    f = feats[0]
    assert f.label == "tree"
    assert f.score == pytest.approx(0.73)
    assert f.tile_key == "173000_316000"
    assert f.area_m2 == pytest.approx(100.0)

    # No mask upstream, so rect degenerates to the box and the circle is the
    # equal-area equivalent - the same treatment any box-only backend gets.
    assert f.mask is None
    assert f.rect.equals(f.bbox)
    assert f.circle.area == pytest.approx(100.0, rel=0.01)


def test_score_is_lifted_out_of_properties(legacy_dir):
    f = read_legacy_tile(legacy_dir / "173000_316000.geojson")[0]
    assert "score" not in f.props


def test_iter_and_keys(legacy_dir):
    assert legacy_keys(legacy_dir) == ["173000_316000"]
    assert len(list(iter_legacy_cache(legacy_dir))) == 1
