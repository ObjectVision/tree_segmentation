# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Shape derivation: a rectangle and a circle from every detection."""

import math

import numpy as np
import pytest

from tseg.geometry.shapes import (
    calc_circle_bbox,
    calc_rectangle_bbox,
    derive_shapes,
    mask_to_polygon,
    passes_size_filter,
)


def test_equal_area_circle_matches_mask_area(square_mask):
    mask, bbox = square_mask
    s = derive_shapes(mask, bbox, "equal_area")

    assert s["area_px"] == 400
    assert s["circle_params"][2] == pytest.approx(math.sqrt(400 / math.pi), rel=1e-6)

    # The circle is drawn as a 32-gon, so its polygon area sits just under the
    # true circle area by the known discretisation factor - not by more.
    ratio = s["circle"].area / s["area_px"]
    assert 0.99 < ratio < 1.0


def test_min_enclosing_overestimates_merged_crowns(merged_crowns_mask):
    """Why equal_area is the default.

    When two crowns merge into one mask, the minimum enclosing circle spans
    both and reports a radius far larger than either tree. This is a routine
    outcome in dense canopy, not a corner case.
    """
    mask, bbox = merged_crowns_mask
    eq = derive_shapes(mask, bbox, "equal_area")["circle_params"][2]
    mn = derive_shapes(mask, bbox, "min_enclosing")["circle_params"][2]

    assert mn > eq * 2, f"expected a large overestimate, got {mn:.1f} vs {eq:.1f}"


def test_box_only_backend_degenerates_rect_to_bbox():
    """DeepForest returns no mask; rect must fall back to the box."""
    s = derive_shapes(None, (10.0, 10.0, 30.0, 30.0), "equal_area")

    assert s["mask"] is None
    assert s["rect"].equals(s["bbox"])
    assert s["area_px"] == pytest.approx(400.0)


def test_rotated_rect_is_tighter_than_bbox():
    """A diagonal mask is where minAreaRect earns its keep."""
    m = np.zeros((80, 80), dtype=bool)
    for i in range(10, 70):
        m[i - 4:i + 4, i - 4:i + 4] = True
    s = derive_shapes(m, (6.0, 6.0, 74.0, 74.0), "equal_area")

    assert s["rect"].area < s["bbox"].area


def test_unknown_circle_method_is_rejected(square_mask):
    mask, bbox = square_mask
    with pytest.raises(ValueError, match="equal_area or min_enclosing"):
        derive_shapes(mask, bbox, "nearest_pub")


def test_mask_to_polygon_returns_none_for_empty():
    assert mask_to_polygon(np.zeros((10, 10), dtype=bool)) is None


def test_size_filter_drops_tiny_and_slivers():
    assert passes_size_filter((0, 0, 20, 20), min_size=8, min_ratio=0.2)
    assert not passes_size_filter((0, 0, 4, 4), min_size=8, min_ratio=0.2)
    assert not passes_size_filter((0, 0, 40, 2), min_size=1, min_ratio=0.2)


# --- labelme ingest, ported from urban-tree (MIT) --------------------------

def test_calc_circle_bbox_from_centre_and_rim():
    got = calc_circle_bbox([[50, 50], [50, 60]], 100, 100)
    assert got == {"xmin": 40.0, "ymin": 40.0, "xmax": 60.0, "ymax": 60.0}


def test_calc_circle_bbox_clips_to_image():
    got = calc_circle_bbox([[5, 5], [5, 25]], 100, 100)
    assert got["xmin"] == 0 and got["ymin"] == 0


def test_calc_rectangle_bbox():
    got = calc_rectangle_bbox([[10, 10], [40, 40]], 100, 100)
    assert got == {"xmin": 10, "ymin": 10, "xmax": 40, "ymax": 40}
