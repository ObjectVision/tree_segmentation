# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Shared fixtures.

torch is imported before anything that pulls in GDAL: on Windows, loading
GDAL/OpenMP DLLs first makes torch's c10.dll fail with WinError 1114. The same
guard lives in tseg/__init__.py; it is repeated here because pytest imports
test modules in its own order.
"""

import torch  # noqa: F401  -- MUST precede rasterio/geopandas on Windows

import numpy as np
import pytest
from shapely.geometry import Point, box

from tseg.config import load_profile
from tseg.records import Feature


@pytest.fixture
def trees_profile():
    return load_profile("trees")


@pytest.fixture
def square_mask():
    """20x20 filled square inside a 40x40 frame; area is exactly 400 px."""
    m = np.zeros((40, 40), dtype=bool)
    m[10:30, 10:30] = True
    return m, (10.0, 10.0, 30.0, 30.0)


@pytest.fixture
def merged_crowns_mask():
    """Two blobs joined by a thin bridge - the dense-canopy failure case."""
    m = np.zeros((60, 120), dtype=bool)
    m[20:40, 10:30] = True
    m[20:40, 90:110] = True
    m[29:31, 30:90] = True
    return m, (10.0, 20.0, 110.0, 40.0)


def make_feature(i=0, score=0.5, label="tree", tile_key="1000_2000"):
    g = box(1000 + i, 2000, 1010 + i, 2010)
    return Feature(label=label, score=score, backend="test", tile_key=tile_key,
                   mask=g, bbox=g, rect=g,
                   circle=Point(1005 + i, 2005).buffer(5),
                   cx=1005 + i, cy=2005, radius_m=5.0, area_m2=78.5)


@pytest.fixture
def features():
    return [make_feature(i, score=0.1 + (i % 10) / 10.0) for i in range(50)]
