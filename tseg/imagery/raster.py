# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""GeoTIFF write + px<->RD helpers. Ported from deepforest_province.py:145-151."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from tseg.config import CRS


def bounds_transform(xmin, ymin, xmax, ymax, width, height):
    """Affine mapping pixel (col, row) -> RD (x, y)."""
    return from_bounds(xmin, ymin, xmax, ymax, width, height)


def write_geotiff(arr: np.ndarray, path, xmin, ymin, xmax, ymax, crs: str = CRS):
    t = from_bounds(xmin, ymin, xmax, ymax, arr.shape[1], arr.shape[0])
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=arr.shape[2] if arr.ndim == 3 else 1,
        dtype="uint8", crs=crs, transform=t,
    ) as dst:
        if arr.ndim == 3:
            for b in range(arr.shape[2]):
                dst.write(arr[:, :, b], b + 1)
        else:
            dst.write(arr, 1)
    return t
