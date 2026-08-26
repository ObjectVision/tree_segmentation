# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""RD (EPSG:28992) tile grid over an AOI.

Ported from ``deepforest_province.py:100-115``, with one addition: tiles can
overlap.

Why overlap matters. The original 500 m tiles butt up against each other, so a
crown sitting on a boundary is detected twice -- truncated in each tile -- and
the centroid-in-tile test at ``deepforest_province.py:197-198`` keeps *both*
halves whenever both centroids land inside. Padding each tile by ~one crown
means the object is seen whole at least once; global NMS
(``tseg.geometry.dedupe``) then drops the duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import box
from shapely.prepared import prep


@dataclass(frozen=True)
class Tile:
    """A grid cell. ``(x, y)`` is the un-padded south-west corner and is the
    tile's identity -- the cache key and the dedupe ownership window."""

    x: float
    y: float
    size: float
    overlap: float = 0.0

    @property
    def key(self) -> str:
        return f"{int(self.x)}_{int(self.y)}"

    @property
    def core(self) -> tuple[float, float, float, float]:
        """Un-padded bounds; a detection belongs to the tile whose core holds
        its centroid."""
        return (self.x, self.y, self.x + self.size, self.y + self.size)

    @property
    def padded(self) -> tuple[float, float, float, float]:
        """Bounds actually fetched and run through the model."""
        o = self.overlap
        return (self.x - o, self.y - o, self.x + self.size + o, self.y + self.size + o)


def tile_bounds(tile: Tile, padded: bool = True):
    return tile.padded if padded else tile.core


def make_grid(poly, tile_m: float, overlap_m: float = 0.0) -> list[Tile]:
    """Snap-to-grid tiles covering ``poly.bounds``, keeping those that
    intersect ``poly``."""
    minx, miny, maxx, maxy = poly.bounds
    x0 = (int(minx) // tile_m) * tile_m
    y0 = (int(miny) // tile_m) * tile_m
    pgeom = prep(poly)

    tiles: list[Tile] = []
    y = y0
    while y < maxy:
        x = x0
        while x < maxx:
            if pgeom.intersects(box(x, y, x + tile_m, y + tile_m)):
                tiles.append(Tile(x, y, tile_m, overlap_m))
            x += tile_m
        y += tile_m
    return tiles
