"""Pixel <-> RD (EPSG:28992) conversion via a rasterio affine."""

from __future__ import annotations

from shapely.ops import transform as shapely_transform


def px_to_rd(transform, col: float, row: float):
    x, y = transform * (col, row)
    return float(x), float(y)


def affine_polygon(poly, transform):
    """Reproject a pixel-space polygon into RD using the tile affine."""
    if poly is None or poly.is_empty:
        return None

    def _fn(xs, ys):
        pts = [transform * (x, y) for x, y in zip(xs, ys)]
        return tuple(zip(*pts))

    out = shapely_transform(_fn, poly)
    if not out.is_valid:
        out = out.buffer(0)
    return out


def rd_area(poly) -> float:
    """Area in square metres. RD is a metric projection, so this is just
    shapely area -- kept as a named helper so call sites read correctly."""
    return float(poly.area) if poly is not None else 0.0
