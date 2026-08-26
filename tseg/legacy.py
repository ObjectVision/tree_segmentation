"""Read the pre-tseg tile cache.

The original deepforest_province.py wrote one GeoJSON FeatureCollection per
tile (output/<name>_deepforest/tiles/{x}_{y}.geojson) whose features were plain
axis-aligned boxes carrying a single score property. That cache represents
6816 tiles of finished CPU work over Limburg, so it is read rather than
discarded: tseg can merge it, and the shim will not recompute a tile that
already has a legacy entry.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import shape

from tseg.geometry.shapes import circle_polygon, equal_area_circle
from tseg.records import Feature


def legacy_path(cache_dir, key: str) -> Path:
    return Path(cache_dir) / f"{key}.geojson"


def has_legacy(cache_dir, key: str) -> bool:
    return legacy_path(cache_dir, key).exists()


def read_legacy_tile(path, label: str = "tree", backend: str = "deepforest"):
    """One legacy tile file -> Features, with rect and circle derived.

    The legacy boxes had no mask, so rect degenerates to the box and the circle
    is the equal-area equivalent -- the same treatment any box-only backend
    gets today.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    key = path.stem

    out = []
    for f in data.get("features", []):
        geom = shape(f["geometry"])
        if geom.is_empty:
            continue
        c = geom.centroid
        cx, cy, r = equal_area_circle(geom.area, c.x, c.y)
        props = dict(f.get("properties") or {})
        score = float(props.pop("score", 1.0))
        out.append(Feature(
            label=label, score=score, backend=backend, tile_key=key,
            mask=None, bbox=geom.envelope, rect=geom.envelope,
            circle=circle_polygon(cx, cy, r),
            cx=float(cx), cy=float(cy), radius_m=float(r),
            area_m2=float(geom.area), props=props,
        ))
    return out


def iter_legacy_cache(cache_dir, label: str = "tree", backend: str = "deepforest"):
    """Stream every legacy tile. Never builds the whole list in memory."""
    for p in sorted(Path(cache_dir).glob("*.geojson")):
        yield from read_legacy_tile(p, label, backend)


def legacy_keys(cache_dir):
    return sorted(p.stem for p in Path(cache_dir).glob("*.geojson"))
