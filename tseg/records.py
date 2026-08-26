# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""The one record type that flows through the whole pipeline.

Backend -> Detection (pixel space, tile-local)
        -> Feature   (RD space, georeferenced, rect + circle derived)
        -> cache JSON / GeoPackage row / review-store row / COCO annotation

Masks are stored as *polygons*, never as rasters: a raster mask per detection
would be gigabytes across the Limburg run, and COCO wants polygons anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from shapely import wkt as shapely_wkt
from shapely.geometry import mapping


@dataclass
class Detection:
    """What a backend returns. Pixel coordinates, local to the tile or chip."""

    bbox: tuple[float, float, float, float]
    score: float
    label: str = "object"
    mask: Any = None          # HxW bool ndarray, or None for box-only backends


@dataclass
class Feature:
    """A georeferenced detection in EPSG:28992, with every shape derived."""

    label: str
    score: float
    backend: str
    tile_key: str = ""
    # RD geometries. mask may be None (box-only backends).
    mask: Any = None
    bbox: Any = None
    rect: Any = None
    circle: Any = None
    cx: float = 0.0
    cy: float = 0.0
    radius_m: float = 0.0
    area_m2: float = 0.0
    props: dict = field(default_factory=dict)

    @property
    def geometry(self):
        """Geometry used for NMS and ownership: the mask when we have one,
        else the box."""
        return self.mask if self.mask is not None else self.bbox

    def to_json(self) -> dict:
        d = {
            "label": self.label,
            "score": float(self.score),
            "backend": self.backend,
            "tile_key": self.tile_key,
            "cx": self.cx,
            "cy": self.cy,
            "radius_m": self.radius_m,
            "area_m2": self.area_m2,
            "props": self.props,
        }
        for name in ("mask", "bbox", "rect", "circle"):
            g = getattr(self, name)
            d[name] = g.wkt if g is not None else None
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Feature":
        kw = {k: d[k] for k in
              ("label", "score", "backend", "tile_key", "cx", "cy",
               "radius_m", "area_m2", "props") if k in d}
        for name in ("mask", "bbox", "rect", "circle"):
            w = d.get(name)
            kw[name] = shapely_wkt.loads(w) if w else None
        return cls(**kw)

    def as_geojson(self, geom_field: str = "circle") -> dict:
        """One GeoJSON Feature using the chosen geometry column."""
        g = getattr(self, geom_field) or self.geometry
        props = {
            "label": self.label,
            "score": round(float(self.score), 4),
            "backend": self.backend,
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "radius_m": round(self.radius_m, 3),
            "area_m2": round(self.area_m2, 3),
        }
        props.update(self.props)
        return {"type": "Feature", "geometry": mapping(g), "properties": props}


def dumps(features) -> str:
    return json.dumps([f.to_json() for f in features])


def loads(text: str) -> list[Feature]:
    return [Feature.from_json(d) for d in json.loads(text)]


__all__ = ["Detection", "Feature", "dumps", "loads"]
