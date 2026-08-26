# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Streaming vector writers.

The original merge built every feature from all 6816 tiles into one Python
list and then called json.dumps once (deepforest_province.py:276-282),
producing a 758 MB file and a proportional RAM spike. Everything here writes
incrementally in bounded batches instead.

Formats:
  gpkg      GeoPackage (default). One layer per shape column, so rect and
            circle both survive -- OGR allows only one geometry per layer.
  fgb       FlatGeobuf. Same layout, faster to scan.
  geojson   Hand-streamed FeatureCollection. Kept because the downstream
            GeoDMS FSS store (output/limburg_gemeenten_trees.fss/) reads it.
"""

from __future__ import annotations

import json
from pathlib import Path

from tseg import attribution
from tseg.config import EPSG

SHAPE_COLUMNS = ("circle", "rect", "bbox", "mask")
BATCH = 5000


class _BaseWriter:
    def __init__(self, path, shapes=("circle", "rect"), layer_name="features"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        bad = set(shapes) - set(SHAPE_COLUMNS)
        if bad:
            raise ValueError(f"unknown shape column(s) {sorted(bad)}; "
                             f"choose from {SHAPE_COLUMNS}")
        self.shapes = tuple(shapes)
        self.layer_name = layer_name
        self.count = 0

    def write(self, feature):
        raise NotImplementedError

    def extend(self, features):
        for f in features:
            self.write(f)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _row(feature, geom_field):
    props = {
        "label": feature.label,
        "score": float(feature.score),
        "backend": feature.backend,
        "tile_key": feature.tile_key,
        "cx": float(feature.cx),
        "cy": float(feature.cy),
        "radius_m": float(feature.radius_m),
        "area_m2": float(feature.area_m2),
    }
    for k, v in (feature.props or {}).items():
        # Keep attribute types OGR can store.
        props[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
    props["geometry"] = getattr(feature, geom_field) or feature.geometry
    return props


class OGRWriter(_BaseWriter):
    """GeoPackage / FlatGeobuf via pyogrio, appending in batches."""

    def __init__(self, path, shapes=("circle", "rect"), driver="GPKG",
                 layer_name="features", profile=None):
        super().__init__(path, shapes, layer_name)
        self.driver = driver
        self.profile = profile
        self._buf = {s: [] for s in self.shapes}
        self._started = set()

    def write(self, feature):
        for s in self.shapes:
            if getattr(feature, s) is not None:
                self._buf[s].append(_row(feature, s))
        self.count += 1
        if self.count % BATCH == 0:
            self.flush()

    def flush(self):
        # pyogrio directly rather than GeoDataFrame.to_file: only the former
        # exposes dataset_metadata / layer_metadata, which is where the CC-BY
        # attribution has to live so it travels inside the file itself.
        import geopandas as gpd
        import pyogrio

        md = attribution.metadata(self.profile)
        for s, rows in self._buf.items():
            if not rows:
                continue
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{EPSG}")
            layer = f"{self.layer_name}_{s}"
            first = layer not in self._started
            kwargs = {}
            if first and self.driver == "GPKG":
                # GDAL rejects metadata on append, so it goes on layer creation.
                kwargs = {"dataset_metadata": md,
                          "layer_metadata": dict(md, SHAPE=s)}
            pyogrio.write_dataframe(
                gdf, self.path, layer=layer, driver=self.driver,
                append=not first, **kwargs,
            )
            self._started.add(layer)
            rows.clear()

    def close(self):
        self.flush()
        attribution.write_notice(self.path.parent)


class GeoJSONWriter(_BaseWriter):
    """Hand-streamed FeatureCollection.

    Only one geometry column can live in a GeoJSON layer, so this writes the
    first entry of ``shapes``. This is the format the GeoDMS FSS store
    consumes, so its structure is deliberately unchanged from the original:
    a FeatureCollection with a named CRS.
    """

    def __init__(self, path, shapes=("circle",), layer_name="features",
                 profile=None):
        super().__init__(path, shapes, layer_name)
        self.geom_field = self.shapes[0]
        self.profile = profile
        self._fh = self.path.open("w", encoding="utf-8")
        # Structure deliberately unchanged from the original merge, which the
        # GeoDMS FSS store reads -- attribution is added as an extra member,
        # which GeoJSON readers ignore, rather than by reshaping the document.
        self._fh.write(
            '{"type": "FeatureCollection", '
            '"crs": {"type": "name", "properties": {"name": "EPSG:%d"}}, '
            '"attribution": %s, "name": %s, "features": ['
            % (EPSG, json.dumps(attribution.SHORT), json.dumps(layer_name))
        )
        self._first = True

    def write(self, feature):
        obj = feature.as_geojson(self.geom_field)
        if not self._first:
            self._fh.write(",")
        self._fh.write(json.dumps(obj))
        self._first = False
        self.count += 1

    def close(self):
        if not self._fh.closed:
            self._fh.write("]}")
            self._fh.close()
            attribution.write_notice(self.path.parent)


def open_writer(path, fmt: str = "gpkg", shapes=("circle", "rect"),
                layer_name: str = "features", profile=None):
    fmt = fmt.lower()
    if fmt == "gpkg":
        return OGRWriter(path, shapes, driver="GPKG", layer_name=layer_name,
                         profile=profile)
    if fmt in ("fgb", "flatgeobuf"):
        return OGRWriter(path, shapes, driver="FlatGeobuf",
                         layer_name=layer_name, profile=profile)
    if fmt == "geojson":
        return GeoJSONWriter(path, shapes, layer_name=layer_name,
                             profile=profile)
    raise ValueError(f"format must be gpkg | fgb | geojson, got {fmt!r}")
