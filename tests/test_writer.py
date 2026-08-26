# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Streaming vector output, and the CC-BY attribution that must travel with it."""

import json

import pytest

from tseg import attribution
from tseg.io.writer import open_writer


def test_geopackage_gets_one_layer_per_shape(tmp_path, features, trees_profile):
    import pyogrio

    path = tmp_path / "out.gpkg"
    with open_writer(path, "gpkg", ("circle", "rect", "bbox"), "trees",
                     profile=trees_profile) as w:
        w.extend(features)

    layers = {name for name, _ in pyogrio.list_layers(path)}
    assert layers == {"trees_circle", "trees_rect", "trees_bbox"}

    info = pyogrio.read_info(path, layer="trees_circle")
    assert info["features"] == len(features)
    assert info["crs"].endswith("28992")


def test_geopackage_carries_cc_by_attribution(tmp_path, features, trees_profile):
    """A line in the README does not travel with a file someone was emailed."""
    import pyogrio

    path = tmp_path / "out.gpkg"
    with open_writer(path, "gpkg", ("circle",), "trees",
                     profile=trees_profile) as w:
        w.extend(features)

    md = pyogrio.read_info(path, layer="trees_circle")["layer_metadata"]
    assert md["LICENSE"] == "CC-BY-4.0"
    assert "PDOK" in md["ATTRIBUTION"]
    assert md["TSEG_LAYER"] == trees_profile.imagery.layer
    assert (tmp_path / "NOTICE.txt").exists()


def test_geojson_keeps_the_structure_geodms_reads(tmp_path, features):
    """The FSS store downstream reads this file; its shape must not change."""
    path = tmp_path / "out.geojson"
    with open_writer(path, "geojson", ("circle",), "limburg") as w:
        w.extend(features)

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["type"] == "FeatureCollection"
    assert doc["crs"]["properties"]["name"] == "EPSG:28992"
    assert doc["name"] == "limburg"
    assert len(doc["features"]) == len(features)
    assert doc["attribution"] == attribution.SHORT

    props = doc["features"][0]["properties"]
    assert {"label", "score", "radius_m", "area_m2"} <= set(props)


def test_streaming_does_not_accumulate(tmp_path, features):
    """Batches flush as they fill; the writer must not hold the full set."""
    from tseg.io import writer as w_mod

    path = tmp_path / "big.gpkg"
    with open_writer(path, "gpkg", ("circle",), "t") as w:
        for i in range(w_mod.BATCH + 10):
            w.write(features[i % len(features)])
            if i == w_mod.BATCH:
                # Past the first flush, the buffer has been emptied and refilled.
                assert len(w._buf["circle"]) < w_mod.BATCH
        assert w.count == w_mod.BATCH + 10


def test_unknown_format_and_shape_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="gpkg"):
        open_writer(tmp_path / "x.dxf", "dxf")
    with pytest.raises(ValueError, match="unknown shape"):
        open_writer(tmp_path / "x.gpkg", "gpkg", ("triangle",))
