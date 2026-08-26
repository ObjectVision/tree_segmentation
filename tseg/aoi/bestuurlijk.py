# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Gemeente polygons from the PDOK bestuurlijke-gebieden WFS.

Ported verbatim in behaviour from deepforest_province.py:74-97, including the
workaround: this WFS ignores CQL_FILTER, so we fetch all gemeenten and select
client-side by identificatie (the GM-prefixed CBS code), then dissolve into
one AOI.
"""

from __future__ import annotations

from shapely.geometry import shape
from shapely.ops import unary_union

from tseg.aoi.wfs import get_feature

WFS = "https://service.pdok.nl/kadaster/brk-bestuurlijke-gebieden/wfs/v1_0"
GEM_TYPE = "bestuurlijkegebieden:Gemeentegebied"

# Noord, Zuid and part of Midden Limburg (CBS gemeentecodes, GM-prefixed).
DEFAULT_CODES = (
    "GM0889,GM0893,GM0907,GM1507,GM0944,GM1894,GM0983,GM0984,GM0957,GM0888,"
    "GM1954,GM0899,GM1903,GM1729,GM0917,GM0928,GM0882,GM0935,GM0938,GM0965,"
    "GM1883,GM0971,GM0981,GM0994,GM0986"
)


def fetch_area(codes, verbose: bool = True):
    """Union of the selected gemeenten as one shapely geometry."""
    codes = {c.strip().upper() for c in
             (codes.split(",") if isinstance(codes, str) else codes) if c.strip()}

    d = get_feature(WFS, GEM_TYPE)
    picked = [f for f in d["features"]
              if str(f["properties"]["identificatie"]).upper() in codes]

    found = {f["properties"]["identificatie"].upper() for f in picked}
    missing = codes - found
    if missing and verbose:
        print(f"WARNING: {len(missing)} code(s) not found: {sorted(missing)}")
    if not picked:
        raise SystemExit("no matching gemeenten")
    if verbose:
        names = ", ".join(sorted(f["properties"]["naam"] for f in picked))
        print(f"selected {len(picked)} gemeenten: {names}")

    return unary_union([shape(f["geometry"]) for f in picked])
