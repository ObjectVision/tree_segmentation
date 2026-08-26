# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Minimal WFS 2.0 GetFeature helper shared by the AOI sources."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from tseg.config import EPSG


def get_feature(wfs: str, typename: str, *, bbox=None, count=None,
                start_index=None, timeout: int = 120, retries: int = 3,
                extra: dict | None = None) -> dict:
    """GetFeature as GeoJSON. Returns the decoded FeatureCollection."""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": typename,
        "outputFormat": "application/json",
        "srsName": f"urn:ogc:def:crs:EPSG::{EPSG}",
    }
    if bbox is not None:
        params["bbox"] = ",".join(str(v) for v in (*bbox, f"EPSG:{EPSG}"))
    if count is not None:
        params["count"] = count
    if start_index is not None:
        params["startIndex"] = start_index
    if extra:
        params.update(extra)

    url = f"{wfs}?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"WFS GetFeature failed for {typename}: {last}") from last
