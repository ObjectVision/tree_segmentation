# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""PDOK WMS GetMap client.

Ported from ``deepforest_province.py:120-142``. Two things changed on the way:

  * parameters go through ``urllib.parse.urlencode`` instead of a manual
    ``"&".join``. The original only survived because every value was digits and
    commas -- it breaks the moment a layer name, style or CQL filter needs
    escaping.
  * requests retry with backoff. A single 502 used to poison a whole tile.

We stay on WMS rather than WMTS deliberately: arbitrary-bbox GetMap is what
makes per-pand chips possible, and a WMTS tile grid cannot frame a building.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

from tseg.config import CRS, ImageryCfg


class WMSError(RuntimeError):
    pass


class WMSClient:
    def __init__(self, cfg: ImageryCfg):
        self.cfg = cfg

    # ------------------------------------------------------------------ raw
    def _get_block(self, x0: float, y0: float, x1: float, y1: float,
                   w: int, h: int) -> np.ndarray:
        """One GetMap request, <= cfg.maxpx on each side."""
        params = {
            "service": "WMS",
            "request": "GetMap",
            "version": "1.3.0",
            "layers": self.cfg.layer,
            "styles": "",
            "crs": CRS,
            # PDOK accepts easting-first for EPSG:28992 despite WMS 1.3.0
            # nominally implying northing-first for this CRS.
            "bbox": f"{x0},{y0},{x1},{y1}",
            "width": int(w),
            "height": int(h),
            "format": self.cfg.fmt,
        }
        url = f"{self.cfg.wms}?" + urllib.parse.urlencode(params)

        last: Exception | None = None
        for attempt in range(self.cfg.retries):
            try:
                with urllib.request.urlopen(url, timeout=self.cfg.timeout) as r:
                    ctype = r.headers.get("Content-Type", "")
                    data = r.read()
                if "xml" in ctype.lower():
                    # WMS reports errors as a ServiceException document with a
                    # 200 status, which would otherwise decode as a broken image.
                    raise WMSError(f"WMS exception: {data[:400].decode(errors='replace')}")
                return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            except (urllib.error.URLError, OSError, WMSError) as exc:
                last = exc
                if attempt < self.cfg.retries - 1:
                    time.sleep(2 ** attempt)
        raise WMSError(f"GetMap failed after {self.cfg.retries} tries: {last}") from last

    # ----------------------------------------------------------------- tiles
    def fetch_bbox(self, xmin: float, ymin: float, xmax: float, ymax: float,
                   npx: int | None = None, npy: int | None = None) -> np.ndarray:
        """Fetch an arbitrary RD bbox as a mosaic of <= maxpx sub-blocks.

        Returns HxWx3 uint8, north-up (row 0 = ymax).
        """
        res = self.cfg.res
        w = npx if npx is not None else max(1, int(round((xmax - xmin) / res)))
        h = npy if npy is not None else max(1, int(round((ymax - ymin) / res)))
        sx = (xmax - xmin) / w
        sy = (ymax - ymin) / h

        arr = np.zeros((h, w, 3), dtype="uint8")
        maxpx = self.cfg.maxpx
        rows = list(range(0, h, maxpx)) + [h]
        cols = list(range(0, w, maxpx)) + [w]
        for r0, r1 in zip(rows, rows[1:]):
            for c0, c1 in zip(cols, cols[1:]):
                bx0, bx1 = xmin + c0 * sx, xmin + c1 * sx
                by1, by0 = ymax - r0 * sy, ymax - r1 * sy
                arr[r0:r1, c0:c1] = self._get_block(bx0, by0, bx1, by1,
                                                    c1 - c0, r1 - r0)
        return arr
