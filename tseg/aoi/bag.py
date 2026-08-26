"""BAG pand (building footprint) access and per-pand image chips.

Endpoint and attribute names verified live against
https://service.pdok.nl/lv/bag/wfs/v2_0 (typeNames=bag:pand): a feature carries
identificatie, bouwjaar, status, gebruiksdoel, oppervlakte_min/max and
aantal_verblijfsobjecten. The older geodata.nationaalgeoregister.nl host is
deprecated and is not used.

A chip is the footprint bbox padded by pad_m, fetched at the profile
resolution, with the footprint rasterised as a mask. mask_mode "soft" dims the
surroundings rather than zeroing them: a hard cut removes the eave and ridge
line, which is diagnostic for thatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rasterio.features import rasterize
from shapely.geometry import shape

from tseg.aoi.wfs import get_feature
from tseg.config import BagCfg
from tseg.imagery.raster import bounds_transform


@dataclass
class Pand:
    identificatie: str
    geom: Any
    props: dict = field(default_factory=dict)

    @property
    def area_m2(self) -> float:
        return float(self.geom.area)

    def padded_bounds(self, pad_m: float):
        xmin, ymin, xmax, ymax = self.geom.bounds
        return (xmin - pad_m, ymin - pad_m, xmax + pad_m, ymax + pad_m)


def fetch_panden(bbox, cfg: BagCfg, aoi=None, max_features: int | None = None):
    """Page through bag:pand inside an RD bbox.

    Filtering is client-side and deliberate: the PDOK BAG WFS accepts a bbox
    but we do not rely on it for attribute predicates, mirroring the
    CQL_FILTER caveat already documented for the bestuurlijke-gebieden WFS.
    """
    out: list[Pand] = []
    start = 0
    while True:
        d = get_feature(cfg.wfs, cfg.typename, bbox=bbox,
                        count=cfg.page_size, start_index=start)
        feats = d.get("features", [])
        if not feats:
            break

        for f in feats:
            props = dict(f.get("properties") or {})
            if cfg.status and props.get("status") != cfg.status:
                continue
            geom = shape(f["geometry"])
            if geom.is_empty or not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty or geom.area < cfg.min_area_m2:
                continue
            if aoi is not None and not aoi.intersects(geom):
                continue
            out.append(Pand(str(props.get("identificatie", "")), geom, props))
            if max_features and len(out) >= max_features:
                return out

        if len(feats) < cfg.page_size:
            break
        start += cfg.page_size
    return out


def pand_chip(pand: Pand, wms, cfg: BagCfg, res: float):
    """Return (chip_uint8_HxWx3, bounds, transform, footprint_mask).

    The chip keeps its native aspect ratio; resizing to a square happens in the
    classifier backend so the geometry written to disk stays truthful.
    """
    bounds = pand.padded_bounds(cfg.pad_m)
    xmin, ymin, xmax, ymax = bounds
    w = max(1, int(round((xmax - xmin) / res)))
    h = max(1, int(round((ymax - ymin) / res)))

    img = wms.fetch_bbox(xmin, ymin, xmax, ymax, npx=w, npy=h)
    transform = bounds_transform(xmin, ymin, xmax, ymax, w, h)

    mask = rasterize(
        [(pand.geom, 1)], out_shape=(h, w), transform=transform,
        fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)

    if cfg.mask_mode == "hard":
        img = img * mask[:, :, None]
    elif cfg.mask_mode == "soft":
        dim = np.clip(cfg.mask_dim, 0.0, 1.0)
        factor = np.where(mask, 1.0, dim)[:, :, None]
        img = (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    elif cfg.mask_mode != "none":
        raise ValueError(
            "bag.mask_mode must be soft | hard | none, got " + repr(cfg.mask_mode)
        )

    return img, bounds, transform, mask
