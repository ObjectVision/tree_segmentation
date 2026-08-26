"""Chip rendering for the review UI.

A chip is a crop of the tile around one candidate, with the model geometry
drawn on it. Tile imagery is fetched once and memoised, so a page of 24 chips
drawn from three tiles costs three WMS requests, not 24.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from tseg.imagery.grid import Tile
from tseg.imagery.raster import bounds_transform

OUTLINE = {
    "pred": (255, 210, 0),
    "accept": (60, 220, 120),
    "reject": (240, 70, 70),
}


class TileImages:
    """Memoised tile fetcher. Bounded so a long review session cannot grow
    without limit."""

    def __init__(self, profile, wms, maxsize: int = 24):
        self.profile = profile
        self.wms = wms
        self._get = lru_cache(maxsize=maxsize)(self._fetch)

    def _fetch(self, tile_key: str):
        x, y = (float(v) for v in tile_key.split("_"))
        tile = Tile(x, y, self.profile.grid.tile_m, self.profile.grid.overlap_m)
        xmin, ymin, xmax, ymax = tile.padded
        img = self.wms.fetch_bbox(xmin, ymin, xmax, ymax)
        transform = bounds_transform(xmin, ymin, xmax, ymax,
                                     img.shape[1], img.shape[0])
        return img, transform

    def get(self, tile_key: str):
        return self._get(tile_key)


def _to_px(poly, transform):
    inv = ~transform
    xs, ys = poly.exterior.coords.xy
    return [tuple(float(v) for v in (inv * (x, y))) for x, y in zip(xs, ys)]


def chip_for(feature, tile_images: TileImages, size: int = 192,
             margin: float = 1.6, state: str = "pred") -> Image.Image:
    """Crop around one feature and draw its geometry."""
    img, transform = tile_images.get(feature.tile_key)
    h, w = img.shape[:2]

    geom = feature.circle or feature.geometry
    px = _to_px(geom, transform)
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * margin / 2
    half = max(half, 12.0)

    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (size, size), (30, 30, 30))

    crop = Image.fromarray(img[y0:y1, x0:x1]).convert("RGB")
    scale = size / max(crop.width, crop.height)
    crop = crop.resize((max(1, int(crop.width * scale)),
                        max(1, int(crop.height * scale))), Image.BILINEAR)

    draw = ImageDraw.Draw(crop)
    colour = OUTLINE.get(state, OUTLINE["pred"])
    for name in ("circle", "rect"):
        g = getattr(feature, name, None)
        if g is None:
            continue
        pts = [((px_ - x0) * scale, (py_ - y0) * scale)
               for px_, py_ in _to_px(g, transform)]
        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=colour,
                      width=2 if name == "circle" else 1)
    return crop


def contact_sheet(chips, rows: int, cols: int, cell: int = 192,
                  states=None, pad: int = 6) -> Image.Image:
    """Compose chips into a numbered grid with a status border per cell."""
    states = states or ["pred"] * len(chips)
    W = cols * (cell + pad) + pad
    H = rows * (cell + pad) + pad
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)

    for i, chip in enumerate(chips):
        r, c = divmod(i, cols)
        if r >= rows:
            break
        ox = pad + c * (cell + pad)
        oy = pad + r * (cell + pad)
        box = Image.new("RGB", (cell, cell), (18, 18, 20))
        box.paste(chip, ((cell - chip.width) // 2, (cell - chip.height) // 2))
        sheet.paste(box, (ox, oy))

        colour = OUTLINE.get(states[i], OUTLINE["pred"])
        draw.rectangle([ox - 2, oy - 2, ox + cell + 1, oy + cell + 1],
                       outline=colour, width=3)
        draw.rectangle([ox, oy, ox + 22, oy + 16], fill=(0, 0, 0))
        draw.text((ox + 5, oy + 3), str(i + 1), fill=(255, 255, 255))
    return sheet


def paint_to_masks(painted: np.ndarray, min_px: int = 25):
    """Turn a brush layer from the ImageEditor into individual boolean masks.

    Painting is faster than dragging boxes for crowns, and it produces a real
    mask rather than a rectangle, which is what the segmentation head wants.
    """
    import cv2

    if painted is None:
        return []
    a = np.asarray(painted)
    if a.ndim == 3 and a.shape[2] == 4:
        binary = a[:, :, 3] > 0
    elif a.ndim == 3:
        binary = a.sum(axis=2) > 0
    else:
        binary = a > 0

    n, labels = cv2.connectedComponents(binary.astype(np.uint8))
    out = []
    for i in range(1, n):
        m = labels == i
        if m.sum() >= min_px:
            out.append(m)
    return out
