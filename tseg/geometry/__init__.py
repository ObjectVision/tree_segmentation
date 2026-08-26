"""Shape derivation (rect + circle), px<->RD georeferencing, cross-tile NMS."""

from tseg.geometry.dedupe import nms
from tseg.geometry.georef import affine_polygon, px_to_rd
from tseg.geometry.shapes import (
    calc_circle_bbox,
    calc_rectangle_bbox,
    derive_shapes,
    mask_to_polygon,
)

__all__ = [
    "nms", "affine_polygon", "px_to_rd",
    "derive_shapes", "mask_to_polygon",
    "calc_circle_bbox", "calc_rectangle_bbox",
]
