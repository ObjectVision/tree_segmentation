"""Turn a detection into a rectangle *and* a circle.

Masks are a superset of both, so we always derive both and write them as
separate geometry columns -- the consumer picks. Backends that only produce
boxes (DeepForest) pass mask=None and everything falls back to the bbox.

calc_rectangle_bbox / calc_circle_bbox are the *ingest* direction: labelme
shapes drawn by hand in the review UI -> bbox. Both are ported from
urban-tree/urbantree/deepforest/detection.py:55-111. Only these two functions
are ported; the surrounding module targets the old DeepForest API
(use_release(), config['score_thresh']) and will not run against the installed
deepforest 2.1.0.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import Polygon, box


# --------------------------------------------------------------- mask -> poly
def mask_to_polygon(mask: np.ndarray, simplify_px: float = 1.0):
    """Largest external contour of a boolean mask, as a pixel-space Polygon."""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 3:
        return None
    poly = Polygon(c.reshape(-1, 2).astype(float))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if simplify_px:
        poly = poly.simplify(simplify_px, preserve_topology=True)
    return poly if (poly.is_valid and not poly.is_empty) else None


# ------------------------------------------------------------------- circles
def equal_area_circle(area_px: float, cx: float, cy: float):
    """Circle with the same area as the mask, centred on its centroid.

    This is the default. min_enclosing badly overestimates when two crowns
    merge into a single mask -- a real failure mode in dense canopy -- because
    the enclosing circle then spans both crowns.
    """
    r = math.sqrt(max(area_px, 0.0) / math.pi)
    return cx, cy, r


def min_enclosing_circle(poly_or_mask):
    if isinstance(poly_or_mask, np.ndarray):
        pts = cv2.findNonZero(poly_or_mask.astype(np.uint8))
        if pts is None:
            return None
    else:
        pts = np.array(poly_or_mask.exterior.coords, dtype=np.float32).reshape(-1, 1, 2)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return float(cx), float(cy), float(r)


def circle_polygon(cx: float, cy: float, r: float, segments: int = 32):
    ang = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    return Polygon(np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)]))


# ------------------------------------------------------------------- derive
def derive_shapes(mask, bbox, circle_method: str = "equal_area") -> dict:
    """Return pixel-space {mask, bbox, rect, circle, circle_params, area_px}.

    rect is the rotated cv2.minAreaRect; for box-only backends it degenerates
    to the axis-aligned bbox, which is correct rather than merely convenient.
    """
    xmin, ymin, xmax, ymax = bbox
    bbox_poly = box(xmin, ymin, xmax, ymax)

    mask_poly = mask_to_polygon(mask) if mask is not None else None

    if mask_poly is not None:
        pts = np.array(mask_poly.exterior.coords, dtype=np.float32)
        rect_poly = Polygon(cv2.boxPoints(cv2.minAreaRect(pts)))
        area = float(mask.sum()) if mask is not None else mask_poly.area
        cx, cy = mask_poly.centroid.x, mask_poly.centroid.y
    else:
        rect_poly = bbox_poly
        area = bbox_poly.area
        cx, cy = bbox_poly.centroid.x, bbox_poly.centroid.y

    if circle_method == "min_enclosing":
        src = mask_poly if mask_poly is not None else bbox_poly
        got = min_enclosing_circle(src)
        cx, cy, r = got if got else equal_area_circle(area, cx, cy)
    elif circle_method == "equal_area":
        cx, cy, r = equal_area_circle(area, cx, cy)
    else:
        raise ValueError(
            "circle_method must be equal_area or min_enclosing, got "
            + repr(circle_method)
        )

    return {
        "mask": mask_poly,
        "bbox": bbox_poly,
        "rect": rect_poly,
        "circle": circle_polygon(cx, cy, r),
        "circle_params": (cx, cy, r),
        "area_px": area,
    }


def passes_size_filter(bbox, min_size: float, min_ratio: float) -> bool:
    """Drop tiny detections and slivers. Mirrors urban-tree's min_bbox_size /
    min_bbox_ratio gates."""
    xmin, ymin, xmax, ymax = bbox
    w, h = abs(xmax - xmin), abs(ymax - ymin)
    if min_size and (w < min_size or h < min_size):
        return False
    long_side = max(w, h)
    if min_ratio and long_side > 0 and (min(w, h) / long_side) < min_ratio:
        return False
    return True


# ------------------------------------------------------- labelme ingest path
def distance(points) -> float:
    """Distance between two points, [[x1, y1], [x2, y2]]."""
    p1, p2 = points
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def calc_rectangle_bbox(points, img_h: int, img_w: int) -> dict:
    """bbox from a labelme rectangle (two corner points), clipped to the image."""
    lt, rb = points
    xmin, ymin = lt
    xmax, ymax = rb
    xmin = min(max(0, xmin), img_w)
    xmax = min(max(0, xmax), img_w)
    ymin = min(max(0, ymin), img_h)
    ymax = min(max(0, ymax), img_h)
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def calc_circle_bbox(points, img_h: int, img_w: int) -> dict:
    """bbox from a labelme circle (centre point + a point on the rim)."""
    center = points[0]
    dist = distance(points)
    xmin = center[0] - dist
    xmax = center[0] + dist
    ymin = center[1] - dist
    ymax = center[1] + dist
    xmin = min(max(0, xmin), img_w)
    xmax = min(max(0, xmax), img_w)
    ymin = min(max(0, ymin), img_h)
    ymax = min(max(0, ymax), img_h)
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
