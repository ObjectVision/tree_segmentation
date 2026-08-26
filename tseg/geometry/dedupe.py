"""Greedy IoU NMS, used both inside a tile and globally on the merged layer.

Cross-tile duplicates are the reason this exists. The original 500 m tiles do
not overlap, so a crown on a boundary is detected twice -- truncated in each
tile -- and the centroid-in-tile test at deepforest_province.py:197-198 keeps
BOTH truncated halves whenever both centroids land inside. With overlapping
tiles the object is seen whole at least once, and this NMS drops the rest.
"""

from __future__ import annotations

from typing import Sequence

from shapely.strtree import STRtree


def iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(items: Sequence, iou_thresh: float = 0.4, geom=None, score=None) -> list:
    """Keep the highest-scoring member of each overlapping cluster.

    STRtree-indexed so this stays usable at the scale of the full Limburg run
    (millions of features) instead of going quadratic.
    """
    if geom is None:
        def geom(f):
            return f.geometry
    if score is None:
        def score(f):
            return f.score

    feats = [f for f in items if geom(f) is not None and not geom(f).is_empty]
    if not feats:
        return []

    geoms = [geom(f) for f in feats]
    order = sorted(range(len(feats)), key=lambda i: score(feats[i]), reverse=True)
    tree = STRtree(geoms)

    suppressed = [False] * len(feats)
    kept: list = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(feats[i])
        for j in tree.query(geoms[i]):
            j = int(j)
            if j == i or suppressed[j]:
                continue
            if iou(geoms[i], geoms[j]) >= iou_thresh:
                suppressed[j] = True
    return kept


def owns(centroid, core_bounds) -> bool:
    """Tile ownership test: a detection belongs to the tile whose UN-PADDED
    core contains its centroid. Half-open, so a centroid on a shared edge is
    claimed by exactly one tile."""
    xmin, ymin, xmax, ymax = core_bounds
    return xmin <= centroid.x < xmax and ymin <= centroid.y < ymax
