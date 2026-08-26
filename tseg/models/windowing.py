"""Sliding-window inference for backends with a fixed input size.

DeepForest has predict_tile built in; RF-DETR and SAM 3 do not, so a 2200 px
tile has to be walked in windows and the results stitched. Windows overlap and
the merge runs NMS, for the same reason tiles overlap: an object on a window
seam would otherwise be split into two truncated halves.
"""

from __future__ import annotations

import numpy as np

from tseg.geometry.dedupe import nms
from tseg.records import Detection


def sliding_windows(height: int, width: int, size: int, overlap: float = 0.25):
    """Yield (row0, col0, row1, col1) windows covering the image.

    The last window in each direction is pulled back flush with the edge
    rather than padded, so no detection sits in dead space.
    """
    step = max(1, int(round(size * (1.0 - overlap))))
    rows = list(range(0, max(1, height - size + 1), step))
    cols = list(range(0, max(1, width - size + 1), step))
    if not rows or rows[-1] + size < height:
        rows.append(max(0, height - size))
    if not cols or cols[-1] + size < width:
        cols.append(max(0, width - size))

    seen = set()
    for r in rows:
        for c in cols:
            r0, c0 = min(r, max(0, height - size)), min(c, max(0, width - size))
            key = (r0, c0)
            if key in seen:
                continue
            seen.add(key)
            yield r0, c0, min(r0 + size, height), min(c0 + size, width)


def offset_detection(det: Detection, row0: int, col0: int,
                     full_shape: tuple[int, int]) -> Detection:
    """Move a window-local detection into full-image coordinates."""
    x0, y0, x1, y1 = det.bbox
    bbox = (x0 + col0, y0 + row0, x1 + col0, y1 + row0)

    mask = None
    if det.mask is not None:
        mask = np.zeros(full_shape, dtype=bool)
        h, w = det.mask.shape
        mask[row0:row0 + h, col0:col0 + w] = det.mask

    return Detection(bbox=bbox, score=det.score, label=det.label, mask=mask)


def merge(detections: list[Detection], iou_thresh: float = 0.4) -> list[Detection]:
    """NMS across window seams, on boxes (cheap and sufficient here)."""
    from shapely.geometry import box as shp_box

    if not detections:
        return []
    boxes = [shp_box(*d.bbox) for d in detections]
    index = {id(d): b for d, b in zip(detections, boxes)}
    return nms(detections, iou_thresh,
               geom=lambda d: index[id(d)], score=lambda d: d.score)


def run_windowed(img: np.ndarray, predict, size: int, overlap: float = 0.25,
                 iou_thresh: float = 0.4) -> list[Detection]:
    """Apply ``predict(window_array) -> list[Detection]`` across the image."""
    h, w = img.shape[:2]
    if h <= size and w <= size:
        return predict(img)

    out: list[Detection] = []
    for r0, c0, r1, c1 in sliding_windows(h, w, size, overlap):
        window = img[r0:r1, c0:c1]
        for det in predict(window):
            out.append(offset_detection(det, r0, c0, (h, w)))
    return merge(out, iou_thresh)
