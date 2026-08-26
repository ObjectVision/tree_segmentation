# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""DeepForest backend -- the frozen baseline.

Kept so the finetuned RF-DETR can be measured against the run that produced
output/limburg_gemeenten_deepforest/. Stock weecology/deepforest-tree weights,
no training path.

Known limitation, carried over from deepforest_pdok.py:11-13: DeepForest was
trained on ~0.10 m NEON imagery and PDOK ortho25 is 0.25 m, so crowns span
fewer pixels than in training and small or young trees are missed. That recall
gap is precisely what the review loop exists to close.

Boxes only -- detections come back with mask=None.
"""

from __future__ import annotations

import numpy as np

from tseg.models.base import BaseBackend, require
from tseg.records import Detection


class DeepForestBackend(BaseBackend):
    name = "deepforest"

    def __init__(self, patch: int = 400, patch_overlap: float = 0.25,
                 iou_threshold: float = 0.15, score_thresh: float = 0.2):
        self.patch = patch
        self.patch_overlap = patch_overlap
        self.iou_threshold = iou_threshold
        self.score_thresh = score_thresh
        self.model = None

    def load(self, weights=None, device=None):
        main = require("deepforest.main", "deepforest")

        self.model = main.deepforest()
        self.model.load_model(weights or "weecology/deepforest-tree")

        if device in ("cuda", "hip"):
            self.model.model.to("cuda")
            self.model.device = "cuda"
        self.model.model.eval()

    def detect(self, img: np.ndarray) -> list[Detection]:
        if self.model is None:
            self.load()

        boxes = self.model.predict_tile(
            image=img,
            patch_size=self.patch,
            patch_overlap=self.patch_overlap,
            iou_threshold=self.iou_threshold,
        )
        if boxes is None or not len(boxes):
            return []

        # Newer DeepForest can return a GeoDataFrame whose bounds live in a
        # geometry column rather than xmin/ymin/xmax/ymax. Handle both, as
        # deepforest_pdok.py:123-126 already had to.
        if "xmin" not in boxes.columns and "geometry" in boxes.columns:
            b = boxes.geometry.bounds
            boxes = boxes.assign(xmin=b.minx, ymin=b.miny,
                                 xmax=b.maxx, ymax=b.maxy)

        out = []
        for _, r in boxes.iterrows():
            if float(r.score) < self.score_thresh:
                continue
            out.append(Detection(
                bbox=(float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax)),
                score=float(r.score),
                label=str(getattr(r, "label", "tree")) or "tree",
                mask=None,
            ))
        return out
