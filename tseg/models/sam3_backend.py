# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""SAM 3 backend -- zero-shot label bootstrapping only.

Purpose is narrow: give round 0 of the review loop something to triage instead
of a blank canvas. SAM 3 is never trained here. Its masks become annotations in
the review store, you accept or reject them, and RF-DETR trains on the result.

LICENCE WARNING. SAM 3 ships under Meta's custom SAM License. That licence
carries field-of-use restrictions, so it is NOT an OSI-approved open source
licence, and it is not GPL-compatible. This module exists only because
LICENSE-EXCEPTIONS grants a GPL-3.0 section 7 permission to link with it.

Nothing in tseg depends on this backend. Tree bootstrapping uses DeepForest
(MIT) and rare-class mining uses DINOv2 embeddings (Apache-2.0), both fully
open source. Reach for SAM 3 only when you need open-vocabulary prompting for
a class neither of those covers, and confirm the licence terms yourself before
it reaches production.

Prefers samgeo (SamGeo3) when installed, since it already handles geospatial
rasters; falls back to HuggingFace transformers otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np

from tseg.models.base import BaseBackend
from tseg.records import Detection

MIN_MASK_PX = 16

LICENCE_WARNING = (
    "SAM 3 is licensed under Meta's custom SAM License, which has field-of-use "
    "restrictions and is NOT an OSI-approved open source licence. tseg links to "
    "it under the GPL-3.0 section 7 permission in LICENSE-EXCEPTIONS. Nothing "
    "requires this backend -- use 'deepforest' to bootstrap trees or 'tseg mine' "
    "for rare classes if you would rather stay fully open source."
)


class SAM3Backend(BaseBackend):
    name = "sam3"

    def __init__(self, prompt: str = "tree", score_thresh: float = 0.3,
                 backend: str = "auto", model_id: str = "facebook/sam3"):
        self.prompt = prompt
        self.score_thresh = score_thresh
        self.backend = backend
        self.model_id = model_id
        self.impl = None
        self._device = "cpu"

    def load(self, weights=None, device=None):
        # Warned on load, not on import: importing the module is harmless, but
        # actually pulling the weights is the moment the licence applies.
        warnings.warn(LICENCE_WARNING, stacklevel=2)
        self._device = device or "cpu"
        torch_device = "cuda" if self._device in ("cuda", "hip") else "cpu"

        chosen = self.backend
        if chosen == "auto":
            try:
                import samgeo  # noqa: F401

                chosen = "samgeo"
            except ImportError:
                chosen = "transformers"

        if chosen == "samgeo":
            from samgeo import SamGeo3

            self.impl = ("samgeo", SamGeo3(model_type="sam3", backend="meta",
                                           device=torch_device))
        else:
            import torch
            from transformers import Sam3Model, Sam3Processor

            processor = Sam3Processor.from_pretrained(weights or self.model_id)
            model = Sam3Model.from_pretrained(weights or self.model_id).to(torch_device)
            model.eval()
            self.impl = ("transformers", (processor, model, torch))

    # ------------------------------------------------------------------ util
    @staticmethod
    def _masks_to_detections(masks, scores, label) -> list[Detection]:
        out = []
        for m, s in zip(masks, scores):
            m = np.asarray(m).astype(bool)
            if m.ndim == 3:
                m = m[0]
            if m.sum() < MIN_MASK_PX:
                continue
            ys, xs = np.nonzero(m)
            out.append(Detection(
                bbox=(float(xs.min()), float(ys.min()),
                      float(xs.max()) + 1.0, float(ys.max()) + 1.0),
                score=float(s),
                label=label,
                mask=m,
            ))
        return out

    def detect(self, img: np.ndarray) -> list[Detection]:
        if self.impl is None:
            self.load()
        kind, impl = self.impl

        if kind == "samgeo":
            sam = impl
            sam.set_image(img)
            res = sam.generate_masks(prompt=self.prompt)
            masks, scores = _unpack_samgeo(res, sam)
        else:
            processor, model, torch = impl
            from PIL import Image

            inputs = processor(images=Image.fromarray(img), text=self.prompt,
                               return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            post = processor.post_process_instance_segmentation(
                outputs, threshold=self.score_thresh,
                target_sizes=[img.shape[:2]],
            )[0]
            masks = post["masks"].cpu().numpy()
            scores = post["scores"].cpu().numpy()

        dets = self._masks_to_detections(masks, scores, self.prompt)
        return [d for d in dets if d.score >= self.score_thresh]


def _unpack_samgeo(res, sam):
    """samgeo returns different shapes across versions; normalise to
    (masks, scores)."""
    if isinstance(res, dict):
        masks = res.get("masks")
        scores = res.get("scores")
    else:
        masks = res
        scores = None
    if masks is None:
        masks = getattr(sam, "masks", None)
    if masks is None:
        return [], []
    masks = np.asarray(masks)
    if scores is None:
        scores = getattr(sam, "scores", None)
    if scores is None:
        scores = np.ones(len(masks), dtype=float)
    return masks, np.asarray(scores).reshape(-1)
