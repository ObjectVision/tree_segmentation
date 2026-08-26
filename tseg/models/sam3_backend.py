"""SAM 3 backend -- zero-shot label bootstrapping only.

Purpose is narrow: give round 0 of the review loop something to triage instead
of a blank canvas. SAM 3 is never trained here. Its masks become annotations in
the review store, you accept or reject them, and RF-DETR trains on the result.

Licensing note: SAM 3 ships under Metas custom SAM License, not an OSI
licence. It permits commercial use with restrictions. It is used here at
inference time to pre-label your own imagery, and the resulting annotations are
your data -- but confirm that reading before this reaches production.

Prefers samgeo (SamGeo3) when installed, since it already handles geospatial
rasters; falls back to HuggingFace transformers otherwise.
"""

from __future__ import annotations

import numpy as np

from tseg.models.base import BaseBackend
from tseg.records import Detection

MIN_MASK_PX = 16


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
