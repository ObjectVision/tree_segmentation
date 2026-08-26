# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""RF-DETR-Seg backend -- the finetune target.

Apache-2.0, DINOv2 backbone, instance-mask head. This is the model the review
loop actually trains: detection and segmentation share one training API, and
the seg checkpoints (Nano through 2XL) are all permissively licensed, which is
why this and not Ultralytics YOLO26 (AGPL-3.0).

Size guidance for the RX 9060 XT: 8 GB -> Small at 640 px with gradient
accumulation; 16 GB -> Medium.
"""

from __future__ import annotations

import numpy as np

from tseg.models.base import BaseBackend, require
from tseg.models.windowing import run_windowed
from tseg.records import Detection

# Detection-only variants are listed too: a box-only checkpoint is still useful
# as a first round when no mask labels exist yet.
SEG_MODELS = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
    "2xlarge": "RFDETRSeg2XLarge",
}
DET_MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


class RFDETRBackend(BaseBackend):
    name = "rfdetr"

    def __init__(self, size: str = "small", segmentation: bool = True,
                 resolution: int = 640, score_thresh: float = 0.3,
                 window_overlap: float = 0.25, nms_iou: float = 0.4,
                 classes=None):
        self.size = size
        self.segmentation = segmentation
        self.resolution = resolution
        self.score_thresh = score_thresh
        self.window_overlap = window_overlap
        self.nms_iou = nms_iou
        self.classes = classes
        self.model = None

    # ------------------------------------------------------------------ load
    def _class_name(self, cls_name: str):
        rfdetr = require("rfdetr", "rfdetr")

        try:
            return getattr(rfdetr, cls_name)
        except AttributeError:
            raise ImportError(
                f"{cls_name} not found in the installed rfdetr. Segmentation "
                f"checkpoints need rfdetr >= 1.4; upgrade with "
                f"'pip install -U rfdetr'."
            ) from None

    def load(self, weights=None, device=None):
        table = SEG_MODELS if self.segmentation else DET_MODELS
        try:
            cls_name = table[self.size]
        except KeyError:
            raise ValueError(
                f"size must be one of {sorted(table)}, got {self.size!r}"
            ) from None

        cls = self._class_name(cls_name)

        kwargs = {"resolution": self.resolution}
        if weights:
            # A finetuned checkpoint from a previous round.
            kwargs["pretrain_weights"] = weights
        self.model = cls(**kwargs)

        if device in ("cuda", "hip") and hasattr(self.model, "model"):
            try:
                self.model.model.to("cuda")
            except Exception:
                # rfdetr usually manages placement itself; a failure here is
                # not fatal, inference just stays where the library put it.
                pass

    # --------------------------------------------------------------- predict
    def _predict_window(self, window: np.ndarray) -> list[Detection]:
        from PIL import Image

        result = self.model.predict(
            Image.fromarray(window), threshold=self.score_thresh
        )

        xyxy = np.asarray(result.xyxy).reshape(-1, 4)
        conf = np.asarray(
            result.confidence if result.confidence is not None
            else np.ones(len(xyxy))
        )
        masks = getattr(result, "mask", None)
        names = (result.data or {}).get("class_name") if hasattr(result, "data") else None
        class_ids = getattr(result, "class_id", None)

        out = []
        for i in range(len(xyxy)):
            if names is not None and i < len(names):
                label = str(names[i])
            elif self.classes and class_ids is not None:
                label = self.classes[int(class_ids[i]) % len(self.classes)]
            else:
                label = "object"
            out.append(Detection(
                bbox=tuple(float(v) for v in xyxy[i]),
                score=float(conf[i]),
                label=label,
                mask=np.asarray(masks[i]).astype(bool) if masks is not None else None,
            ))
        return out

    def detect(self, img: np.ndarray) -> list[Detection]:
        if self.model is None:
            self.load()
        return run_windowed(img, self._predict_window, self.resolution,
                            self.window_overlap, self.nms_iou)

    # ----------------------------------------------------------------- train
    def train(self, dataset_dir, output_dir, epochs: int = 30,
              batch_size: int = 4, grad_accum: int = 4, lr: float = 1e-4,
              resume: str | None = None, **kwargs):
        """Finetune on a COCO instance-segmentation dataset.

        RF-DETR autodetects COCO vs YOLO layout, so the exporter only has to
        produce a standard train/valid split.
        """
        if self.model is None:
            self.load(weights=resume)
        return self.model.train(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum,
            lr=lr,
            **kwargs,
        )
