"""Chip classifier -- the riet (thatched roof) path.

Why a classifier and not a detector. BAG already gives the footprint, so the
location is known and the only open question is the roof material: that is
classification, not detection. Three consequences follow.

  * Label cost. One keypress per building instead of a traced polygon. Since
    no pretrained thatch model exists anywhere, labelling IS the project.
  * Data efficiency. A frozen backbone with a linear head converges on a few
    hundred examples per class; a detector wants thousands.
  * Imbalance. Thatch is a low single-digit percentage of Dutch building stock
    and geographically clumped. With chips you resample the batch; with a
    detector you take whatever the tile happens to contain.

Default backbone is DINOv2 ViT-S/14, frozen, with a linear head on top. Frozen
means the head trains in seconds on CPU, cannot overfit 300 examples the way a
full finetune would, and yields embeddings for free -- which the review UI uses
to mine more candidates that look like a confirmed positive.

ConvNeXt-Tiny is the upgrade path once labels pass ~1000, not a competitor:
both live behind the same timm API, so it is a config string change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tseg.models.base import BaseBackend

DEFAULT_BACKBONE = "vit_small_patch14_dinov2.lvd142m"


class ClassifierBackend(BaseBackend):
    name = "classifier"

    def __init__(self, backbone: str = DEFAULT_BACKBONE,
                 classes=None, resolution: int = 224,
                 freeze: bool = True, batch_size: int = 32):
        self.backbone_name = backbone
        self.classes = list(classes or ["overig", "riet"])
        self.resolution = resolution
        self.freeze = freeze
        self.batch_size = batch_size
        self.backbone = None
        self.head = None
        self.transform = None
        self._device = "cpu"

    # ------------------------------------------------------------------ load
    def load(self, weights=None, device=None):
        import timm
        import torch

        from tseg.device import torch_device

        self._device = device or "cpu"
        dev = torch_device(self._device)

        self.backbone = timm.create_model(
            self.backbone_name, pretrained=True, num_classes=0,
        )
        # DINOv2 uses patch 14, so not every input size is legal; take the
        # config the checkpoint was trained with and override only the size.
        cfg = timm.data.resolve_model_data_config(self.backbone)
        cfg["input_size"] = (3, self.resolution, self.resolution)
        self.transform = timm.data.create_transform(**cfg, is_training=False)

        self.backbone.eval().to(dev)
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.head = torch.nn.Linear(
            self.backbone.num_features, len(self.classes)
        ).to(dev)
        if weights:
            self.load_head(weights)

    # -------------------------------------------------------------- features
    def embed(self, chips) -> np.ndarray:
        """Backbone features for a list of uint8 HxWx3 chips."""
        import torch
        from PIL import Image

        from tseg.device import torch_device

        if self.backbone is None:
            self.load()
        dev = torch_device(self._device)

        if isinstance(chips, np.ndarray) and chips.ndim == 3:
            chips = [chips]

        feats = []
        with torch.no_grad():
            for i in range(0, len(chips), self.batch_size):
                batch = chips[i:i + self.batch_size]
                x = torch.stack([
                    self.transform(Image.fromarray(np.asarray(c).astype("uint8")))
                    for c in batch
                ]).to(dev)
                feats.append(self.backbone(x).float().cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1))

    # -------------------------------------------------------------- classify
    def classify(self, chip: np.ndarray):
        labels, probs = self.classify_batch([chip])
        return labels[0], float(probs[0].max())

    def classify_batch(self, chips):
        import torch

        from tseg.device import torch_device

        if self.head is None:
            self.load()
        feats = self.embed(chips)
        with torch.no_grad():
            x = torch.from_numpy(feats).to(torch_device(self._device))
            probs = torch.softmax(self.head(x), dim=1).cpu().numpy()
        labels = [self.classes[int(i)] for i in probs.argmax(axis=1)]
        return labels, probs

    # ------------------------------------------------------------------ fit
    def fit(self, chips, labels, epochs: int = 40, lr: float = 1e-3,
            class_weight: bool = True, val_split: float = 0.0, seed: int = 1337):
        """Train the linear head on frozen features.

        Class weighting is on by default and matters here: with thatch at a few
        percent of the sample, an unweighted head learns to answer with the
        majority class for everything and reports a flattering accuracy.
        """
        import torch

        from tseg.device import torch_device

        if self.head is None:
            self.load()
        dev = torch_device(self._device)

        y = np.array([self.classes.index(str(v)) for v in labels], dtype=np.int64)
        X = self.embed(chips)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]

        n_val = int(len(y) * val_split)
        Xv, yv = (X[:n_val], y[:n_val]) if n_val else (None, None)
        Xt, yt = X[n_val:], y[n_val:]

        weight = None
        if class_weight:
            counts = np.bincount(yt, minlength=len(self.classes)).astype(np.float64)
            counts[counts == 0] = 1.0
            w = counts.sum() / (len(self.classes) * counts)
            weight = torch.tensor(w, dtype=torch.float32, device=dev)

        Xt_t = torch.from_numpy(Xt).float().to(dev)
        yt_t = torch.from_numpy(yt).to(dev)
        opt = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-4)
        lossfn = torch.nn.CrossEntropyLoss(weight=weight)

        self.head.train()
        history = []
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossfn(self.head(Xt_t), yt_t)
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
        self.head.eval()

        metrics = {
            "final_loss": history[-1] if history else None,
            "n_train": int(len(yt)),
            "class_counts": {c: int((yt == i).sum())
                             for i, c in enumerate(self.classes)},
        }
        if n_val:
            with torch.no_grad():
                xv = torch.from_numpy(Xv).float().to(dev)
                pred = self.head(xv).argmax(1).cpu().numpy()
            metrics["val_accuracy"] = float((pred == yv).mean())
            metrics["n_val"] = int(n_val)
        return metrics

    # ----------------------------------------------------------- persistence
    def save_head(self, path):
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.head.state_dict(),
                    "classes": self.classes,
                    "backbone": self.backbone_name,
                    "resolution": self.resolution}, path)
        path.with_suffix(".json").write_text(
            json.dumps({"classes": self.classes, "backbone": self.backbone_name}),
            encoding="utf-8")

    def load_head(self, path):
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved = ckpt.get("classes", self.classes)
        if saved != self.classes:
            # Silently reordering classes would corrupt every prediction.
            raise ValueError(
                f"head was trained on classes {saved} but this profile declares "
                f"{self.classes}; align profile model.classes with the checkpoint"
            )
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()
