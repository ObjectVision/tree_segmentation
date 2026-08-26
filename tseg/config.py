# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Typed config with YAML profiles (``profiles/trees.yaml``, ``profiles/riet.yaml``).

A profile is the whole run description: which imagery, which grid, which
backend, how to derive shapes, and how to train. The CLI loads one and lets
individual fields be overridden by flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# PDOK RD constants, lifted from deepforest_province.py.
CRS = "EPSG:28992"
EPSG = 28992

WMS_LUCHTFOTO = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"

# Ground sample distance per PDOK layer. Resolution is a property of the layer,
# not a global constant -- the original script hardcoded RES = 0.25.
LAYER_RES = {
    "Actueel_ortho25": 0.25,     # 25 cm leaf-on RGB, nationwide
    "Actueel_ortho25IR": 0.25,   # 25 cm false-colour infrared
    "Actueel_orthoHR": 0.08,     # 7.5-8 cm high-res RGB
}

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"


@dataclass
class ImageryCfg:
    wms: str = WMS_LUCHTFOTO
    layer: str = "Actueel_ortho25"
    res: float | None = None   # None -> look up from LAYER_RES
    maxpx: int = 2000          # PDOK GetMap size cap per request
    fmt: str = "image/jpeg"
    timeout: int = 120
    retries: int = 3

    def __post_init__(self):
        if self.res is None:
            try:
                self.res = LAYER_RES[self.layer]
            except KeyError:
                raise ValueError(
                    f"unknown layer {self.layer!r}; pass imagery.res explicitly "
                    f"or add it to LAYER_RES (known: {sorted(LAYER_RES)})"
                ) from None


@dataclass
class GridCfg:
    tile_m: int = 500
    # Tiles overlap so a crown straddling a boundary is seen whole at least
    # once; global NMS then removes the duplicate. 0 reproduces the original
    # non-overlapping behaviour for regression testing.
    overlap_m: float = 25.0


@dataclass
class ModelCfg:
    backend: str = "deepforest"          # deepforest | rfdetr | sam3 | classifier
    # weights is a TRAINED checkpoint (a previous round's output). backbone is
    # the pretrained architecture to build on. Conflating them silently feeds a
    # timm model name to torch.load, which fails far from the cause.
    weights: str | None = None
    backbone: str | None = None          # classifier only; None -> DINOv2 ViT-S/14
    classes: list[str] = field(default_factory=lambda: ["tree"])
    score_thresh: float = 0.2
    nms_iou: float = 0.15                # within-tile NMS
    dedupe_iou: float = 0.4              # cross-tile NMS on the merged layer
    resolution: int = 640                # rfdetr / classifier input size
    patch: int = 400                     # deepforest predict_tile patch (px)
    patch_overlap: float = 0.25
    prompt: str | None = None            # sam3 text prompt


@dataclass
class ShapeCfg:
    # equal_area (r = sqrt(area/pi)) or min_enclosing. equal_area is the
    # default: min_enclosing badly overestimates when two crowns merge into
    # one mask, which is a real failure mode in dense canopy.
    circle_method: str = "equal_area"
    min_bbox_size: float = 0.0    # px; drop detections smaller than this
    min_bbox_ratio: float = 0.0   # short/long side; drop slivers


@dataclass
class BagCfg:
    wfs: str = "https://service.pdok.nl/lv/bag/wfs/v2_0"
    typename: str = "bag:pand"
    page_size: int = 1000
    pad_m: float = 2.0
    min_area_m2: float = 20.0
    status: str = "Pand in gebruik"
    # soft keeps the eave/ridge line (diagnostic for thatch); hard zeroes
    # everything outside the footprint.
    mask_mode: str = "soft"
    mask_dim: float = 0.35        # soft-mask multiplier outside the footprint
    chip_px: int = 224


@dataclass
class ReviewCfg:
    rows: int = 4
    cols: int = 6
    chip_px: int = 192
    # Uncertainty-first: |score - 0.5| ascending. Reviewing confident hits
    # teaches the model almost nothing.
    sort: str = "uncertainty"     # uncertainty | score | random


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    val_fraction: float = 0.2
    # The holdout is frozen at round 0 and never touched by the review loop,
    # otherwise the metric drifts with the training data.
    holdout_seed: int = 1337


@dataclass
class Profile:
    name: str = "default"
    imagery: ImageryCfg = field(default_factory=ImageryCfg)
    grid: GridCfg = field(default_factory=GridCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    shapes: ShapeCfg = field(default_factory=ShapeCfg)
    bag: BagCfg = field(default_factory=BagCfg)
    review: ReviewCfg = field(default_factory=ReviewCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    @property
    def npx(self) -> int:
        """Tile edge in pixels, including overlap padding."""
        return int(round((self.grid.tile_m + 2 * self.grid.overlap_m)
                         / self.imagery.res))

    def to_dict(self) -> dict:
        return _asdict(self)


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    return obj


def _build(cls, data: dict | None):
    """Instantiate a config dataclass from a dict, rejecting unknown keys."""
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"expected any of {sorted(known)}"
        )
    return cls(**data)


def load_profile(name_or_path: str) -> Profile:
    """Load ``profiles/<name>.yaml`` (or an explicit path) into a Profile."""
    path = Path(name_or_path)
    if not path.exists():
        path = PROFILE_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"no profile {name_or_path!r}; available: {available}"
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Profile(
        name=raw.get("name", path.stem),
        imagery=_build(ImageryCfg, raw.get("imagery")),
        grid=_build(GridCfg, raw.get("grid")),
        model=_build(ModelCfg, raw.get("model")),
        shapes=_build(ShapeCfg, raw.get("shapes")),
        bag=_build(BagCfg, raw.get("bag")),
        review=_build(ReviewCfg, raw.get("review")),
        train=_build(TrainCfg, raw.get("train")),
    )


def apply_overrides(profile: Profile, overrides: dict[str, Any]) -> Profile:
    """Apply ``{"model.backend": "rfdetr"}`` style overrides in place."""
    for dotted, value in overrides.items():
        if value is None:
            continue
        section, _, key = dotted.partition(".")
        target = getattr(profile, section) if key else profile
        attr = key or section
        if not hasattr(target, attr):
            raise ValueError(f"no config field {dotted!r}")
        setattr(target, attr, value)
    # Re-derive resolution if the layer changed but res was not given.
    if "imagery.layer" in overrides and "imagery.res" not in overrides:
        profile.imagery.res = LAYER_RES.get(profile.imagery.layer,
                                            profile.imagery.res)
    return profile


def dump_yaml(profile: Profile) -> str:
    return yaml.safe_dump(profile.to_dict(), sort_keys=False, allow_unicode=True)
