# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Persist the profile a run actually used.

Tile geometry must not be re-derived from profiles/*.yaml after the fact. If
detect ran with --tile-size 200 --overlap 20 and the export later reads the
profile default of 500/25, the exported training image covers ground that was
never reviewed -- and every real object in that margin is handed to the trainer
as background. So the effective profile is written next to the cache and every
downstream command reads it back.
"""

from __future__ import annotations

from pathlib import Path

import yaml

FILENAME = "run.yaml"


def save(out_root, profile) -> Path:
    path = Path(out_root) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    from tseg.config import dump_yaml

    path.write_text(dump_yaml(profile), encoding="utf-8")
    return path


def load(out_root, fallback=None, verbose: bool = True):
    """Read back the profile a run used, falling back to the given one."""
    path = Path(out_root) / FILENAME
    if not path.exists():
        if fallback is None:
            raise FileNotFoundError(
                f"no {FILENAME} in {out_root}; run 'tseg detect' or 'tseg pand' first"
            )
        return fallback

    from tseg.config import Profile, _build
    from tseg.config import (BagCfg, GridCfg, ImageryCfg, ModelCfg, ReviewCfg,
                             ShapeCfg, TrainCfg)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    stored = Profile(
        name=raw.get("name", "run"),
        imagery=_build(ImageryCfg, raw.get("imagery")),
        grid=_build(GridCfg, raw.get("grid")),
        model=_build(ModelCfg, raw.get("model")),
        shapes=_build(ShapeCfg, raw.get("shapes")),
        bag=_build(BagCfg, raw.get("bag")),
        review=_build(ReviewCfg, raw.get("review")),
        train=_build(TrainCfg, raw.get("train")),
    )

    if fallback is not None and verbose:
        drift = []
        if stored.grid.tile_m != fallback.grid.tile_m:
            drift.append(f"tile_m {stored.grid.tile_m} != {fallback.grid.tile_m}")
        if stored.grid.overlap_m != fallback.grid.overlap_m:
            drift.append(
                f"overlap_m {stored.grid.overlap_m} != {fallback.grid.overlap_m}")
        if stored.imagery.layer != fallback.imagery.layer:
            drift.append(
                f"layer {stored.imagery.layer} != {fallback.imagery.layer}")
        if drift:
            print(f"using geometry from {path} ({'; '.join(drift)})")
    return stored


def merge_runtime(stored, current):
    """Keep the stored geometry and imagery, take runtime knobs from current.

    Grid and imagery describe data already on disk and are immutable for this
    output directory. Model, training and review settings are free to change
    between rounds -- that is the whole point of the loop.
    """
    stored.model = current.model
    stored.train = current.train
    stored.review = current.review
    stored.shapes = current.shapes
    return stored
