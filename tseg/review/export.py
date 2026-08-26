# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Review store -> COCO instance-segmentation dataset.

One subtlety decides whether the training data is honest. A tile is only
exportable once EVERY candidate in it has a verdict. If unreviewed detections
were left out of the annotation file, the trainer would read them as background
and be actively taught that real objects are not objects -- which is worse than
not training at all. So tiles with pending candidates are skipped and reported,
not silently half-exported.

Layout is the standard COCO split RF-DETR autodetects:

    <out>/train/_annotations.coco.json  + images
    <out>/valid/_annotations.coco.json  + images
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import box as shp_box

from tseg import attribution, runmeta
from tseg.config import EPSG
from tseg.imagery.grid import Tile
from tseg.imagery.raster import bounds_transform
from tseg.imagery.wms import WMSClient
from tseg.review.store import PENDING, REJECT, ReviewStore


def _px_ring(poly, transform):
    """RD polygon -> flat COCO segmentation list in pixel coordinates."""
    inv = ~transform
    xs, ys = poly.exterior.coords.xy
    out = []
    for x, y in zip(xs, ys):
        c, r = inv * (x, y)
        out.extend([round(float(c), 2), round(float(r), 2)])
    return out


def export_coco(store: ReviewStore, profile, out_dir, classes=None,
                geom_field: str = "mask", require_complete_tiles: bool = True,
                wms=None, progress=True, write_images: bool = True):
    """Write train/ and valid/ COCO splits from the reviewed store.

    write_images=False emits annotations plus a regeneration manifest and no
    image files. That is the shape a public release takes: PDOK imagery is
    CC-BY so it *could* be redistributed, but a manifest is smaller, stays
    current, and keeps the imagery licence question with PDOK where it belongs.
    Rebuild with ``tseg export --regenerate``.
    """
    out_dir = Path(out_dir)
    classes = list(classes or profile.model.classes)
    wms = wms or WMSClient(profile.imagery)

    # Which tiles still have unreviewed candidates?
    pending_tiles = {
        r["tile_key"] for r in store.db.execute(
            "SELECT DISTINCT tile_key FROM candidates WHERE verdict = ?",
            (PENDING,))
    }

    by_split: dict[str, dict[str, list]] = {"train": defaultdict(list),
                                            "valid": defaultdict(list)}
    for row in store.labelled():
        split = "valid" if row["is_holdout"] else "train"
        by_split[split][row["tile_key"]].append(row)

    skipped = 0
    manifest = {}
    tile_manifest: dict[str, list] = {}
    for split, tiles in by_split.items():
        images, annotations = [], []
        img_id, ann_id = 1, 1
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        keys = sorted(k for k in tiles if k)
        if progress:
            import tqdm

            keys = tqdm.tqdm(keys, desc=f"export {split}")

        for tile_key in keys:
            if require_complete_tiles and tile_key in pending_tiles:
                skipped += 1
                continue

            rows = tiles[tile_key]
            x, y = (float(v) for v in tile_key.split("_"))
            tile = Tile(x, y, profile.grid.tile_m, profile.grid.overlap_m)
            # Export the UN-padded core. The overlap ring belongs to the
            # neighbouring tiles and was reviewed there, so including it here
            # would present reviewed objects as unlabelled background.
            xmin, ymin, xmax, ymax = tile.core

            # Pixel size is a property of the tile and the resolution, so it
            # is known without fetching anything.
            w = int(round((xmax - xmin) / profile.imagery.res))
            h = int(round((ymax - ymin) / profile.imagery.res))
            transform = bounds_transform(xmin, ymin, xmax, ymax, w, h)
            core_box = shp_box(xmin, ymin, xmax, ymax)

            fname = f"{tile_key}.jpg"
            if write_images:
                img = wms.fetch_bbox(xmin, ymin, xmax, ymax)
                h, w = img.shape[:2]
                transform = bounds_transform(xmin, ymin, xmax, ymax, w, h)
                Image.fromarray(img).save(split_dir / fname, quality=92)
            images.append({"id": img_id, "file_name": fname,
                           "width": int(w), "height": int(h)})
            tile_manifest.setdefault(split, []).append(tile_key)

            for row in rows:
                # Rejects contribute the image but no annotation: that is
                # exactly what a hard negative is.
                if row["verdict"] == REJECT:
                    continue
                label = store.training_label(row)
                if label is None or label not in classes:
                    continue

                feat = store.to_feature(row)
                geom = getattr(feat, geom_field, None) or feat.geometry
                if geom is None or geom.is_empty:
                    continue

                # An object whose centroid owns this tile can still spill over
                # the core edge; clip so the polygon stays inside the image.
                geom = geom.intersection(core_box)
                if geom.is_empty:
                    continue
                if geom.geom_type == "MultiPolygon":
                    geom = max(geom.geoms, key=lambda g: g.area)
                if geom.geom_type != "Polygon":
                    continue

                seg = _px_ring(geom, transform)
                px = np.array(seg).reshape(-1, 2)
                x0, y0 = px.min(axis=0)
                x1, y1 = px.max(axis=0)

                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": classes.index(label) + 1,
                    "segmentation": [seg],
                    "bbox": [float(x0), float(y0),
                             float(x1 - x0), float(y1 - y0)],
                    "area": float(geom.area / (profile.imagery.res ** 2)),
                    "iscrowd": 0,
                })
                ann_id += 1
            img_id += 1

        coco = {
            "info": {"description": f"tseg {profile.name} {split}",
                     "attribution": attribution.SHORT,
                     "license_url": attribution.LICENSE_URL},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i + 1, "name": c, "supercategory": "none"}
                           for i, c in enumerate(classes)],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8")
        manifest[split] = {"images": len(images), "annotations": len(annotations)}

    attribution.write_notice(out_dir)
    _write_regenerate(out_dir, profile, tile_manifest, write_images)

    if skipped:
        print(f"skipped {skipped} tile(s) with unreviewed candidates -- finish "
              f"reviewing them or pass --allow-incomplete to include them "
              f"(they will train the model that unreviewed objects are background)")
    if not write_images:
        print("no images written; rebuild them with 'tseg export --regenerate'")
    print(json.dumps(manifest, indent=2))
    return manifest


def _write_regenerate(out_dir, profile, tile_manifest, images_present):
    """Everything needed to rebuild the image files from PDOK."""
    (Path(out_dir) / "regenerate.json").write_text(json.dumps({
        "note": "Rebuild the image files with: tseg export --regenerate "
                "--out <dir>. Imagery is not redistributed here; it is fetched "
                "from PDOK under CC-BY-4.0.",
        "images_present": bool(images_present),
        "attribution": attribution.SHORT,
        "wms": profile.imagery.wms,
        "layer": profile.imagery.layer,
        "res_m": profile.imagery.res,
        "format": profile.imagery.fmt,
        "tile_m": profile.grid.tile_m,
        "overlap_m": profile.grid.overlap_m,
        "crs": f"EPSG:{EPSG}",
        "tiles": tile_manifest,
    }, indent=2), encoding="utf-8")


def regenerate_images(out_dir, profile=None, progress=True):
    """Rebuild the image files a --no-images export left out.

    Reads regenerate.json, refetches each tile core from PDOK at the recorded
    layer and resolution, and writes it under the split it belongs to.
    """
    out_dir = Path(out_dir)
    spec = json.loads((out_dir / "regenerate.json").read_text(encoding="utf-8"))

    if profile is None:
        profile = runmeta.load(out_dir.parent.parent.parent, fallback=None)
    profile.imagery.layer = spec["layer"]
    profile.imagery.res = spec["res_m"]
    profile.grid.tile_m = spec["tile_m"]
    profile.grid.overlap_m = spec["overlap_m"]
    wms = WMSClient(profile.imagery)

    n = 0
    for split, keys in spec["tiles"].items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        it = keys
        if progress:
            import tqdm

            it = tqdm.tqdm(keys, desc=f"regen {split}")
        for tile_key in it:
            dst = split_dir / f"{tile_key}.jpg"
            if dst.exists():
                continue
            x, y = (float(v) for v in tile_key.split("_"))
            tile = Tile(x, y, profile.grid.tile_m, profile.grid.overlap_m)
            xmin, ymin, xmax, ymax = tile.core
            img = wms.fetch_bbox(xmin, ymin, xmax, ymax)
            Image.fromarray(img).save(dst, quality=92)
            n += 1
    print(f"regenerated {n} image(s) in {out_dir}")
    return n


def export_chips(store: ReviewStore, out_dir, classes=None):
    """Classifier split: a folder per class, symlink-free copies of the chips.

    Used by the riet path, where the training unit is a whole chip rather than
    a polygon inside a tile.
    """
    import shutil

    out_dir = Path(out_dir)
    counts: dict[str, int] = {}
    dropped = 0
    for split in ("train", "valid"):
        rows = store.labelled(holdout=(split == "valid"), source="pand")
        for row in rows:
            label = store.training_label(row)
            if label is None:
                # A bare reject carries no class. In the review UI a corrected
                # chip is stored as a relabel precisely so this branch stays
                # rare; if it is not rare, corrections are being lost.
                dropped += 1
                continue
            if classes and label not in classes:
                continue
            src = row["chip_path"]
            if not src or not Path(src).exists():
                continue
            dst_dir = out_dir / split / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / Path(src).name)
            key = f"{split}/{label}"
            counts[key] = counts.get(key, 0) + 1
    if dropped:
        print(f"WARNING: {dropped} reviewed chip(s) had no class and were "
              f"dropped. A classifier learns nothing from a bare reject -- in "
              f"the review UI, correct a chip by unticking it and naming its "
              f"real class rather than only rejecting it.")

    train_classes = {k.split("/", 1)[1] for k in counts if k.startswith("train/")}
    if len(train_classes) < 2:
        raise SystemExit(
            f"training chips cover only {sorted(train_classes) or 'no'} class(es). "
            f"A classifier needs at least two: review more panden, and make sure "
            f"negatives are being saved with their class, not rejected."
        )

    print(json.dumps(counts, indent=2))
    return counts
