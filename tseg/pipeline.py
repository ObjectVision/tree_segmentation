"""The two run modes, sharing everything except what they iterate over.

  run_tiles   grid an AOI, detect per tile      -> Features
  run_panden  crop+mask each BAG pand, classify -> Features

Both write into the same resumable cache and the same review store, so the
feedback loop does not care which mode produced a candidate.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tqdm
from shapely.geometry import Point

from tseg import device as devmod
from tseg import runmeta
from tseg.aoi.bag import fetch_panden, pand_chip
from tseg.geometry.dedupe import owns
from tseg.geometry.georef import affine_polygon
from tseg.geometry.shapes import derive_shapes, passes_size_filter
from tseg.imagery.cache import TileCache
from tseg.imagery.grid import Tile, make_grid
from tseg.imagery.raster import bounds_transform
from tseg.imagery.wms import WMSClient
from tseg.models import get_backend
from tseg.records import Feature

# Per-process state, set by _init_worker.
_STATE: dict = {}


# --------------------------------------------------------------- detections
def normalise_label(label: str, classes) -> str:
    """Match a backend label against the profile classes, case-insensitively.

    DeepForest returns "Tree" while the trees profile declares "tree". Without
    this the labels look right in the GeoPackage and then silently drop out of
    the COCO export, which would train on an empty annotation set.
    """
    if not classes:
        return label
    lowered = {c.lower(): c for c in classes}
    return lowered.get(str(label).lower(), label)


def detections_to_features(dets, transform, tile: Tile, profile, aoi=None):
    """Pixel-space Detections -> RD Features, with rect and circle derived."""
    out = []
    for det in dets:
        if not passes_size_filter(det.bbox, profile.shapes.min_bbox_size,
                                  profile.shapes.min_bbox_ratio):
            continue

        px = derive_shapes(det.mask, det.bbox, profile.shapes.circle_method)
        geoms = {k: affine_polygon(px[k], transform)
                 for k in ("mask", "bbox", "rect", "circle")}

        primary = geoms["mask"] or geoms["bbox"]
        if primary is None or primary.is_empty:
            continue
        centroid = primary.centroid

        # A detection belongs to the tile whose UN-padded core holds its
        # centroid. With overlapping tiles this is what stops the same crown
        # being written by both neighbours.
        if not owns(centroid, tile.core):
            continue
        if aoi is not None and not aoi.contains(Point(centroid.x, centroid.y)):
            continue

        cx_px, cy_px, r_px = px["circle_params"]
        out.append(Feature(
            label=normalise_label(det.label, profile.model.classes),
            score=float(det.score),
            backend=profile.model.backend,
            tile_key=tile.key,
            mask=geoms["mask"], bbox=geoms["bbox"],
            rect=geoms["rect"], circle=geoms["circle"],
            cx=float(centroid.x), cy=float(centroid.y),
            radius_m=float(r_px * profile.imagery.res),
            area_m2=float(px["area_px"] * profile.imagery.res ** 2),
        ))
    return out


# -------------------------------------------------------------- tile worker
def _init_worker(profile, aoi, n_threads: int, device_name: str, out_root):
    """Process-pool initializer. out_root travels explicitly: a spawned worker
    starts with an empty _STATE, so anything it needs must arrive via
    initargs."""
    import torch

    torch.set_num_threads(max(1, n_threads))
    _STATE["profile"] = profile
    _STATE["aoi"] = aoi
    _STATE["device"] = device_name
    _STATE["wms"] = WMSClient(profile.imagery)
    _STATE["cache"] = TileCache(out_root)
    _STATE["backend"] = None


def _backend(profile, device_name):
    """One backend per process, built lazily -- models are 1-2 GB of RAM each."""
    if _STATE.get("backend") is None:
        m = profile.model
        kwargs = {}
        if m.backend == "deepforest":
            kwargs = dict(patch=m.patch, patch_overlap=m.patch_overlap,
                          iou_threshold=m.nms_iou, score_thresh=m.score_thresh)
        elif m.backend == "rfdetr":
            kwargs = dict(resolution=m.resolution, score_thresh=m.score_thresh,
                          nms_iou=m.nms_iou, classes=m.classes)
        elif m.backend == "sam3":
            kwargs = dict(prompt=m.prompt or (m.classes[0] if m.classes else "object"),
                          score_thresh=m.score_thresh)
        elif m.backend == "classifier":
            kwargs = dict(classes=m.classes, resolution=m.resolution)

        be = get_backend(m.backend, **kwargs)
        be.load(weights=m.weights, device=device_name)
        _STATE["backend"] = be
    return _STATE["backend"]


def _process_tile(tile: Tile):
    profile = _STATE["profile"]
    cache = _STATE["cache"]
    if cache.has(tile.key):
        return tile.key, -1

    xmin, ymin, xmax, ymax = tile.padded
    img = _STATE["wms"].fetch_bbox(xmin, ymin, xmax, ymax)

    feats = []
    if img.max() > 0:  # fully-outside-coverage tiles come back black
        transform = bounds_transform(xmin, ymin, xmax, ymax,
                                     img.shape[1], img.shape[0])
        dets = _backend(profile, _STATE["device"]).detect(img)
        feats = detections_to_features(dets, transform, tile, profile,
                                       _STATE.get("aoi"))
    cache.write(tile.key, feats)
    return tile.key, len(feats)


# ------------------------------------------------------------------- drivers
def run_tiles(profile, aoi, out_root, max_tiles: int = 0, workers: int = 0,
              device_name: str | None = None, tiles=None, progress=True):
    """Detect over an AOI. Resumable: cached tiles are never reprocessed."""
    device_name = device_name or devmod.resolve()
    runmeta.save(out_root, profile)
    cache = TileCache(out_root)

    if tiles is None:
        tiles = make_grid(aoi, profile.grid.tile_m, profile.grid.overlap_m)

    uncached = [t for t in tiles if not cache.has(t.key)]
    pending = uncached[:max_tiles] if max_tiles > 0 else uncached

    # One model per process only makes sense on CPU. On a GPU the processes
    # would contend for the same device and each hold its own copy of the
    # weights, so we stay single-process there.
    if device_name != "cpu":
        workers = 1
    elif workers <= 0:
        workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    workers = max(1, min(workers, len(pending) or 1))
    threads = max(1, (os.cpu_count() or 2) // workers)

    print(f"{len(tiles) - len(uncached)} already cached, {len(uncached)} remaining; "
          f"processing {len(pending)} this run on {workers} worker(s) "
          f"({threads} torch threads each), device={device_name}")

    if not pending:
        return cache

    if workers == 1:
        _init_worker(profile, aoi, threads, device_name, out_root)
        it = tqdm.tqdm(pending, desc="tiles", disable=not progress)
        for tile in it:
            key, n = _process_tile(tile)
            if progress and n >= 0:
                tqdm.tqdm.write(f"  {key}: {n}")
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(profile, aoi, threads, device_name, out_root),
        ) as ex:
            futs = {ex.submit(_process_tile, t): t for t in pending}
            for fut in tqdm.tqdm(as_completed(futs), total=len(futs),
                                 desc="tiles", disable=not progress):
                key, n = fut.result()
                if progress and n >= 0:
                    tqdm.tqdm.write(f"  {key}: {n}")
    return cache


def run_panden(profile, aoi, out_root, max_panden: int = 0,
               device_name: str | None = None, progress=True,
               write_chips: bool = True):
    """Classify every BAG pand in the AOI.

    Geometry comes from the footprint, so no detection happens here: the
    rectangle is the rotated minimum-area rect of the pand and the circle is
    its equal-area equivalent. Chips are written to disk because the review UI
    and the classifier head both need them again later.
    """
    from pathlib import Path

    from PIL import Image

    device_name = device_name or devmod.resolve()
    runmeta.save(out_root, profile)
    cache = TileCache(out_root)
    chips_dir = Path(out_root) / "chips"
    if write_chips:
        chips_dir.mkdir(parents=True, exist_ok=True)

    _STATE.clear()
    _STATE["device"] = device_name
    wms = WMSClient(profile.imagery)
    backend = _backend(profile, device_name)

    grid = make_grid(aoi, profile.grid.tile_m, 0.0)
    print(f"pand mode: {len(grid)} grid cell(s), layer={profile.imagery.layer} "
          f"({profile.imagery.res} m/px), device={device_name}")

    total = 0
    for tile in tqdm.tqdm(grid, desc="cells", disable=not progress):
        if cache.has(tile.key):
            continue

        panden = fetch_panden(tile.core, profile.bag, aoi=aoi)
        feats = []
        for pand in panden:
            # A pand is owned by the cell containing its centroid, so a
            # building straddling a cell edge is processed exactly once.
            if not owns(pand.geom.centroid, tile.core):
                continue

            img, bounds, transform, mask = pand_chip(pand, wms, profile.bag,
                                                     profile.imagery.res)
            label, prob = backend.classify(img)

            if write_chips:
                path = chips_dir / f"{pand.identificatie}.jpg"
                Image.fromarray(img).save(path, quality=92)

            rect = pand.geom.minimum_rotated_rectangle
            area = float(pand.geom.area)
            centroid = pand.geom.centroid
            radius = (area / np.pi) ** 0.5

            feats.append(Feature(
                label=label, score=float(prob), backend=profile.model.backend,
                tile_key=tile.key,
                mask=pand.geom, bbox=pand.geom.envelope, rect=rect,
                circle=centroid.buffer(radius, quad_segs=8),
                cx=float(centroid.x), cy=float(centroid.y),
                radius_m=float(radius), area_m2=area,
                props={
                    "identificatie": pand.identificatie,
                    "bouwjaar": pand.props.get("bouwjaar"),
                    "status": pand.props.get("status"),
                    "gebruiksdoel": pand.props.get("gebruiksdoel"),
                    "chip": str(chips_dir / f"{pand.identificatie}.jpg")
                            if write_chips else None,
                },
            ))
            total += 1
            if max_panden and total >= max_panden:
                break

        cache.write(tile.key, feats)
        if max_panden and total >= max_panden:
            break

    print(f"classified {total} panden")
    return cache


def merge(cache, out_path, fmt: str = "gpkg", shapes=("circle", "rect"),
          dedupe_iou: float = 0.0, layer_name: str = "features"):
    """Stream every cached feature into one vector file.

    Never builds the full list in memory unless dedupe is on -- the original
    merge accumulated 6816 tiles worth of features and produced a 758 MB file
    in one json.dumps.
    """
    from tseg.geometry.dedupe import nms
    from tseg.io.writer import open_writer

    with open_writer(out_path, fmt, shapes, layer_name) as w:
        if dedupe_iou > 0:
            feats = nms(list(cache.iter_all()), dedupe_iou)
            w.extend(feats)
        else:
            for f in cache.iter_all():
                w.write(f)
        n = w.count
    print(f"wrote {n} features -> {out_path}")
    return n
