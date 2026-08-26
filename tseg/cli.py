# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""tseg command line.

    tseg info
    tseg detect  --profile trees --codes GM0983 --out output/trees
    tseg pand    --profile riet  --codes GM0983 --out output/riet --limit 200
    tseg review  --profile trees --out output/trees
    tseg mine    --profile riet  --out output/riet --label riet
    tseg export  --profile trees --out output/trees [--no-images]
    tseg train   --profile trees --out output/trees
    tseg merge   --profile trees --out output/trees --format gpkg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tseg.config import LAYER_RES, apply_overrides, load_profile
from tseg.models import list_backends


def _common(ap):
    ap.add_argument("--profile", default="trees",
                    help="profile name in profiles/ or a path to a yaml file")
    ap.add_argument("--out", required=True, help="output root directory")
    ap.add_argument("--backend", choices=list_backends(),
                    help="override model.backend")
    ap.add_argument("--weights", help="override model.weights (a checkpoint)")
    ap.add_argument("--layer", choices=sorted(LAYER_RES),
                    help="override imagery.layer (resolution follows)")
    ap.add_argument("--device", choices=["cuda", "hip", "directml", "cpu"],
                    help="override device autodetection")
    return ap


def _aoi(ap):
    ap.add_argument("--codes", help="comma-separated CBS gemeentecodes (GM-prefixed)")
    ap.add_argument("--bbox", help="RD bbox xmin,ymin,xmax,ymax instead of --codes")
    return ap


def _profile_from(args, from_run: bool = False):
    """Build the effective profile.

    from_run=True means the command consumes data a previous run produced, so
    grid and imagery come from that run rather than from the yaml defaults --
    otherwise a --tile-size used at detect time would be silently forgotten.
    """
    overrides = {
        "model.backend": getattr(args, "backend", None),
        "model.weights": getattr(args, "weights", None),
        "imagery.layer": getattr(args, "layer", None),
    }
    profile = apply_overrides(load_profile(args.profile), overrides)
    if from_run:
        from tseg import runmeta

        stored = runmeta.load(args.out, fallback=profile)
        profile = runmeta.merge_runtime(stored, profile)
    return profile


def _load_aoi(args):
    from shapely.geometry import box

    from tseg.aoi.bestuurlijk import DEFAULT_CODES, fetch_area

    if getattr(args, "bbox", None):
        vals = [float(v) for v in args.bbox.split(",")]
        if len(vals) != 4:
            raise SystemExit("--bbox needs xmin,ymin,xmax,ymax in EPSG:28992")
        return box(*vals)
    return fetch_area(args.codes or DEFAULT_CODES)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="tseg", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show device, backends and profiles")

    p = _aoi(_common(sub.add_parser("detect", help="tile-level detection")))
    p.add_argument("--max-tiles", type=int, default=0, help="cap new tiles this run")
    p.add_argument("--workers", type=int, default=0, help="CPU worker processes")
    p.add_argument("--tile-size", type=int, help="override grid.tile_m")
    p.add_argument("--overlap", type=float, help="override grid.overlap_m")
    p.add_argument("--to-store", action="store_true",
                   help="also add detections to the review store")
    p.add_argument("--round", type=int, default=0, help="round number for the store")

    p = _aoi(_common(sub.add_parser("pand", help="BAG pand-level classification")))
    p.add_argument("--limit", type=int, default=0, help="cap panden this run")
    p.add_argument("--to-store", action="store_true")
    p.add_argument("--round", type=int, default=0)

    p = _common(sub.add_parser("review", help="launch the Gradio triage UI"))
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--share", action="store_true")

    p = _common(sub.add_parser("mine",
                               help="rank unreviewed chips by similarity to "
                                    "confirmed positives (rare-class hunting)"))
    p.add_argument("--label", help="mine for this class (default: any accepted)")
    p.add_argument("--like", help="comma-separated pand ids / uids to search from")
    p.add_argument("--limit", type=int, default=200, help="candidates to rank")
    p.add_argument("--no-promote", action="store_true",
                   help="rank only; do not move them up the review queue")

    p = _common(sub.add_parser("export", help="review store -> COCO / chip folders"))
    p.add_argument("--allow-incomplete", action="store_true",
                   help="export tiles that still have unreviewed candidates")
    p.add_argument("--no-images", action="store_true",
                   help="write annotations and a regeneration manifest but no "
                        "image files (the shape a public release takes)")
    p.add_argument("--regenerate", action="store_true",
                   help="rebuild the image files a --no-images export omitted")

    p = _common(sub.add_parser("train", help="finetune one round"))
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-resume", action="store_true",
                   help="start from pretrained weights instead of last round")
    p.add_argument("--allow-incomplete", action="store_true")

    p = _common(sub.add_parser("merge", help="stream the cache into one vector file"))
    p.add_argument("--format", default="gpkg", choices=["gpkg", "fgb", "geojson"])
    p.add_argument("--shapes", default="circle,rect",
                   help="geometry columns to write: circle,rect,bbox,mask")
    p.add_argument("--dedupe", type=float, default=None,
                   help="global NMS IoU (default: model.dedupe_iou)")
    p.add_argument("--name", default=None, help="output file stem")

    return ap.parse_args(argv)


def cmd_info(args):
    from tseg import __version__, device
    from tseg.config import PROFILE_DIR

    print(f"tseg {__version__}")
    print(f"device: {device.describe()}")
    print(f"backends: {', '.join(list_backends())}")
    profiles = sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
    print(f"profiles: {', '.join(profiles) or '(none)'}")
    for name in profiles:
        p = load_profile(name)
        print(f"  {name}: backend={p.model.backend} layer={p.imagery.layer} "
              f"({p.imagery.res} m/px) classes={p.model.classes}")


def _store_for(profile, out):
    from tseg.review.store import ReviewStore

    return ReviewStore(Path(out) / "review.db",
                       holdout_fraction=profile.train.val_fraction,
                       seed=profile.train.holdout_seed)


def cmd_detect(args):
    from tseg import device
    from tseg.pipeline import run_tiles

    profile = _profile_from(args)
    if args.tile_size:
        profile.grid.tile_m = args.tile_size
    if args.overlap is not None:
        profile.grid.overlap_m = args.overlap

    aoi = _load_aoi(args)
    cache = run_tiles(profile, aoi, args.out, max_tiles=args.max_tiles,
                      workers=args.workers,
                      device_name=args.device or device.resolve())

    if args.to_store:
        with _store_for(profile, args.out) as store:
            n = store.add(cache.iter_all(), round_no=args.round, source="tile")
            print(f"added {n} new candidate(s) to the review store")


def cmd_pand(args):
    from tseg import device
    from tseg.pipeline import run_panden

    profile = _profile_from(args)
    aoi = _load_aoi(args)
    cache = run_panden(profile, aoi, args.out, max_panden=args.limit,
                       device_name=args.device or device.resolve())

    if args.to_store:
        with _store_for(profile, args.out) as store:
            n = store.add(cache.iter_all(), round_no=args.round, source="pand")
            print(f"added {n} new candidate(s) to the review store")


def cmd_review(args):
    from tseg.review.app import launch

    profile = _profile_from(args, from_run=True)
    launch(profile, args.out, share=args.share, port=args.port)


def cmd_mine(args):
    from tseg import device
    from tseg.models import get_backend
    from tseg.review.mine import rank

    profile = _profile_from(args, from_run=True)
    kwargs = dict(classes=profile.model.classes,
                  resolution=profile.model.resolution)
    if profile.model.backbone:
        kwargs["backbone"] = profile.model.backbone
    backend = get_backend("classifier", **kwargs)
    backend.load(device=args.device or device.resolve())

    like = [v.strip() for v in args.like.split(",")] if args.like else None
    with _store_for(profile, args.out) as store:
        ranked = rank(store, backend, label=args.label, like=like,
                      limit=args.limit, promote=not args.no_promote)
    for row, sim in ranked[:20]:
        ident = (row["uid"] or "").split(":")[-1]
        print(f"  {sim:.3f}  {ident}")
    if len(ranked) > 20:
        print(f"  ... {len(ranked) - 20} more; run 'tseg review' to work through them")


def cmd_export(args):
    from tseg.review.export import export_chips, export_coco, regenerate_images

    profile = _profile_from(args, from_run=True)
    out = Path(args.out)
    with _store_for(profile, out) as store:
        target = out / "rounds" / str(store.max_round()) / "dataset"
        if args.regenerate:
            regenerate_images(target, profile)
        elif profile.model.backend == "classifier":
            export_chips(store, target, classes=profile.model.classes)
        else:
            export_coco(store, profile, target, classes=profile.model.classes,
                        require_complete_tiles=not args.allow_incomplete,
                        write_images=not args.no_images)


def cmd_train(args):
    from tseg import device
    from tseg.training.finetune import run_round

    profile = _profile_from(args, from_run=True)
    run_round(profile, args.out, round_no=args.round, epochs=args.epochs,
              batch_size=args.batch_size,
              device_name=args.device or device.resolve(),
              resume=not args.no_resume,
              allow_incomplete=args.allow_incomplete)


def cmd_merge(args):
    from tseg.imagery.cache import TileCache
    from tseg.pipeline import merge

    profile = _profile_from(args, from_run=True)
    out = Path(args.out)
    cache = TileCache(out)
    shapes = tuple(s.strip() for s in args.shapes.split(",") if s.strip())
    iou = profile.model.dedupe_iou if args.dedupe is None else args.dedupe

    ext = {"gpkg": "gpkg", "fgb": "fgb", "geojson": "geojson"}[args.format]
    stem = args.name or f"{profile.name}_{profile.model.classes[0]}"
    merge(cache, out / f"{stem}.{ext}", fmt=args.format, shapes=shapes,
          dedupe_iou=iou, layer_name=profile.name, profile=profile)


COMMANDS = {
    "info": cmd_info,
    "detect": cmd_detect,
    "pand": cmd_pand,
    "review": cmd_review,
    "mine": cmd_mine,
    "export": cmd_export,
    "train": cmd_train,
    "merge": cmd_merge,
}


def main(argv=None):
    args = parse_args(argv)
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
