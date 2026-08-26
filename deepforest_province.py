# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Batch DeepForest tree detection over a selection of 'bestuurlijke gebieden'.

This is now a thin shim over the tseg package. The CLI flags, the tile cache
layout and the merged output path are unchanged, so the existing Limburg run
(output/limburg_gemeenten_deepforest/, 6816 cached tiles, and the GeoDMS FSS
store built from its merged GeoJSON) keeps working exactly as before.

What it does, unchanged:
  1. Pulls gemeente polygons from the PDOK WFS and dissolves the ones matching
     the given CBS gemeentecodes into one AOI.
  2. Covers the AOI bbox with a regular grid of tiles (EPSG:28992).
  3. Runs the pretrained DeepForest model on each tile (25 cm leaf-on RGB),
     in parallel across N worker processes.
  4. Merges every detected tree box into ONE GeoJSON (EPSG:28992).

Per-tile results are cached so the job is resumable. Legacy .geojson cache
entries written before tseg are read as-is and never recomputed.

Defaults reproduce the original behaviour bit for bit: overlap 0 and no
cross-tile NMS. Pass --overlap / --dedupe to get the improved handling of
crowns that straddle a tile boundary, or use the tseg CLI directly:

    tseg detect --profile trees --codes <codes> --out output/<name>
"""

import argparse
import sys
from pathlib import Path

from tseg.aoi.bestuurlijk import DEFAULT_CODES, fetch_area
from tseg.config import load_profile
from tseg.imagery.cache import TileCache
from tseg.imagery.grid import make_grid
from tseg.io.writer import open_writer
from tseg.legacy import iter_legacy_cache, legacy_keys
from tseg.pipeline import run_tiles


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", default=DEFAULT_CODES,
                    help="comma-separated CBS gemeentecodes (GM-prefixed)")
    ap.add_argument("--name", default="limburg_gemeenten", help="output name")
    ap.add_argument("--tile-size", type=int, default=500, help="tile size (m)")
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="process at most N new tiles (0 = all)")
    ap.add_argument("--patch", type=int, default=400, help="predict_tile patch (px)")
    ap.add_argument("--thresh", type=float, default=0.2, help="score threshold")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes (each loads its own model "
                         "-> ~1-2 GB RAM each; also concurrent WMS requests)")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="tile overlap in m (0 = original behaviour; 25 lets a "
                         "boundary crown be seen whole at least once)")
    ap.add_argument("--dedupe", type=float, default=0.0,
                    help="cross-tile NMS IoU on merge (0 = off, as before)")
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path("output") / f"{args.name}_deepforest"
    cache_dir = out_dir / "tiles"
    merged = out_dir / f"{args.name}_trees.geojson"

    profile = load_profile("trees")
    profile.grid.tile_m = args.tile_size
    profile.grid.overlap_m = args.overlap
    profile.model.patch = args.patch
    profile.model.score_thresh = args.thresh

    poly = fetch_area(args.codes)
    tiles = make_grid(poly, args.tile_size, args.overlap)
    print(f"grid: {len(tiles)} tiles of {args.tile_size} m intersect the area "
          f"(~{len(tiles) * args.tile_size ** 2 / 1e6:.0f} km2)")

    # Legacy entries are finished work: skip them rather than recompute.
    done = set(legacy_keys(cache_dir))
    if done:
        print(f"{len(done)} tile(s) already present in the pre-tseg cache")
    todo = [t for t in tiles if t.key not in done]

    run_tiles(profile, poly, out_dir, max_tiles=args.max_tiles,
              workers=args.workers, tiles=todo)

    # Merge legacy and new cache entries into the one GeoJSON the FSS store reads.
    cache = TileCache(out_dir)
    n = 0
    with open_writer(merged, "geojson", shapes=("bbox",),
                     layer_name=args.name, profile=profile) as w:
        if args.dedupe > 0:
            from tseg.geometry.dedupe import nms

            feats = list(iter_legacy_cache(cache_dir)) + list(cache.iter_all())
            w.extend(nms(feats, args.dedupe))
        else:
            for f in iter_legacy_cache(cache_dir):
                w.write(f)
            for f in cache.iter_all():
                w.write(f)
        n = w.count

    print(f"\nDONE: {n} trees -> {merged}")


if __name__ == "__main__":
    sys.exit(main())
