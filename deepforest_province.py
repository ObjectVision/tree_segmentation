"""Batch DeepForest tree detection over a selection of 'bestuurlijke gebieden'.

Adapted from deepforest_pdok.py. Instead of one hand-picked tile it:
  1. Pulls gemeente polygons from the PDOK WFS (brk-bestuurlijke-gebieden) and
     dissolves the ones matching the given CBS gemeentecodes into one AOI.
  2. Covers the AOI bbox with a regular grid of tiles (EPSG:28992).
  3. Keeps only tiles intersecting the AOI polygon.
  4. Runs the pretrained DeepForest model on each tile (25 cm leaf-on RGB),
     in parallel across N worker processes.
  5. Merges every detected tree box into ONE GeoJSON (EPSG:28992).

Per-tile results are cached so the job is resumable: cached tiles are never
re-processed, and the merge always reads the whole cache. Use --max-tiles to
cap new work per run (the full AOI is thousands of tiles / hours on CPU).

Default AOI: Noord, Zuid and part of Midden Limburg (gemeenten).
"""

import torch  # noqa: F401  -- MUST precede rasterio on Windows (c10.dll init order)

import argparse
import io
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import tqdm
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
from shapely.geometry import shape, box, Point
from shapely.ops import unary_union
from shapely.prepared import prep

WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
WFS = "https://service.pdok.nl/kadaster/brk-bestuurlijke-gebieden/wfs/v1_0"
GEM_TYPE = "bestuurlijkegebieden:Gemeentegebied"
LAYER = "Actueel_ortho25"   # 25 cm leaf-on RGB
CRS = "EPSG:28992"
EPSG = 28992
RES = 0.25
MAXPX = 2000                # PDOK GetMap size cap per request

# Noord, Zuid and part of Midden Limburg (CBS gemeentecodes, GM-prefixed).
DEFAULT_CODES = (
    "GM0889,GM0893,GM0907,GM1507,GM0944,GM1894,GM0983,GM0984,GM0957,GM0888,"
    "GM1954,GM0899,GM1903,GM1729,GM0917,GM0928,GM0882,GM0935,GM0938,GM0965,"
    "GM1883,GM0971,GM0981,GM0994,GM0986"
)

# Per-process state (set by configure(); model/pgeom populated lazily/in workers)
CFG = None       # SimpleNamespace with tile geometry + inference params
PGEOM = None     # prepared AOI polygon (for centre-in-AOI filtering)
MODEL = None     # one DeepForest model per process


def configure(cfg: dict):
    """Populate per-process config (called in the parent and every worker)."""
    global CFG
    CFG = SimpleNamespace(**cfg)
    CFG.npx = int(CFG.tile / RES)
    CFG.cache = Path(cfg["cache"])
    CFG.cache.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- WFS / grid

def fetch_area(codes: set):
    """WFS GetFeature -> unioned shapely geometry of the selected gemeenten.

    This WFS ignores CQL_FILTER, so fetch all gemeenten and select client-side
    by 'identificatie' (the GM-prefixed CBS code), then dissolve into one AOI.
    """
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": GEM_TYPE, "outputFormat": "application/json",
        "srsName": f"urn:ogc:def:crs:EPSG::{EPSG}",
    }
    url = f"{WFS}?" + urllib.parse.urlencode(params)
    d = json.load(urllib.request.urlopen(url, timeout=120))
    picked = [f for f in d["features"]
              if str(f["properties"]["identificatie"]).upper() in codes]
    found = {f["properties"]["identificatie"].upper() for f in picked}
    missing = codes - found
    if missing:
        print(f"WARNING: {len(missing)} code(s) not found: {sorted(missing)}")
    if not picked:
        raise SystemExit("no matching gemeenten")
    names = ", ".join(sorted(f["properties"]["naam"] for f in picked))
    print(f"selected {len(picked)} gemeenten: {names}")
    return unary_union([shape(f["geometry"]) for f in picked])


def make_grid(poly, tile):
    """Snap-to-grid tiles covering poly.bounds, keep those intersecting poly."""
    minx, miny, maxx, maxy = poly.bounds
    x0 = (int(minx) // tile) * tile
    y0 = (int(miny) // tile) * tile
    pgeom = prep(poly)
    tiles = []
    y = y0
    while y < maxy:
        x = x0
        while x < maxx:
            if pgeom.intersects(box(x, y, x + tile, y + tile)):
                tiles.append((x, y))
            x += tile
        y += tile
    return tiles


# ---------------------------------------------------------------- WMS / raster

def _get_block(x0, y0, x1, y1, w, h) -> np.ndarray:
    params = {
        "service": "WMS", "request": "GetMap", "version": "1.3.0",
        "layers": LAYER, "styles": "", "crs": CRS,
        "bbox": f"{x0},{y0},{x1},{y1}",
        "width": w, "height": h, "format": "image/jpeg",
    }
    url = f"{WMS}?" + "&".join(f"{k}={v}" for k, v in params.items())
    data = urllib.request.urlopen(url, timeout=120).read()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def download_tile(xmin, ymin, xmax, ymax) -> np.ndarray:
    """Fetch one grid tile as a mosaic of <=MAXPX sub-blocks."""
    npx = CFG.npx
    arr = np.zeros((npx, npx, 3), dtype="uint8")
    cuts = list(range(0, npx, MAXPX)) + [npx]
    for r0, r1 in zip(cuts, cuts[1:]):
        for c0, c1 in zip(cuts, cuts[1:]):
            bx0, bx1 = xmin + c0 * RES, xmin + c1 * RES
            by1, by0 = ymax - r0 * RES, ymax - r1 * RES
            arr[r0:r1, c0:c1] = _get_block(bx0, by0, bx1, by1, c1 - c0, r1 - r0)
    return arr


def write_geotiff(arr, path, xmin, ymin, xmax, ymax):
    t = from_bounds(xmin, ymin, xmax, ymax, arr.shape[1], arr.shape[0])
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=3, dtype="uint8",
                       crs=CRS, transform=t) as dst:
        for b in range(3):
            dst.write(arr[:, :, b], b + 1)


# ---------------------------------------------------------------- inference

def _model():
    global MODEL
    if MODEL is None:
        from deepforest import main
        MODEL = main.deepforest()
        MODEL.load_model("weecology/deepforest-tree")
    return MODEL


def cache_path(x, y):
    return CFG.cache / f"{int(x)}_{int(y)}.geojson"


def process_tile(xy):
    """Worker task: detect trees in one tile, write its cache file.

    Returns (x, y, n_trees). Skips (and returns -1) if already cached.
    """
    x, y = xy
    cache = cache_path(x, y)
    if cache.exists():
        return (x, y, -1)

    tile = CFG.tile
    xmax, ymax = x + tile, y + tile
    tmp_tif = CFG.cache / f"_tmp_{os.getpid()}.tif"
    arr = download_tile(x, y, xmax, ymax)

    feats = []
    if arr.max() > 0:                         # skip fully-outside-coverage tiles
        write_geotiff(arr, tmp_tif, x, y, xmax, ymax)
        boxes = _model().predict_tile(path=str(tmp_tif), patch_size=CFG.patch,
                                      patch_overlap=0.25, iou_threshold=0.15)
        if boxes is not None and len(boxes):
            boxes = boxes[boxes["score"] >= CFG.thresh]
            t = from_bounds(x, y, xmax, ymax, CFG.npx, CFG.npx)
            for _, r in boxes.iterrows():
                gx0, gy0 = t * (r.xmin, r.ymin)
                gx1, gy1 = t * (r.xmax, r.ymax)
                cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
                # centre must be in this tile (dedupe edge-split crowns) and in AOI
                if not (x <= cx < xmax and y <= cy < ymax):
                    continue
                if PGEOM is not None and not PGEOM.contains(Point(cx, cy)):
                    continue
                ring = [[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1], [gx0, gy0]]
                feats.append({"type": "Feature",
                              "geometry": {"type": "Polygon", "coordinates": [ring]},
                              "properties": {"score": float(r.score)}})
    cache.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return (x, y, len(feats))


def init_worker(cfg: dict, poly, n_threads: int):
    """Process-pool initializer: set torch threads, config and AOI polygon."""
    torch.set_num_threads(max(1, n_threads))
    configure(cfg)
    global PGEOM
    PGEOM = prep(poly)


# ---------------------------------------------------------------- driver

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=DEFAULT_CODES,
                    help="comma-separated CBS gemeentecodes (GM-prefixed)")
    ap.add_argument("--name", default="limburg_gemeenten", help="output name")
    ap.add_argument("--tile-size", type=int, default=500, help="tile size (m)")
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="process at most N new tiles (0 = all)")
    ap.add_argument("--patch", type=int, default=400, help="predict_tile patch (px)")
    ap.add_argument("--thresh", type=float, default=0.2, help="score threshold")
    ap.add_argument("--workers", type=int,
                    default=min(8, max(1, (os.cpu_count() or 2) // 2)),
                    help="parallel worker processes (each loads its own model "
                         "-> ~1-2 GB RAM each; also concurrent WMS requests)")
    return ap.parse_args()


def main():
    args = parse_args()
    codes = {c.strip().upper() for c in args.codes.split(",") if c.strip()}
    out_dir = Path("output") / f"{args.name}_deepforest"
    cache_dir = out_dir / "tiles"
    merged = out_dir / f"{args.name}_trees.geojson"

    cfg = {"tile": args.tile_size, "patch": args.patch,
           "thresh": args.thresh, "cache": str(cache_dir)}
    configure(cfg)                     # parent needs CFG for the merge/paths

    poly = fetch_area(codes)
    tiles = make_grid(poly, args.tile_size)
    print(f"grid: {len(tiles)} tiles of {args.tile_size} m intersect the area "
          f"(~{len(tiles) * args.tile_size ** 2 / 1e6:.0f} km2)")

    uncached = [xy for xy in tiles if not cache_path(*xy).exists()]
    pending = uncached[: args.max_tiles] if args.max_tiles > 0 else uncached
    n_workers = max(1, min(args.workers, len(pending) or 1))
    threads = max(1, (os.cpu_count() or 2) // n_workers)
    print(f"{len(tiles) - len(uncached)} already cached, {len(uncached)} remaining; "
          f"processing {len(pending)} this run on {n_workers} worker(s) "
          f"({threads} torch threads each)")

    if pending:
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker,
                                 initargs=(cfg, poly, threads)) as ex:
            futs = {ex.submit(process_tile, xy): xy for xy in pending}
            for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc="tiles"):
                x, y, n = fut.result()
                done += 1
                if n >= 0:
                    tqdm.tqdm.write(f"  {int(x)},{int(y)}: {n} trees "
                                    f"[{done}/{len(pending)}]")
        rate = (time.time() - t0) / len(pending)
        print(f"processed {len(pending)} tiles at {rate:.1f}s/tile")

    # merge every cached tile (this run + previous runs) into one GeoJSON
    all_feats = []
    for cache in sorted(cache_dir.glob("*.geojson")):
        all_feats.extend(json.loads(cache.read_text())["features"])
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": f"EPSG:{EPSG}"}},
          "name": args.name, "features": all_feats}
    merged.write_text(json.dumps(fc))
    ncached = len(list(cache_dir.glob("*.geojson")))
    print(f"\nDONE: {len(all_feats)} trees across {ncached} cached tiles -> {merged}")


if __name__ == "__main__":
    main()
