"""Batch DeepForest tree detection over a 'bestuurlijk gebied' (province).

Adapted from deepforest_pdok.py. Instead of one hand-picked tile it:
  1. Pulls a province polygon from the PDOK WFS (brk-bestuurlijke-gebieden).
  2. Covers its bbox with a regular grid of tiles (EPSG:28992).
  3. Keeps only tiles intersecting the province polygon.
  4. Runs the pretrained DeepForest model on each tile (25 cm leaf-on RGB).
  5. Merges every detected tree box into ONE GeoJSON (EPSG:28992).

Per-tile results are cached so the job is resumable. Use --max-tiles to cap the
run (a full province is thousands of tiles / hours-days on CPU).

First test: province Limburg (code 31).
"""

import torch  # noqa: F401  -- MUST precede rasterio on Windows (c10.dll init order)

import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
from shapely.geometry import shape, box, Point
from shapely.prepared import prep

WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
WFS = "https://service.pdok.nl/kadaster/brk-bestuurlijke-gebieden/wfs/v1_0"
PROV_TYPE = "bestuurlijkegebieden:Provinciegebied"
LAYER = "Actueel_ortho25"   # 25 cm leaf-on RGB
CRS = "EPSG:28992"
EPSG = 28992
RES = 0.25
MAXPX = 2000                # PDOK GetMap size cap per request

ap = argparse.ArgumentParser()
ap.add_argument("--code", default="31", help="province code (31 = Limburg)")
ap.add_argument("--name", default=None, help="output name (default: province code)")
ap.add_argument("--tile-size", type=int, default=500, help="tile size in meters")
ap.add_argument("--max-tiles", type=int, default=6,
                help="process at most N tiles (0 = all; full province = hours+)")
ap.add_argument("--patch", type=int, default=400, help="predict_tile patch size (px)")
ap.add_argument("--thresh", type=float, default=0.2, help="score threshold")
args = ap.parse_args()

NAME = args.name or f"prov{args.code}"
TILE = args.tile_size
NPX = int(TILE / RES)

OUT = Path("output") / f"{NAME}_deepforest"
TILE_CACHE = OUT / "tiles"
OUT.mkdir(parents=True, exist_ok=True)
TILE_CACHE.mkdir(exist_ok=True)
MERGED = OUT / f"{NAME}_trees.geojson"


def fetch_province(code: str):
    """WFS GetFeature -> shapely (Multi)Polygon of the province in EPSG:28992.

    There are only 12 provinces and this WFS ignores CQL_FILTER, so fetch all
    and select by 'code' client-side.
    """
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": PROV_TYPE, "outputFormat": "application/json",
        "srsName": f"urn:ogc:def:crs:EPSG::{EPSG}",
    }
    url = f"{WFS}?" + urllib.parse.urlencode(params)
    d = json.load(urllib.request.urlopen(url, timeout=120))
    match = [f for f in d["features"] if str(f["properties"]["code"]) == str(code)]
    if not match:
        raise SystemExit(f"no province with code {code}")
    f = match[0]
    print(f"province: {f['properties']['naam']} (code {code})")
    return shape(f["geometry"])


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
    arr = np.zeros((NPX, NPX, 3), dtype="uint8")
    cuts = list(range(0, NPX, MAXPX)) + [NPX]
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


def make_grid(poly):
    """Snap-to-grid tiles covering poly.bounds, keep those intersecting poly."""
    minx, miny, maxx, maxy = poly.bounds
    x0 = (int(minx) // TILE) * TILE
    y0 = (int(miny) // TILE) * TILE
    pgeom = prep(poly)
    tiles = []
    y = y0
    while y < maxy:
        x = x0
        while x < maxx:
            if pgeom.intersects(box(x, y, x + TILE, y + TILE)):
                tiles.append((x, y))
            x += TILE
        y += TILE
    return tiles


def load_model():
    from deepforest import main
    m = main.deepforest()
    m.load_model("weecology/deepforest-tree")
    return m


def detect_tile(model, xmin, ymin, tmp_tif):
    """Return list of GeoJSON features (boxes in EPSG:28992) for one tile."""
    xmax, ymax = xmin + TILE, ymin + TILE
    arr = download_tile(xmin, ymin, xmax, ymax)
    if arr.max() == 0:
        return []                       # fully outside coverage
    write_geotiff(arr, tmp_tif, xmin, ymin, xmax, ymax)
    boxes = model.predict_tile(path=str(tmp_tif), patch_size=args.patch,
                               patch_overlap=0.25, iou_threshold=0.15)
    if boxes is None or len(boxes) == 0:
        return []
    boxes = boxes[boxes["score"] >= args.thresh]
    t = from_bounds(xmin, ymin, xmax, ymax, NPX, NPX)
    feats = []
    for _, r in boxes.iterrows():
        gx0, gy0 = t * (r.xmin, r.ymin)
        gx1, gy1 = t * (r.xmax, r.ymax)
        # keep only boxes whose centre falls in this tile (avoids double-count
        # of crowns clipped at the shared edge of neighbouring tiles)
        cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
        if not (xmin <= cx < xmax and ymin <= cy < ymax):
            continue
        ring = [[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1], [gx0, gy0]]
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"score": float(r.score),
                                     "cx": cx, "cy": cy}})
    return feats


def main():
    poly = fetch_province(args.code)
    tiles = make_grid(poly)
    print(f"grid: {len(tiles)} tiles of {TILE} m intersect the province "
          f"(~{len(tiles) * TILE * TILE / 1e6:.0f} km2 covered)")
    todo = tiles if args.max_tiles == 0 else tiles[: args.max_tiles]
    print(f"processing {len(todo)} tile(s) this run")

    # keep only boxes whose centre is truly inside the province polygon
    pgeom = prep(poly)
    model = load_model()
    tmp_tif = TILE_CACHE / "_tmp.tif"

    all_feats = []
    for i, (x, y) in enumerate(todo, 1):
        cache = TILE_CACHE / f"{int(x)}_{int(y)}.geojson"
        if cache.exists():
            feats = json.loads(cache.read_text())["features"]
        else:
            t0 = time.time()
            feats = detect_tile(model, x, y, tmp_tif)
            feats = [f for f in feats
                     if pgeom.contains(Point(f["properties"]["cx"],
                                             f["properties"]["cy"]))]
            cache.write_text(json.dumps(
                {"type": "FeatureCollection", "features": feats}))
            print(f"  [{i}/{len(todo)}] tile {int(x)},{int(y)}: "
                  f"{len(feats)} trees ({time.time() - t0:.0f}s)")
        all_feats.extend(feats)

    for f in all_feats:                 # drop helper props
        f["properties"].pop("cx", None)
        f["properties"].pop("cy", None)
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": f"EPSG:{EPSG}"}},
          "name": NAME, "features": all_feats}
    MERGED.write_text(json.dumps(fc))
    print(f"\nDONE: {len(all_feats)} trees across {len(todo)} tiles -> {MERGED}")


if __name__ == "__main__":
    main()
