"""Test DeepForest on PDOK 25cm aerial RGB (Actueel_ortho25 WMS).

DeepForest = deep-learning individual-tree-crown *detection* (RetinaNet),
pretrained on NEON airborne RGB. It returns bounding boxes per tree, unlike
DetecTree (pixel tree/non-tree) or PyCrown (CHM crown polygons).

Pipeline (mirrors detectree_pdok.py):
  1. WMS GetMap JPEG -> georeferenced GeoTIFF (EPSG:28992, 25 cm).
  2. DeepForest pretrained model -> per-tree bounding boxes via predict_tile.
  3. Export boxes as GeoJSON (RD New) + a preview PNG with boxes drawn.

Caveat: DeepForest was trained at ~0.1 m NEON imagery; PDOK is 0.25 m, so
crowns span fewer pixels than in training -> expect small/young trees missed.
"""

import torch  # noqa: F401  -- MUST precede rasterio on Windows (torch c10.dll init
#                              fails if GDAL/OpenMP DLLs load first: WinError 1114)

import argparse
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image, ImageDraw

WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
LAYER = "Actueel_ortho25"   # 25 cm RGB; leaf-on (summer). The 5/8cm orthoHR
#                             products are leaf-off (winter) -> worse for detection.
CRS = "EPSG:28992"
EPSG = 28992
RES = 0.25

AREAS = {
    "oudzuid": (120128, 485328),
    "vondelpark": (120400, 485850),
    "bos": (117500, 480000),
}

p = argparse.ArgumentParser()
p.add_argument("area", nargs="?", default="vondelpark", choices=AREAS)
p.add_argument("--size", type=int, default=300, help="tile size in meters")
p.add_argument("--patch", type=int, default=400, help="predict_tile patch size (px)")
p.add_argument("--thresh", type=float, default=0.2, help="score threshold")
args = p.parse_args()

CX, CY = AREAS[args.area]
S = args.size
XMIN, YMIN, XMAX, YMAX = CX - S / 2, CY - S / 2, CX + S / 2, CY + S / 2
NPX = int(S / RES)

OUT = Path("output")
OUT.mkdir(exist_ok=True)
TAG = f"{args.area}_df"
RGB_TIF = OUT / f"{TAG}_rgb.tif"
GEOJSON = OUT / f"{TAG}_boxes.geojson"
PREVIEW = OUT / f"{TAG}_preview.png"


MAXPX = 2000  # PDOK GetMap rejects very large requests ("image size too large")


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


def download_wms() -> np.ndarray:
    """Fetch the tile as a mosaic of <=MAXPX sub-blocks (WMS size cap)."""
    arr = np.zeros((NPX, NPX, 3), dtype="uint8")
    cuts = list(range(0, NPX, MAXPX)) + [NPX]
    nblk = (len(cuts) - 1) ** 2
    print(f"GET {LAYER} {NPX}x{NPX}px in {nblk} block(s)")
    for r0, r1 in zip(cuts, cuts[1:]):
        for c0, c1 in zip(cuts, cuts[1:]):
            bx0 = XMIN + c0 * RES
            bx1 = XMIN + c1 * RES
            by1 = YMAX - r0 * RES   # top
            by0 = YMAX - r1 * RES   # bottom
            arr[r0:r1, c0:c1] = _get_block(bx0, by0, bx1, by1, c1 - c0, r1 - r0)
    print("downloaded", arr.shape)
    return arr


def write_geotiff(arr: np.ndarray, path: Path) -> None:
    transform = from_bounds(XMIN, YMIN, XMAX, YMAX, arr.shape[1], arr.shape[0])
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=3, dtype="uint8", crs=CRS, transform=transform,
    ) as dst:
        for b in range(3):
            dst.write(arr[:, :, b], b + 1)
    print("wrote", path)


def load_model():
    from deepforest import main
    m = main.deepforest()
    m.load_model("weecology/deepforest-tree")  # pretrained NEON tree model (HF)
    return m


def pixel_boxes(model, path):
    """Run predict_tile, return DataFrame with pixel xmin/ymin/xmax/ymax/score."""
    boxes = model.predict_tile(
        path=str(path), patch_size=args.patch,
        patch_overlap=0.25, iou_threshold=0.15,
    )
    if boxes is None or len(boxes) == 0:
        return None
    print("box columns:", list(boxes.columns))
    # ensure pixel bbox columns exist (derive from geometry if needed)
    if "xmin" not in boxes.columns and "geometry" in boxes.columns:
        b = boxes.geometry.bounds
        boxes["xmin"], boxes["ymin"] = b.minx, b.miny
        boxes["xmax"], boxes["ymax"] = b.maxx, b.maxy
    boxes = boxes[boxes["score"] >= args.thresh].reset_index(drop=True)
    return boxes


def main_run() -> None:
    rgb = download_wms()
    write_geotiff(rgb, RGB_TIF)

    model = load_model()
    boxes = pixel_boxes(model, RGB_TIF)
    if boxes is None:
        print("no trees detected")
        return
    print(f"trees detected (score>={args.thresh}): {len(boxes)}")

    # pixel boxes -> RD New coords via the raster transform
    transform = from_bounds(XMIN, YMIN, XMAX, YMAX, NPX, NPX)
    feats = []
    for _, r in boxes.iterrows():
        x0, y0 = transform * (r.xmin, r.ymin)
        x1, y1 = transform * (r.xmax, r.ymax)
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"score": float(r.score)}})
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": f"EPSG:{EPSG}"}},
          "features": feats}
    GEOJSON.write_text(json.dumps(fc))
    print("wrote", GEOJSON, f"({len(feats)} boxes)")

    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for _, r in boxes.iterrows():
        draw.rectangle([r.xmin, r.ymin, r.xmax, r.ymax], outline=(255, 0, 0), width=2)
    img.save(PREVIEW)
    print("wrote", PREVIEW)
    print(f"\nDONE {args.area}: {len(boxes)} trees")


if __name__ == "__main__":
    main_run()
