"""Test DetecTree on PDOK 25cm aerial RGB (Actueel_ortho25 WMS).

Pipeline:
  1. Download an RGB tile from the PDOK luchtfoto WMS (JPEG, only format offered).
  2. Wrap the JPEG into a georeferenced GeoTIFF (EPSG:28992 / RD New).
  3. Run the DetecTree pretrained classifier -> binary tree/non-tree mask.
  4. Write mask GeoTIFF + a side-by-side PNG preview.
"""

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

import detectree as dtr

WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
LAYER = "Actueel_ortho25"      # 25 cm current RGB orthophoto
CRS = "EPSG:28992"             # RD New (meters)
RES = 0.25                     # meters/pixel

# Predefined test areas: (cx, cy) center in RD New (EPSG:28992).
AREAS = {
    "oudzuid": (120128, 485328),      # residential Amsterdam-Zuid (mixed)
    "vondelpark": (120400, 485850),   # Vondelpark, Amsterdam
    "bos": (117500, 480000),          # Amsterdamse Bos (dense forest)
}

p = argparse.ArgumentParser()
p.add_argument("area", nargs="?", default="oudzuid", choices=AREAS)
p.add_argument("--size", type=int, default=256, help="tile size in meters")
args = p.parse_args()

CX, CY = AREAS[args.area]
SIZE_M = args.size
XMIN, YMIN = CX - SIZE_M / 2, CY - SIZE_M / 2
XMAX, YMAX = CX + SIZE_M / 2, CY + SIZE_M / 2
NPX = int(SIZE_M / RES)

OUT = Path("output")
OUT.mkdir(exist_ok=True)
RGB_TIF = OUT / f"{args.area}_rgb.tif"
MASK_TIF = OUT / f"{args.area}_treemask.tif"
PREVIEW = OUT / f"{args.area}_preview.png"


def download_wms() -> np.ndarray:
    """GetMap JPEG -> HxWx3 uint8 array."""
    params = {
        "service": "WMS", "request": "GetMap", "version": "1.3.0",
        "layers": LAYER, "styles": "", "crs": CRS,
        "bbox": f"{XMIN},{YMIN},{XMAX},{YMAX}",
        "width": NPX, "height": NPX, "format": "image/jpeg",
    }
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{WMS}?{q}"
    print("GET", url)
    data = urllib.request.urlopen(url, timeout=120).read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(img)
    print("downloaded", arr.shape, arr.dtype)
    return arr


def write_geotiff(arr: np.ndarray, path: Path) -> None:
    """Write HxWx3 uint8 as a 3-band georeferenced GeoTIFF."""
    transform = from_bounds(XMIN, YMIN, XMAX, YMAX, arr.shape[1], arr.shape[0])
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=3,
        dtype="uint8", crs=CRS, transform=transform,
    ) as dst:
        for b in range(3):
            dst.write(arr[:, :, b], b + 1)
    print("wrote", path)


def main() -> None:
    rgb = download_wms()
    write_geotiff(rgb, RGB_TIF)

    # Pretrained classifier auto-downloads from HuggingFace (martibosch/detectree).
    clf = dtr.Classifier()
    y_pred = clf.predict_img(str(RGB_TIF), output_filepath=str(MASK_TIF))
    tree_frac = y_pred.mean()
    print(f"tree cover fraction: {tree_frac:.1%}  mask -> {MASK_TIF}")

    # Preview: RGB | mask | overlay
    mask = (y_pred > 0)
    overlay = rgb.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype("uint8")
    mask_rgb = np.stack([np.zeros_like(mask), mask.astype("uint8") * 255,
                         np.zeros_like(mask)], axis=-1)
    strip = np.concatenate([rgb, mask_rgb, overlay], axis=1)
    Image.fromarray(strip).save(PREVIEW)
    print("wrote", PREVIEW)


if __name__ == "__main__":
    main()
