"""Test PyCrown on PDOK AHN LiDAR-derived rasters (rasterized LiDAR).

PyCrown does individual-tree-crown delineation from a Canopy Height Model.
AHN (Actueel Hoogtebestand Nederland) is the national LiDAR elevation product.
Its WMS only serves rendered 8-bit images, so we use the sibling WCS endpoint,
which serves the actual float32 heights we need to build a metric CHM.

Pipeline (mirrors detectree_pdok.py):
  1. WCS GetCoverage -> float32 DSM + DTM GeoTIFFs (EPSG:28992, 0.5 m).
  2. CHM = DSM - DTM (nodata handled).
  3. Real PyCrown: smooth -> local-maxima tree tops -> Dalponte crown delineation.
  4. Export tree tops + crown polygons as GeoJSON, plus a preview PNG.

NB: LAS-dependent PyCrown steps (correct_tree_tops, crowns_to_polys_smooth) are
skipped -- AHN gives rasters, not a point cloud.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image, ImageDraw
from shapely.geometry import mapping

sys.path.insert(0, str(Path(__file__).parent / "pycrown"))
from pycrown import PyCrown

WCS = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"
EPSG = 28992
RES = 0.5  # AHN native resolution, m/px

AREAS = {
    "bos": (117500, 480000),        # Amsterdamse Bos - dense forest
    "vondelpark": (120400, 485850),  # urban park
}

p = argparse.ArgumentParser()
p.add_argument("area", nargs="?", default="bos", choices=AREAS)
p.add_argument("--size", type=int, default=300, help="tile size in meters")
p.add_argument("--hmin", type=float, default=8.0, help="min tree height (m)")
args = p.parse_args()

CX, CY = AREAS[args.area]
S = args.size
XMIN, YMIN, XMAX, YMAX = CX - S / 2, CY - S / 2, CX + S / 2, CY + S / 2

OUT = Path("output")
OUT.mkdir(exist_ok=True)
DSM_TIF = OUT / f"{args.area}_dsm.tif"
DTM_TIF = OUT / f"{args.area}_dtm.tif"
CHM_TIF = OUT / f"{args.area}_chm.tif"
RESULT = OUT / f"{args.area}_pycrown"


def wcs_download(coverage: str, path: Path) -> None:
    """WCS GetCoverage -> float32 GeoTIFF on disk."""
    q = (
        f"{WCS}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={coverage}&FORMAT=image/tiff"
        f"&SUBSETTINGCRS={CRS_URI}&OUTPUTCRS={CRS_URI}"
        f"&SUBSET=x({XMIN},{XMAX})&SUBSET=y({YMIN},{YMAX})"
    )
    print("GET", coverage)
    data = urllib.request.urlopen(q, timeout=180).read()
    path.write_bytes(data)
    with rasterio.open(path) as ds:
        print(f"  {coverage}: {ds.width}x{ds.height} {ds.dtypes[0]} "
              f"nodata={ds.nodata} crs={ds.crs}")


def build_chm() -> None:
    with rasterio.open(DSM_TIF) as ds:
        dsm = ds.read(1).astype("float32")
        prof = ds.profile
        dsm_nd = ds.nodata
    with rasterio.open(DTM_TIF) as dt:
        dtm = dt.read(1).astype("float32")
        dtm_nd = dt.nodata

    valid = np.ones(dsm.shape, bool)
    for arr, nd in ((dsm, dsm_nd), (dtm, dtm_nd)):
        if nd is not None:
            valid &= arr != nd
        valid &= np.isfinite(arr)
    chm = np.where(valid, dsm - dtm, 0.0).astype("float32")
    chm[chm < 0] = 0.0  # negative = artefact

    prof.update(dtype="float32", count=1, nodata=0.0)
    with rasterio.open(CHM_TIF, "w", **prof) as dst:
        dst.write(chm, 1)
    print(f"CHM built: max={chm.max():.1f}m  >{args.hmin}m cover="
          f"{(chm > args.hmin).mean():.1%}")


def run_pycrown() -> PyCrown:
    PC = PyCrown(CHM_TIF, DTM_TIF, DSM_TIF, outpath=RESULT)
    PC.filter_chm(5, ws_in_pixels=True)                       # 5px median smooth
    PC.tree_detection(PC.chm, ws=5, ws_in_pixels=True, hmin=args.hmin)
    print(f"tree tops (local maxima): {len(PC.trees)}")
    PC.clip_trees_to_bbox(inbuf=int(2 * RES) + 2)             # drop edge trees
    PC.crown_delineation(algorithm="dalponteCIRC_numba", th_tree=args.hmin,
                         th_seed=0.7, th_crown=0.55, max_crown=15.)
    PC.get_tree_height_elevation(loc="top")
    PC.screen_small_trees(hmin=args.hmin, loc="top")
    PC.crowns_to_polys_raster()
    # all_good=True: no tree-top correction (that step needs a LiDAR point cloud,
    # which AHN rasters don't provide), so skip the tt_corrected filter.
    PC.quality_control(all_good=True)
    print(f"trees after delineation + QC: {len(PC.trees)}")
    return PC


def export_geojson(PC: PyCrown) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    for name, col in (("tops", "top"), ("crowns", "crown_poly_raster")):
        feats = []
        for i in range(len(PC.trees)):
            t = PC.trees.iloc[i]
            geom = t[col]
            if geom is None:
                continue
            feats.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"id": int(i),
                               "height_m": float(t.top_height)},
            })
        fc = {"type": "FeatureCollection",
              "crs": {"type": "name",
                      "properties": {"name": f"EPSG:{EPSG}"}},
              "features": feats}
        (RESULT / f"tree_{name}.geojson").write_text(json.dumps(fc))
        print(f"wrote {RESULT / f'tree_{name}.geojson'} ({len(feats)} feats)")


def preview(PC: PyCrown) -> None:
    """CHM grayscale + crown outlines (cyan) + tree tops (red)."""
    chm = PC.chm
    g = np.clip(chm / max(chm.max(), 1) * 255, 0, 255).astype("uint8")
    rgb = np.stack([g, g, g], -1)

    crowns = PC.crowns  # labeled crown raster
    if crowns is not None:
        b = np.zeros(crowns.shape, bool)
        b[:-1] |= crowns[:-1] != crowns[1:]
        b[:, :-1] |= crowns[:, :-1] != crowns[:, 1:]
        b &= crowns > 0
        rgb[b] = (0, 255, 255)

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    res = PC.resolution
    for i in range(len(PC.trees)):
        pt = PC.trees.iloc[i]["top"]
        col = (pt.x - PC.ul_lon) / res
        row = (PC.ul_lat - pt.y) / res
        draw.ellipse([col - 2, row - 2, col + 2, row + 2],
                     fill=(255, 0, 0))
    out = OUT / f"{args.area}_pycrown_preview.png"
    img.save(out)
    print("wrote", out)


def main() -> None:
    wcs_download("dsm_05m", DSM_TIF)
    wcs_download("dtm_05m", DTM_TIF)
    build_chm()
    PC = run_pycrown()
    export_geojson(PC)
    preview(PC)
    print(f"\nDONE {args.area}: {len(PC.trees)} trees delineated")


if __name__ == "__main__":
    main()
