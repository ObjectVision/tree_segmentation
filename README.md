# tseg

Finetunable aerial-object segmentation over PDOK imagery.

Two things run on the same machinery:

* **trees** — detect crowns in the 25 cm luchtfoto, tile by tile.
* **riet** — classify roof material per BAG pand, from the 8 cm high-res ortho.

Both emit a **rectangle and a circle** for every object, feed one review store,
and finetune through one loop.

```
detect / pand  ->  review  ->  export  ->  train  ->  detect ...
```

## Install

The repo spans two environments on purpose.

| venv | Python | torch | for |
|---|---|---|---|
| `.venv` | 3.14 | 2.12 **+cpu** | the original CPU pipeline, unchanged |
| `.venv-gpu` | **3.12** | **2.9 + ROCm 7.2.1** | GPU inference and all training |

The Python 3.12 pin is not a preference. The RX 9060 XT is gfx1200 (RDNA4) and
AMD's Windows ROCm wheel is published only for ROCm 7.2.1 / PyTorch 2.9 /
Python 3.12. Python 3.14 has no ROCm build.

```powershell
py -3.12 -m venv .venv-gpu
.venv-gpu\Scripts\activate
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch torchvision
pip install -e ".[rfdetr,sam3,classifier,review,train]"
```

Check it landed on the GPU — ROCm reports through the `cuda` namespace, so
`torch.version.hip` is what distinguishes it:

```
tseg info
# device: hip (torch 2.9.x, AMD Radeon RX 9060 XT, 16.0 GB VRAM, HIP 7.2.1)
```

CPU-only setup, for the existing pipeline:

```powershell
.venv\Scripts\activate
pip install -e ".[deepforest]"
```

If the Windows ROCm preview misbehaves, WSL2 with the Linux ROCm wheels is the
fallback. No code changes — `tseg.device.resolve()` returns `hip` either way.

## Models

| Role | Model | Licence |
|---|---|---|
| Finetune target, trees | **RF-DETR-Seg** (DINOv2 backbone) | Apache-2.0 |
| Finetune target, riet | **DINOv2 ViT-S/14 frozen + linear head** (timm) | Apache-2.0 |
| Cold-start labels | **SAM 3**, text-prompted, inference only | SAM License |
| Baseline | **DeepForest**, stock weights, frozen | MIT |

Ultralytics YOLO26 is deliberately not used: AGPL-3.0 would require
open-sourcing the derivative work or buying an Enterprise Licence.

SAM 3 ships under Meta's custom SAM License, not an OSI licence. It permits
commercial use with restrictions and is used here only to pre-label your own
imagery — but have that reading confirmed before it reaches production.

**Why a classifier for riet and a detector for trees.** BAG already gives the
footprint, so for roofs the location is known and only the material is in
question — that is classification. A chip label is one keypress; a segmentation
label is a traced polygon. Since no pretrained thatch model exists anywhere,
labelling is the bulk of the work, and that ratio decides the project.

## Use

```bash
tseg info                                     # device, backends, profiles

# trees, tile level
tseg detect --profile trees --codes GM0983 --out output/trees --to-store
tseg review --profile trees --out output/trees
tseg train  --profile trees --out output/trees
tseg merge  --profile trees --out output/trees --format gpkg

# riet, BAG pand level
tseg pand   --profile riet --codes GM0983 --out output/riet --limit 500 --to-store
tseg review --profile riet --out output/riet
tseg train  --profile riet --out output/riet
```

Bootstrap round 0 with SAM 3 so the first review pass is triage rather than a
blank canvas:

```bash
tseg detect --profile trees --backend sam3 --codes GM0983 --out output/trees --to-store
```

## Output

Every detection carries both shapes, as separate layers in one GeoPackage:

| layer | geometry |
|---|---|
| `<profile>_circle` | equal-area circle, `r = sqrt(area/pi)` at the centroid |
| `<profile>_rect` | rotated `minAreaRect` (degenerates to the box for box-only backends) |
| `<profile>_bbox` | axis-aligned extent |
| `<profile>_mask` | the mask polygon, when the backend produces one |

Circles default to **equal-area**, not minimum-enclosing: an enclosing circle
badly overestimates when two crowns merge into one mask, which is a routine
failure in dense canopy. `shapes.circle_method: min_enclosing` switches it.

`--format geojson` is kept for the GeoDMS FSS store that reads
`output/limburg_gemeenten_deepforest/limburg_gemeenten_trees.geojson`.

## The review loop

`tseg review` opens a local Gradio app with two tabs.

**Triage** — a contact sheet of candidates, **accepted by default**. Untick the
wrong ones, submit, next page. Keys: `1`–`9` toggle, `a` all, `r` none,
`u` undo page, `Enter` submit. Ordered by **uncertainty** (`|score − 0.5|`
ascending), not by confidence: reviewing what the model already gets right
teaches it almost nothing.

**Add missing** — paint over objects the detector missed. Each connected stroke
becomes a hand-drawn positive. This tab is not optional. Accept/reject can only
ever suppress false positives; without hand-drawn positives the model never
learns about objects it failed to propose, and recall plateaus — and recall is
exactly the known weakness of DeepForest at 25 cm.

Rejects are stored as hard negatives, never deleted.

**The holdout is frozen.** Each candidate is assigned to train or validation by
a hash of its id at insert time, and the review UI never serves holdout rows.
If the loop could relabel its own validation set, the metric would drift with
the training data and stop meaning anything. If round N+1 does not beat round N
on that holdout, the loop is not working — diagnose it rather than adding
rounds.

A tile is only exported once **every** candidate in it has a verdict.
Half-reviewed tiles would hand real objects to the trainer as background, which
is worse than not training. `--allow-incomplete` overrides this; it is
virtually never the right call.

## Migration

`deepforest_province.py` is now a shim over `tseg` with its flags, cache layout
and output path unchanged. Its defaults (`--overlap 0`, `--dedupe 0`) reproduce
the original behaviour exactly. The 6816 pre-tseg `.geojson` cache entries are
read as finished work, never recomputed.

Two defects were fixed in passing:

* **Boundary duplicates.** The old 500 m tiles did not overlap, so a crown on a
  seam was detected twice — truncated in each tile — and the centroid-in-tile
  test kept both halves. `--overlap 25 --dedupe 0.4` fixes it.
* **The 758 MB merge.** The old merge accumulated all 6816 tiles in one Python
  list before a single `json.dumps`. Output is now streamed in bounded batches.

Untouched: `pycrown_pdok.py`, `detectree_pdok.py`, `deepforest_pdok.py`,
`pycrown/`, `urban-tree/`.

## Layout

```
tseg/
  config.py      profiles, typed
  device.py      cuda | hip | directml | cpu
  runmeta.py     the profile a run actually used
  records.py     Detection -> Feature
  imagery/       WMS GetMap, RD grid, GeoTIFF, resumable cache
  aoi/           gemeente WFS, BAG pand WFS + chips
  models/        deepforest | rfdetr | sam3 | classifier
  geometry/      rect + circle, px<->RD, cross-tile NMS
  io/            streaming GeoPackage / FlatGeobuf / GeoJSON
  review/        SQLite store, Gradio app, COCO export
  training/      round driver
  cli.py
```

WMS, not WMTS, on purpose: arbitrary-bbox `GetMap` is what makes per-pand chips
possible, and a fixed WMTS tile grid cannot frame a building.
