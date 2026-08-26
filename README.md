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

## Licence

Code is **GPL-3.0-or-later** (`LICENSE`). Model weights are **CC-BY-4.0** — the
two differ deliberately, because whether copyleft propagates from training code
to weights is untested law and should not be left to inference. See
`MODEL_CARD.md`.

Every core dependency is permissive (Apache-2.0, MIT, BSD), and
`tests/test_licences.py` fails the build if that stops being true.

## Models

| Role | Model | Licence |
|---|---|---|
| Finetune target, trees | **RF-DETR-Seg** (DINOv2 backbone) | Apache-2.0 |
| Finetune target, riet | **DINOv2 ViT-S/14** frozen + linear head (timm) | Apache-2.0 |
| Bootstrap + baseline | **DeepForest**, stock weights | MIT |
| *Optional*, off by default | SAM 3 | ⚠ SAM License — **not open source** |

Two exclusions worth knowing about, so nobody re-litigates them:

**Ultralytics YOLO26 is not used.** Not primarily for licensing — Roboflow's own
benchmark has RF-DETR-Seg ahead of YOLO26 on segmentation, so AGPL-3.0 would buy
no accuracy while making every downstream reuser copyleft.

**RF-DETR *detection* XLarge/2XLarge are not used.** Those two checkpoints are
Platform Model License 1.0, not Apache-2.0. Every *segmentation* checkpoint,
Nano through 2XLarge, is Apache-2.0, and segmentation is what tseg uses.

### About SAM 3

SAM 3 gives excellent zero-shot pre-labels from a text prompt, but Meta's SAM
License carries field-of-use restrictions, so it is **not OSI-approved** and not
GPL-compatible. tseg can call it only under the GPL §7 permission in
`LICENSE-EXCEPTIONS`, it is never installed by default, and it warns on load.

Nothing needs it. The open-source paths are better suited anyway:

- **Trees** — bootstrap with DeepForest (MIT). You already have 6,816 cached
  tiles of its output to seed round 0.
- **Riet** — `tseg mine` ranks unreviewed chips by similarity to confirmed
  positives using frozen DINOv2 embeddings. For finding a rare texture with no
  crisp English name, a handful of confirmed examples describes the target far
  better than the word "thatch" does.

Reach for SAM 3 only for an open-vocabulary class neither path covers.

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

Bootstrap round 0 with DeepForest so the first review pass is triage rather
than a blank canvas:

```bash
tseg detect --profile trees --backend deepforest --codes GM0983 --out output/trees --to-store
```

For a rare class like riet, hunt by similarity instead of paging through
negatives — accept one example, then let embeddings find the rest:

```bash
tseg mine --profile riet --out output/riet --like 0363100012061959 --limit 200
tseg mine --profile riet --out output/riet --label riet --limit 200
```

Ranked candidates are promoted to the front of the review queue.

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

### Attribution travels with the data

PDOK imagery and BAG are CC-BY-4.0, so anything derived from them must carry
the attribution. A line in this README does not travel with a GeoPackage
someone was emailed, so tseg writes it into the outputs: GeoPackage dataset and
layer metadata, a top-level `attribution` member in GeoJSON, and a `NOTICE.txt`
beside every export. Keep it with anything you publish downstream.

### Releasing a dataset

```bash
tseg export --profile trees --out output/trees --no-images   # annotations + manifest
tseg export --profile trees --out output/trees --regenerate  # rebuild the images
```

`--no-images` writes the COCO annotations and a `regenerate.json` naming the
WMS layer, resolution, tile geometry and tile keys — but no image files. PDOK
is CC-BY so the imagery *could* be redistributed; a manifest is simply smaller,
stays current, and leaves the imagery licence with PDOK. `*.jpg` is gitignored
so chips cannot be committed by accident.

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

`detectree_pdok.py` and `pycrown_pdok.py` were **removed** for the open-source
release: `detectree` and `pycrown` are both GPL-3.0, and a copyleft dependency
in the core would change the licence obligations of everyone reusing tseg. Both
scripts still work and are recoverable from git — see `CONTRIBUTING.md`.

Untouched: `deepforest_pdok.py` (MIT-clean, still the quickest single-tile
visual check).

## Layout

```
tseg/
  config.py      profiles, typed
  device.py      cuda | hip | directml | cpu
  attribution.py CC-BY notice, embedded into every output
  runmeta.py     the profile a run actually used
  records.py     Detection -> Feature
  imagery/       WMS GetMap, RD grid, GeoTIFF, resumable cache
  aoi/           gemeente WFS, BAG pand WFS + chips
  models/        deepforest | rfdetr | sam3 | classifier
  geometry/      rect + circle, px<->RD, cross-tile NMS
  io/            streaming GeoPackage / FlatGeobuf / GeoJSON
  review/        SQLite store, Gradio app, COCO export, embedding miner
  training/      round driver
  cli.py
```

WMS, not WMTS, on purpose: arbitrary-bbox `GetMap` is what makes per-pand chips
possible, and a fixed WMTS tile grid cannot frame a building.
