# Model card — tseg

Covers the checkpoints produced by `tseg train`. It is a template: fill the
numbers in from `rounds/<N>/metrics.json` before publishing a release.

## What these models are

| | Trees | Riet |
|---|---|---|
| Task | instance segmentation | per-building classification |
| Architecture | RF-DETR-Seg (DINOv2 backbone) | DINOv2 ViT-S/14, frozen + linear head |
| Base checkpoint | `rfdetr` COCO seg, Apache-2.0 | `vit_small_patch14_dinov2.lvd142m`, Apache-2.0 |
| Input | 25 cm RGB (`Actueel_ortho25`) | 8 cm RGB (`Actueel_orthoHR`), BAG-masked chip |
| Output | mask → rectangle **and** circle, EPSG:28992 | class + probability per pand |

## Licence

**Code:** GPL-3.0-or-later. **Weights:** CC-BY-4.0.

These are deliberately different. Whether copyleft propagates from training
code to model weights is genuinely unsettled and untested in court, so the
weights carry their own explicit grant rather than leaving anyone to infer one.

Both base checkpoints are Apache-2.0, which permits redistributing derivatives
under other terms provided the upstream notice travels along — see `NOTICE`.

## Training data

Derived from PDOK open data, **CC-BY-4.0**, © PDOK / Kadaster / Beeldmateriaal
Nederland. Labels were produced in the `tseg review` loop.

Releases ship **annotations and a regeneration manifest, not imagery**
(`tseg export --no-images`). Rebuild the images from PDOK with
`tseg export --regenerate`; that keeps the imagery licence with PDOK and the
dataset current.

- Region: _fill in_ · Vintage: _fill in_ · Rounds: _fill in_
- Objects labelled: _fill in_ (accepted / rejected / hand-drawn)

## Evaluation

The validation split is **frozen at round 0** by a hash of each candidate's
uid, and the review UI never serves holdout rows. Reported numbers are on that
holdout only.

| Round | Train objects | mAP50 / accuracy | mAP50-95 | Notes |
|---|---|---|---|---|
| 0 | | | | bootstrap |

If a round does not beat its predecessor on the frozen holdout, the loop is not
working. Diagnose it rather than adding rounds.

## Known limitations

**Trees**
- 25 cm is coarse for crowns. The DeepForest baseline was trained on ~10 cm
  NEON imagery, and small or young trees are missed at PDOK resolution. This
  recall gap is the reason the review UI has a hand-drawing mode; accept/reject
  alone can only suppress false positives.
- Merged crowns in dense canopy come back as one mask. Circles use the
  equal-area convention (`r = √(area/π)`) precisely because a minimum-enclosing
  circle overestimates the radius roughly 3× on such a mask.
- Leaf-on imagery only. Deciduous crowns in a leaf-off vintage will not match.

**Riet**
- Severe class imbalance: thatch is a low single-digit percentage of Dutch
  building stock and is geographically clumped. Class weighting is on by
  default; without it the head learns to answer with the majority class and
  reports a flattering accuracy. Read per-class recall, not accuracy.
- Partly-thatched roofs are real; the `partial` class exists for them and will
  be the noisiest.
- Solar panels, moss and shadow all alter roof texture and are plausible
  confusions.
- The footprint comes from BAG. A stale or wrong footprint produces a wrong
  chip, and the model has no way to notice.

**Both**
- Output is model prediction, not an authoritative record. Do not treat it as
  a register.
- Trained on Dutch imagery at Dutch resolutions. Transfer elsewhere is untested.

## Reproducing

```bash
tseg detect --profile trees --codes GM0983 --out output/trees --to-store
tseg review --profile trees --out output/trees
tseg train  --profile trees --out output/trees
```

Exact geometry and imagery settings for any run are recorded in `run.yaml`
beside its output, and `rounds/<N>/metrics.json` records each round.
