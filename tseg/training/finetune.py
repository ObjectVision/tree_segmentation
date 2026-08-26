"""The finetune half of the loop.

    round 0   pre-label with SAM 3 (or DeepForest)  -> store
    round N   review            -> verdicts
              export --coco     -> data/rounds/N/{train,valid}
              train --resume    -> checkpoint from round N-1
              detect --weights  -> store, round N+1

Every round keeps its own checkpoint and metrics.json so a regression is
visible rather than averaged away. The validation split is frozen at round 0
by a hash of each candidate uid, and the review UI never serves holdout rows --
if the loop could relabel its own validation set the metric would drift with
the training data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tseg.models import get_backend
from tseg.review.export import export_chips, export_coco
from tseg.review.store import ReviewStore


def round_dir(out_root, round_no: int) -> Path:
    return Path(out_root) / "rounds" / str(round_no)


def previous_checkpoint(out_root, round_no: int):
    """Best checkpoint of the most recent completed round, if any."""
    for prev in range(round_no - 1, -1, -1):
        for name in ("checkpoint_best_total.pth", "checkpoint_best.pth",
                     "checkpoint.pth", "head.pt"):
            p = round_dir(out_root, prev) / "train" / name
            if p.exists():
                return p
    return None


def run_round(profile, out_root, round_no: int | None = None,
              store_path=None, epochs=None, batch_size=None,
              device_name=None, resume: bool = True,
              allow_incomplete: bool = False):
    """Export the reviewed data and finetune one round on it."""
    out_root = Path(out_root)
    store = ReviewStore(store_path or out_root / "review.db",
                        holdout_fraction=profile.train.val_fraction,
                        seed=profile.train.holdout_seed)
    try:
        round_no = store.max_round() if round_no is None else round_no
        rdir = round_dir(out_root, round_no)
        rdir.mkdir(parents=True, exist_ok=True)

        stats = store.stats()
        if stats["reviewed"] == 0:
            raise SystemExit(
                "nothing reviewed yet -- run 'tseg review' before 'tseg train'"
            )

        if profile.model.backend == "classifier":
            result = _train_classifier(profile, store, rdir, device_name,
                                       epochs, resume, out_root, round_no)
        else:
            result = _train_detector(profile, store, rdir, device_name,
                                     epochs, batch_size, resume, out_root,
                                     round_no, allow_incomplete)

        meta = {
            "round": round_no,
            "profile": profile.name,
            "backend": profile.model.backend,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "review_stats": stats,
            "result": result,
        }
        (rdir / "metrics.json").write_text(json.dumps(meta, indent=2),
                                           encoding="utf-8")
        print(json.dumps(meta, indent=2, default=str))
        return meta
    finally:
        store.close()


def _train_detector(profile, store, rdir, device_name, epochs, batch_size,
                    resume, out_root, round_no, allow_incomplete):
    dataset_dir = rdir / "dataset"
    manifest = export_coco(
        store, profile, dataset_dir, classes=profile.model.classes,
        require_complete_tiles=not allow_incomplete,
    )
    if manifest.get("train", {}).get("images", 0) == 0:
        raise SystemExit(
            "no complete training tiles -- finish reviewing at least one tile, "
            "or pass --allow-incomplete (which teaches the model that "
            "unreviewed objects are background)"
        )

    m = profile.model
    backend = get_backend("rfdetr", resolution=m.resolution,
                          score_thresh=m.score_thresh, nms_iou=m.nms_iou,
                          classes=m.classes)
    ckpt = previous_checkpoint(out_root, round_no) if resume else None
    if ckpt:
        print(f"resuming from {ckpt}")
    backend.load(weights=str(ckpt) if ckpt else None, device=device_name)

    backend.train(
        dataset_dir=dataset_dir,
        output_dir=rdir / "train",
        epochs=epochs or profile.train.epochs,
        batch_size=batch_size or profile.train.batch_size,
        grad_accum=profile.train.grad_accum,
        lr=profile.train.lr,
    )
    return {"dataset": manifest, "resumed_from": str(ckpt) if ckpt else None,
            "output": str(rdir / "train")}


def _train_classifier(profile, store, rdir, device_name, epochs, resume,
                      out_root, round_no):
    import numpy as np
    from PIL import Image

    counts = export_chips(store, rdir / "dataset", classes=profile.model.classes)

    chips, labels = [], []
    for row in store.labelled(holdout=False, source="pand"):
        label = store.training_label(row)
        path = row["chip_path"]
        if label is None or not path or not Path(path).exists():
            continue
        chips.append(np.asarray(Image.open(path).convert("RGB")))
        labels.append(label)

    if len(chips) < 2 * len(profile.model.classes):
        raise SystemExit(
            f"only {len(chips)} labelled chips for {len(profile.model.classes)} "
            f"classes -- review more panden before training"
        )

    backend = get_backend("classifier", backbone=profile.model.weights
                          or "vit_small_patch14_dinov2.lvd142m",
                          classes=profile.model.classes,
                          resolution=profile.model.resolution)
    backend.load(device=device_name)

    ckpt = previous_checkpoint(out_root, round_no) if resume else None
    if ckpt and ckpt.name == "head.pt":
        print(f"resuming head from {ckpt}")
        backend.load_head(ckpt)

    metrics = backend.fit(chips, labels,
                          epochs=epochs or profile.train.epochs,
                          lr=profile.train.lr,
                          val_split=0.0)

    # Score the frozen holdout separately -- never mixed into fit().
    holdout = [(row, store.training_label(row))
               for row in store.labelled(holdout=True, source="pand")]
    holdout = [(r, l) for r, l in holdout
               if l is not None and r["chip_path"] and Path(r["chip_path"]).exists()]
    if holdout:
        hchips = [np.asarray(Image.open(r["chip_path"]).convert("RGB"))
                  for r, _ in holdout]
        preds, _ = backend.classify_batch(hchips)
        truth = [l for _, l in holdout]
        metrics["holdout_accuracy"] = float(
            sum(p == t for p, t in zip(preds, truth)) / len(truth))
        metrics["holdout_n"] = len(truth)
        metrics["holdout_per_class"] = {
            c: {
                "n": sum(1 for t in truth if t == c),
                "recall": (sum(1 for p, t in zip(preds, truth) if t == c and p == t)
                           / max(1, sum(1 for t in truth if t == c))),
            }
            for c in profile.model.classes
        }

    out = rdir / "train"
    out.mkdir(parents=True, exist_ok=True)
    backend.save_head(out / "head.pt")
    return {"chips": counts, "metrics": metrics, "output": str(out / "head.pt")}
