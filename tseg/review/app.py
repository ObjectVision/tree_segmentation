# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Gradio triage UI -- the visual half of the feedback loop.

Two tabs, because accept/reject alone cannot teach a model to find things it
never proposed:

  Triage        a contact sheet of candidates, default-accept. Untick the wrong
                ones and submit. Ordered by uncertainty, so the labels you give
                are the ones that move the model most.
  Add missing   paint over objects the model failed to detect. Those become
                hand-drawn positives, which is the only thing that raises
                recall -- and recall is exactly the known weakness of the
                25 cm baseline.

Run with:  tseg review --profile trees --out output/trees
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tseg.imagery.grid import Tile
from tseg.imagery.raster import bounds_transform
from tseg.imagery.wms import WMSClient
from tseg.geometry.georef import affine_polygon
from tseg.geometry.shapes import derive_shapes
from tseg.records import Feature
from tseg.review.render import TileImages, chip_for, contact_sheet, paint_to_masks
from tseg.review.store import ACCEPT, REJECT, RELABEL, ReviewStore

KEYS_JS = """
() => {
  if (window.__tsegKeys) return;
  window.__tsegKeys = true;
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const click = (id) => { const b = document.getElementById(id); if (b) b.click(); };
    if (e.key === 'Enter') { click('tseg-submit'); e.preventDefault(); }
    else if (e.key === 'a') { click('tseg-all'); }
    else if (e.key === 'r') { click('tseg-none'); }
    else if (e.key === 'u') { click('tseg-undo'); }
    else if (/^[0-9]$/.test(e.key)) {
      const boxes = document.querySelectorAll('#tseg-picks input[type=checkbox]');
      const idx = (e.key === '0' ? 9 : parseInt(e.key) - 1);
      if (boxes[idx]) boxes[idx].click();
    }
  });
}
"""


def build_app(profile, out_root, store_path=None):
    import gradio as gr

    out_root = Path(out_root)
    store = ReviewStore(store_path or out_root / "review.db",
                        holdout_fraction=profile.train.val_fraction,
                        seed=profile.train.holdout_seed)
    wms = WMSClient(profile.imagery)
    tiles = TileImages(profile, wms)

    rows, cols = profile.review.rows, profile.review.cols
    page_size = rows * cols
    classes = list(profile.model.classes)

    state = {"page": [], "last": []}

    def _chip(row):
        """Prefer the chip already written to disk.

        Pand candidates come with their own crop; re-deriving one from the tile
        would refetch a whole 500 m cell at 8 cm (over 6000 px square) just to
        show one roof.
        """
        path = row["chip_path"]
        if path and Path(path).exists():
            from PIL import Image

            img = Image.open(path).convert("RGB")
            side = profile.review.chip_px
            img.thumbnail((side, side), Image.BILINEAR)
            return img
        return chip_for(store.to_feature(row), tiles, profile.review.chip_px)

    # ------------------------------------------------------------- triage
    def load_page():
        batch = store.batch(limit=page_size, sort=profile.review.sort)
        state["page"] = batch
        if not batch:
            blank = np.full((240, 720, 3), 24, dtype=np.uint8)
            return blank, gr.update(choices=[], value=[]), _stats_md(), ""

        chips, labels = [], []
        for i, row in enumerate(batch):
            chips.append(_chip(row))
            labels.append(f"{i + 1}. {row['pred_label']} {row['score']:.2f}")

        sheet = contact_sheet(chips, rows, cols, profile.review.chip_px)
        return (np.asarray(sheet),
                gr.update(choices=labels, value=labels),
                _stats_md(),
                f"page of {len(batch)} - unticked cells are rejected")

    def _stats_md():
        s = store.stats()
        pct = (100.0 * s["reviewed"] / s["total"]) if s["total"] else 0.0
        return (f"**round {store.max_round()}** - "
                f"{s['reviewed']}/{s['total']} reviewed ({pct:.0f}%) - "
                f"accept {s['accept']} / reject {s['reject']} / "
                f"relabel {s['relabel']} - holdout {s['holdout']} (never shown)")

    def submit(picked, relabel_to):
        batch = state["page"]
        if not batch:
            return load_page()

        kept = {int(p.split(".", 1)[0]) - 1 for p in (picked or [])}
        pairs, ids = [], []
        for i, row in enumerate(batch):
            if i in kept:
                if relabel_to and relabel_to != row["pred_label"]:
                    pairs.append((row["id"], RELABEL, relabel_to))
                else:
                    pairs.append((row["id"], ACCEPT, row["pred_label"]))
            else:
                pairs.append((row["id"], REJECT, None))
            ids.append(row["id"])
        store.set_verdicts(pairs)
        state["last"] = ids
        return load_page()

    def undo():
        for row_id in state["last"]:
            store.undo(row_id)
        state["last"] = []
        return load_page()

    # -------------------------------------------------- add missing objects
    def load_tile(tile_key):
        tile_key = (tile_key or "").strip()
        if not tile_key:
            return None, "give a tile key, e.g. 173000_316000"
        try:
            img, _ = tiles.get(tile_key)
        except Exception as exc:
            return None, f"could not load {tile_key}: {exc}"
        return img, f"{tile_key} loaded - paint over objects the model missed"

    def save_painted(editor_value, tile_key, label, round_no):
        tile_key = (tile_key or "").strip()
        if not editor_value or not tile_key:
            return "nothing to save"

        layers = editor_value.get("layers") or []
        if not layers:
            return "no paint layer found - draw with the brush first"

        painted = np.zeros(np.asarray(layers[0]).shape[:2], dtype=bool)
        for layer in layers:
            for m in paint_to_masks(layer):
                painted |= m

        masks = paint_to_masks(painted.astype(np.uint8)[:, :, None])
        if not masks:
            return "no strokes large enough to keep"

        x, y = (float(v) for v in tile_key.split("_"))
        tile = Tile(x, y, profile.grid.tile_m, profile.grid.overlap_m)
        xmin, ymin, xmax, ymax = tile.padded
        h, w = painted.shape
        transform = bounds_transform(xmin, ymin, xmax, ymax, w, h)

        n = 0
        for m in masks:
            ys, xs = np.nonzero(m)
            bbox = (float(xs.min()), float(ys.min()),
                    float(xs.max()) + 1, float(ys.max()) + 1)
            px = derive_shapes(m, bbox, profile.shapes.circle_method)
            geoms = {k: affine_polygon(px[k], transform)
                     for k in ("mask", "bbox", "rect", "circle")}
            primary = geoms["mask"] or geoms["bbox"]
            if primary is None:
                continue
            c = primary.centroid
            _, _, r_px = px["circle_params"]
            store.add_manual(Feature(
                label=label, score=1.0, backend="human", tile_key=tile_key,
                mask=geoms["mask"], bbox=geoms["bbox"], rect=geoms["rect"],
                circle=geoms["circle"], cx=float(c.x), cy=float(c.y),
                radius_m=float(r_px * profile.imagery.res),
                area_m2=float(px["area_px"] * profile.imagery.res ** 2),
            ), true_label=label, round_no=int(round_no))
            n += 1
        return f"saved {n} hand-drawn {label} object(s) on {tile_key}"

    # ------------------------------------------------------------------ ui
    with gr.Blocks(title=f"tseg review - {profile.name}") as demo:
        gr.Markdown(f"## tseg review - profile `{profile.name}` - "
                    f"backend `{profile.model.backend}`")
        stats = gr.Markdown(_stats_md())

        with gr.Tab("Triage"):
            gr.Markdown(
                "Cells are **accepted by default**. Untick the wrong ones and "
                "submit. Keys: `1-9` toggle - `a` all - `r` none - "
                "`u` undo page - `Enter` submit."
            )
            sheet = gr.Image(label="candidates", height=640,
                             show_download_button=False, interactive=False)
            picks = gr.CheckboxGroup(label="keep", choices=[], value=[],
                                     elem_id="tseg-picks")
            with gr.Row():
                relabel = gr.Dropdown(label="relabel kept cells as",
                                      choices=[""] + classes, value="")
                btn_all = gr.Button("all", elem_id="tseg-all")
                btn_none = gr.Button("none", elem_id="tseg-none")
                btn_undo = gr.Button("undo page", elem_id="tseg-undo")
                btn_submit = gr.Button("submit + next", variant="primary",
                                       elem_id="tseg-submit")
            note = gr.Markdown("")

            btn_submit.click(submit, [picks, relabel], [sheet, picks, stats, note])
            btn_undo.click(undo, None, [sheet, picks, stats, note])
            btn_all.click(lambda c: gr.update(value=c), [picks], [picks])
            btn_none.click(lambda: gr.update(value=[]), None, [picks])
            demo.load(load_page, None, [sheet, picks, stats, note])
            demo.load(None, None, None, js=KEYS_JS)

        with gr.Tab("Add missing"):
            gr.Markdown(
                "Paint over objects the detector **missed**. Each connected "
                "stroke becomes one hand-drawn positive. This is the only part "
                "of the loop that can improve recall."
            )
            with gr.Row():
                tile_key = gr.Textbox(label="tile key", placeholder="173000_316000")
                miss_label = gr.Dropdown(label="label", choices=classes,
                                         value=classes[0] if classes else None)
                miss_round = gr.Number(label="round", value=store.max_round(),
                                       precision=0)
                btn_load = gr.Button("load tile")
            editor = gr.ImageEditor(label="paint missed objects", height=640,
                                    brush=gr.Brush(colors=["#00ff88"],
                                                   default_size=12))
            btn_save = gr.Button("save painted objects", variant="primary")
            miss_note = gr.Markdown("")

            btn_load.click(load_tile, [tile_key], [editor, miss_note])
            btn_save.click(save_painted, [editor, tile_key, miss_label, miss_round],
                           [miss_note])

    return demo, store


def launch(profile, out_root, store_path=None, share=False, port=None):
    demo, store = build_app(profile, out_root, store_path)
    try:
        demo.launch(share=share, server_port=port, inbrowser=True)
    finally:
        store.close()
