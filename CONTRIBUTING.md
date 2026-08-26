# Contributing to tseg

## Setup

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
pytest -m "not network"
```

`torch` is installed first on purpose. On Windows, if GDAL or OpenMP DLLs load
before it, `torch`'s `c10.dll` fails with `WinError 1114`. `tseg/__init__.py`
imports `torch` before anything else for the same reason — do not move it.

For GPU work see the README: AMD needs a separate Python **3.12** environment,
because AMD's Windows ROCm wheel is published only for ROCm 7.2.1 / PyTorch 2.9
/ Python 3.12.

## Tests

`pytest -m "not network"` is what CI runs. Anything touching PDOK must be
marked `@pytest.mark.network` so CI does not depend on a third-party service.

Tests use synthetic geometry, not fixtures downloaded at runtime. Keep it that
way — a test suite that needs the internet stops being run.

## Licensing rules

tseg is **GPL-3.0-or-later**. Two rules, both enforced by
`tests/test_licences.py`:

1. **Core dependencies stay permissive** (Apache-2.0 / MIT / BSD). This project
   once shipped a GPL-3.0 dependency without anyone noticing, which is why the
   sweep exists. A new copyleft dependency is a decision — declare it in
   `KNOWN_COPYLEFT` with a reason, do not let it drift in.
2. **Nothing non-OSI becomes required.** SAM 3 is the only such component; it
   lives in the `sam3` extra under the GPL §7 permission in
   `LICENSE-EXCEPTIONS`, and nothing depends on it.

Every file under `tseg/` needs an SPDX header. Ported code keeps its upstream
copyright notice — see the `urban-tree` (MIT) block in
`tseg/geometry/shapes.py` for the pattern.

Adding a model backend? Check its **checkpoint** licence, not just its package
licence. They differ more often than you would expect: RF-DETR's package is
Apache-2.0, and so are all its segmentation checkpoints, but its *detection*
XLarge and 2XLarge weights are under Platform Model License 1.0.

## Data

PDOK imagery is CC-BY-4.0. Attribution has to travel with the data, so
`tseg/attribution.py` writes it into GeoPackage metadata, into the GeoJSON
document, and as `NOTICE.txt` beside every export. If you add an output format,
add the attribution too.

**Never commit imagery.** `*.jpg` and `data/rounds/` are gitignored. Releases
ship annotations plus a regeneration manifest (`tseg export --no-images`).

## Adding a backend

Implement the protocol in `tseg/models/base.py` — `load`, then `detect` and/or
`classify` — returning `Detection` objects in **pixel** coordinates local to
the image. Georeferencing, shape derivation and deduplication all happen
downstream, so a backend never needs to know about RD or tiles. Register it in
`_REGISTRY` and import its heavy dependencies inside methods, so the core stays
importable without them.

## Removed code

`detectree_pdok.py` and `pycrown_pdok.py` were deleted before the open-source
release: both depend on GPL-3.0 packages (`detectree`, `pycrown`), which would
have changed the licence obligations for everyone reusing tseg. They still work
and are recoverable:

```bash
git show a362307:detectree_pdok.py > detectree_pdok.py
git show a362307:pycrown_pdok.py   > pycrown_pdok.py
git clone https://github.com/manaakiwhenua/pycrown.git   # GPL-3.0
git clone https://github.com/easz/urban-tree.git         # MIT
```

Both clone directories are gitignored so they cannot re-enter the tree.
