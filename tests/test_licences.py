# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Licence sweep.

This project shipped a GPL-3.0 dependency (detectree) and a vendored GPL-3.0
tree (pycrown) without anyone noticing, because nothing checked. The rule this
encodes: every runtime dependency must be permissively licensed, and the one
non-OSI component (SAM 3) must stay optional.

Run it in CI, not from memory.
"""

import importlib.metadata as md
import pathlib
import re

import pytest

# Permissive licences that can be combined into a GPL-3.0 work without
# imposing further obligations on anyone downstream.
PERMISSIVE = re.compile(
    r"apache|\bmit\b|bsd|isc|python software foundation|psf|"
    r"historical permission|zlib|unlicense|mozilla public license 2",
    re.I,
)
# Copyleft is not automatically wrong for a GPL-3.0 project, but a *new* one
# appearing is a decision, not an accident - so it must be declared here.
KNOWN_COPYLEFT: set[str] = set()

CORE = ["numpy", "pillow", "rasterio", "shapely", "pyproj", "geopandas",
        "pyogrio", "opencv-python", "pyyaml", "tqdm"]

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _licence_of(dist_name: str) -> str:
    """Read every place a licence can hide.

    Modern packages declare PEP 639 License-Expression and leave the legacy
    License field empty. Reading only the old field silently skipped half the
    core dependencies -- a sweep that passes by not looking is worse than none.
    """
    meta = md.metadata(dist_name)
    parts = [
        meta.get("License-Expression") or "",
        meta.get("License") or "",
    ]
    parts += [c for c in (meta.get_all("Classifier") or []) if "License" in c]
    return " ".join(p for p in parts if p).strip()


@pytest.mark.parametrize("dist", CORE)
def test_core_dependency_is_permissive(dist):
    try:
        licence = _licence_of(dist)
    except md.PackageNotFoundError:
        pytest.skip(f"{dist} not installed in this environment")

    assert licence, (
        f"{dist} declares no licence metadata anywhere (License-Expression, "
        f"License, or classifiers). Verify it by hand and record the result."
    )
    if dist in KNOWN_COPYLEFT:
        return

    assert PERMISSIVE.search(licence), (
        f"{dist} is licensed {licence!r}, which is not on the permissive list. "
        f"If this is deliberate, add it to KNOWN_COPYLEFT with a reason."
    )


def test_no_gpl_dependency_sneaks_into_core():
    """detectree and pycrown are the specific mistake this guards against."""
    for dist in CORE:
        try:
            licence = _licence_of(dist)
        except md.PackageNotFoundError:
            continue
        assert "GNU" not in licence and "GPL" not in licence.upper(), (
            f"{dist} is {licence!r}; a copyleft core dependency changes the "
            f"licence obligations of everyone who reuses tseg"
        )


def test_sam3_is_not_a_core_dependency():
    """SAM 3 is not OSI-approved. It must stay in the optional extra."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core_block = pyproject.split("[project.optional-dependencies]")[0]

    for name in ("sam3", "segment-geospatial", "segment_anything"):
        assert name not in core_block, (
            f"{name} appears in the core dependencies; it must remain in the "
            f"optional 'sam3' extra (see LICENSE-EXCEPTIONS)"
        )


def test_removed_gpl_probes_stay_removed():
    for gone in ("detectree_pdok.py", "pycrown_pdok.py"):
        assert not (ROOT / gone).exists(), (
            f"{gone} was deleted because it depends on GPL-3.0 code; "
            f"reintroducing it changes the project licence story"
        )
    for gone in ("pycrown", "urban-tree"):
        assert not (ROOT / gone / ".git").exists(), (
            f"vendored {gone}/ is back in the tree; keep it out (see .gitignore)"
        )


def test_every_module_declares_its_licence():
    missing = [
        str(f.relative_to(ROOT))
        for f in sorted((ROOT / "tseg").rglob("*.py"))
        if "SPDX-License-Identifier" not in f.read_text(encoding="utf-8")
    ]
    assert not missing, f"missing SPDX header: {missing}"


def test_ported_code_keeps_its_upstream_notice():
    """MIT requires the notice to travel with the code."""
    shapes = (ROOT / "tseg" / "geometry" / "shapes.py").read_text(encoding="utf-8")
    assert "Copyright (c) 2022 easz@github" in shapes
    assert "MIT License" in shapes


def test_licence_files_are_present():
    for name in ("LICENSE", "LICENSE-EXCEPTIONS", "NOTICE"):
        assert (ROOT / name).exists(), f"{name} is missing"
    assert "PDOK" in (ROOT / "NOTICE").read_text(encoding="utf-8")
