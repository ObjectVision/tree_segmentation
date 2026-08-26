# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""CC-BY attribution that travels with the data.

PDOK imagery and BAG footprints are CC-BY 4.0. That licence obliges whoever
redistributes a derived work to carry the attribution, and a line in the README
does not travel with a GeoPackage someone was handed over email. So the notice
is written into the outputs themselves: dataset and layer metadata inside the
file, and a NOTICE.txt beside every export directory.
"""

from __future__ import annotations

from pathlib import Path

SOURCE = "PDOK / Kadaster / Beeldmateriaal Nederland"
LICENSE = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

SHORT = f"(c) {SOURCE}, {LICENSE}"

TEXT = f"""\
Data attribution
================

This dataset is derived from open data published by PDOK
(Publieke Dienstverlening Op de Kaart, https://www.pdok.nl):

    Luchtfoto Beeldmateriaal (aerial imagery)
    Basisregistratie Adressen en Gebouwen (BAG) - building footprints
    Kadaster brk-bestuurlijke-gebieden - administrative boundaries

    (c) {SOURCE}
    Licensed under Creative Commons Attribution 4.0 International
    {LICENSE_URL}

Detections and classifications in this dataset were produced by tseg
(GPL-3.0-or-later). They are model output, not authoritative records, and
carry no guarantee of accuracy. Verify before relying on them.

If you redistribute this dataset or anything derived from it, keep this
attribution with it - CC-BY requires it.
"""


def metadata(profile=None, extra: dict | None = None) -> dict:
    """Key/value pairs to embed in GeoPackage dataset and layer metadata."""
    md = {
        "SOURCE": SOURCE,
        "LICENSE": LICENSE,
        "LICENSE_URL": LICENSE_URL,
        "ATTRIBUTION": SHORT,
        "PRODUCED_BY": "tseg (GPL-3.0-or-later)",
        "DISCLAIMER": "Model output, not an authoritative record.",
    }
    if profile is not None:
        md["TSEG_PROFILE"] = str(profile.name)
        md["TSEG_BACKEND"] = str(profile.model.backend)
        md["TSEG_LAYER"] = str(profile.imagery.layer)
        md["TSEG_RES_M"] = str(profile.imagery.res)
    if extra:
        md.update({k: str(v) for k, v in extra.items()})
    return md


def write_notice(directory) -> Path:
    """Drop NOTICE.txt next to an export so the attribution ships with it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "NOTICE.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path
