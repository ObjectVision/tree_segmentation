# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""tseg - finetunable aerial-object segmentation over PDOK imagery.

Two operating modes:
  * tile level  (``tseg detect``) - grid an AOI, run a detector on each tile.
  * pand level  (``tseg pand``)   - crop+mask each BAG building, classify it.

Both feed one review store (``tseg review``) and one finetune loop
(``tseg train``).
"""

# MUST precede rasterio on Windows (c10.dll init order): if GDAL/OpenMP DLLs
# load first, torch's c10.dll init fails with WinError 1114. Importing torch
# here fixes the order for every module in the package. Guarded because the
# core (geometry, io, imagery) is usable without a torch install.
try:  # noqa: SIM105
    import torch  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"

__all__ = ["__version__"]
