# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Single source of truth for which compute device we run on.

The repo has to span two environments:

  * ``.venv``     Python 3.14 / torch 2.12+cpu  - the original CPU pipeline.
  * ``.venv-gpu`` Python 3.12 / torch 2.9 ROCm 7.2.1 - the RX 9060 XT (gfx1200).

ROCm reports itself through the ``torch.cuda`` namespace, so ``is_available()``
is True on both NVIDIA and AMD; ``torch.version.hip`` is what tells them apart.
"""

from __future__ import annotations

import os
from functools import lru_cache

_VALID = ("cuda", "hip", "directml", "cpu")


@lru_cache(maxsize=1)
def resolve(prefer: str | None = None) -> str:
    """Return one of ``cuda`` | ``hip`` | ``directml`` | ``cpu``.

    ``TSEG_DEVICE`` overrides autodetection; ``prefer`` overrides both when it
    is actually available (otherwise we fall through and warn on use).
    """
    forced = prefer or os.environ.get("TSEG_DEVICE")
    if forced:
        forced = forced.lower()
        if forced not in _VALID:
            raise ValueError(f"device must be one of {_VALID}, got {forced!r}")
        return forced

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        # ROCm builds expose HIP under the cuda namespace.
        return "hip" if getattr(torch.version, "hip", None) else "cuda"

    try:
        import torch_directml  # noqa: F401

        if torch_directml.is_available():
            return "directml"
    except ImportError:
        pass

    return "cpu"


def torch_device(name: str | None = None):
    """Map our device name onto a real ``torch.device``."""
    import torch

    name = name or resolve()
    if name == "directml":
        import torch_directml

        return torch_directml.device()
    # Both cuda and hip address the GPU as "cuda" in torch's API.
    return torch.device("cuda" if name in ("cuda", "hip") else "cpu")


def describe() -> str:
    """Human-readable one-liner for logs and the ``tseg info`` command."""
    name = resolve()
    try:
        import torch
    except ImportError:
        return f"{name} (torch not installed)"

    bits = [f"torch {torch.__version__}"]
    if name in ("cuda", "hip"):
        try:
            bits.append(torch.cuda.get_device_name(0))
            free, total = torch.cuda.mem_get_info()
            bits.append(f"{total / 1024**3:.1f} GB VRAM")
        except Exception as exc:  # pragma: no cover - driver-dependent
            bits.append(f"<device query failed: {exc}>")
    if getattr(torch.version, "hip", None):
        bits.append(f"HIP {torch.version.hip}")
    return f"{name} (" + ", ".join(bits) + ")"
