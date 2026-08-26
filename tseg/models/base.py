# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Object Vision B.V. and tseg contributors
"""Backend protocol and registry.

Every backend takes a uint8 HxWx3 image and returns Detections in *pixel*
coordinates local to that image. Georeferencing, shape derivation and dedupe
all happen downstream, so a backend never needs to know about RD or tiles.

Backends that only produce boxes (DeepForest) return mask=None. Everything
downstream tolerates that and falls back to the bbox.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from tseg.records import Detection


@runtime_checkable
class Backend(Protocol):
    name: str

    def load(self, weights: str | None = None, device: str | None = None) -> None:
        """Load weights onto the device. Called once per process."""

    def detect(self, img: np.ndarray) -> list[Detection]:
        """Instance detections in pixel coordinates."""

    def classify(self, chip: np.ndarray) -> tuple[str, float]:
        """Single label + probability for a whole chip."""


class UnsupportedTask(NotImplementedError):
    """Raised when a backend is asked for a task it does not implement."""


def require(module: str, extra: str):
    """Import an optional dependency, or explain how to install it.

    Backends import their heavy dependencies inside load() so the core package
    stays importable without any of them. The cost is that a missing extra
    surfaces as a bare ModuleNotFoundError at load time, which tells a new user
    nothing. This turns it into the install command.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module!r} is not installed, which the {extra!r} backend needs. "
            f"Install it with:  pip install 'tseg[{extra}]'"
        ) from exc


class BaseBackend:
    """Shared default behaviour: unsupported tasks fail loudly and specifically."""

    name = "base"

    def load(self, weights=None, device=None):
        raise NotImplementedError

    def detect(self, img):
        raise UnsupportedTask(
            f"backend {self.name!r} does not do detection; "
            f"use it with 'tseg pand' or pick a detection backend"
        )

    def classify(self, chip):
        raise UnsupportedTask(
            f"backend {self.name!r} does not do classification; "
            f"use it with 'tseg detect' or pick 'classifier'"
        )


_REGISTRY = {
    "deepforest": ("tseg.models.deepforest_backend", "DeepForestBackend"),
    "rfdetr": ("tseg.models.rfdetr_backend", "RFDETRBackend"),
    "sam3": ("tseg.models.sam3_backend", "SAM3Backend"),
    "classifier": ("tseg.models.classifier_backend", "ClassifierBackend"),
}


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str, **kwargs):
    """Instantiate a backend by name, importing its module only now."""
    try:
        module_name, cls_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; available: {list_backends()}"
        ) from None

    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"backend {name!r} needs an optional dependency that is not "
            f"installed: {exc}. Try: pip install -e .[{name}]"
        ) from exc

    return getattr(module, cls_name)(**kwargs)
