"""Swappable model backends.

Imports are lazy on purpose: every backend has heavy optional dependencies
(deepforest, rfdetr, samgeo, timm) and the core package must stay importable
with none of them installed.
"""

from tseg.models.base import Backend, get_backend, list_backends

__all__ = ["Backend", "get_backend", "list_backends"]
