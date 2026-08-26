"""Resumable per-tile result cache.

Same contract as the original: one file per tile, named by its RD grid corner,
so a run can be killed and restarted and never redoes finished work. What
changed is the payload -- Feature JSON (mask/bbox/rect/circle as WKT) instead
of a bare GeoJSON FeatureCollection with a lone score property.

An optional imagery cache is new. The original refetched the JPEG every time a
tile was reprocessed, which made iterating on thresholds needlessly slow and
hammered PDOK.
"""

from __future__ import annotations

import json
from pathlib import Path

from tseg import records
from tseg.records import Feature


class TileCache:
    def __init__(self, root, cache_imagery: bool = False):
        self.root = Path(root)
        self.results = self.root / "tiles"
        self.results.mkdir(parents=True, exist_ok=True)
        self.cache_imagery = cache_imagery
        self.images = self.root / "imagery"
        if cache_imagery:
            self.images.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- results
    def path(self, key: str) -> Path:
        return self.results / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def read(self, key: str) -> list[Feature]:
        p = self.path(key)
        if not p.exists():
            return []
        return records.loads(p.read_text(encoding="utf-8"))

    def write(self, key: str, feats) -> None:
        # Write to a temp file and replace, so a crash mid-write cannot leave a
        # truncated cache entry that later reads as "done".
        p = self.path(key)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(records.dumps(list(feats)), encoding="utf-8")
        tmp.replace(p)

    def keys(self):
        return sorted(p.stem for p in self.results.glob("*.json"))

    def iter_all(self):
        """Stream every cached feature. Never builds one big list -- the
        original merge accumulated all 6816 tiles in RAM before serialising."""
        for p in sorted(self.results.glob("*.json")):
            for d in json.loads(p.read_text(encoding="utf-8")):
                yield Feature.from_json(d)

    # -------------------------------------------------------------- imagery
    def image_path(self, key: str, ext: str = "tif") -> Path:
        return self.images / f"{key}.{ext}"

    def has_image(self, key: str) -> bool:
        return self.cache_imagery and self.image_path(key).exists()
