"""Shared helpers for dataset downloaders.

Each downloader produces the same on-disk layout under
``raw_data/<source>/``:

    images/<class_name>/<image_id>.jpg   # ImageFolder convention
    metadata.json                        # consumed by manifest.py

The metadata.json shape is fixed so ``data/manifest.py`` is a pure
consumer that never has to re-walk the directory tree.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SourceMetadata:
    """Stable shape consumed by ``data/manifest.py``.

    Per-image arrays are parallel — ``image_paths[i]`` carries label
    ``labels[i]`` (an index into ``class_names``) and belongs to split
    ``splits[i]``. We keep parallel arrays rather than per-image dicts
    because the manifest builder reads millions of rows and a few
    columns of pure Python lists serialize much faster than 3M dicts.
    """

    source: str
    downloaded_at: str
    image_count: int
    class_names: list[str]
    # Paths are RELATIVE to ``raw_data/<source>/`` — e.g. ``images/Northern_Cardinal/001.jpg``.
    image_paths: list[str]
    labels: list[int]
    # One of "train", "val", "test", "unknown". The unified manifest
    # honors the source-declared split when present; downloaders that
    # don't carry a canonical split (yard data) use "unknown" and the
    # manifest splits by hash.
    splits: list[str]
    # Free-form provenance the model card eventually inlines.
    notes: dict = field(default_factory=dict)


def write_metadata(out_dir: Path, meta: SourceMetadata) -> None:
    """Persist metadata.json. Validates that parallel arrays line up."""
    n = meta.image_count
    assert len(meta.image_paths) == n, f"image_paths len mismatch: {len(meta.image_paths)} vs {n}"
    assert len(meta.labels) == n, f"labels len mismatch: {len(meta.labels)} vs {n}"
    assert len(meta.splits) == n, f"splits len mismatch: {len(meta.splits)} vs {n}"
    assert all(0 <= y < len(meta.class_names) for y in meta.labels), "label index OOR"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(name: str) -> str:
    """Filesystem-safe directory name from a label string."""
    return name.replace(" ", "_").replace("/", "_").replace("'", "")
