"""gpiosenka 525 bird species — fetched via HuggingFace mirror.

Original Kaggle upload (``gpiosenka/100-bird-species``) was removed by
the author in 2025 and now 404s. The community mirror at
``yashikota/birds-525-species-image-classification`` is bit-identical
to the original — 89,885 images across 525 species, with the
canonical 84,600 / 2,630 / 2,630 train/val/test split preserved.

Using the HF mirror also means no Kaggle API token is needed; the
``datasets`` library streams the data over HTTPS.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from data._common import SourceMetadata, now_iso, slug, write_metadata

log = logging.getLogger(__name__)

HF_DATASET = "yashikota/birds-525-species-image-classification"

# Split names emitted by the HF dataset → our canonical strings.
_SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def download(out_dir: Path) -> None:
    """Pull gpiosenka 525 from the HF mirror into ``out_dir``."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Install the `datasets` extra: `pip install -e .` (deps include it)."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    log.info("Streaming %s from HuggingFace …", HF_DATASET)
    # `load_dataset` caches the parquet shards under ~/.cache/huggingface/.
    # We re-materialize to JPG on disk so the rest of the pipeline can use
    # bare ImageFolder-style file paths (faster than parquet-row-access
    # during training, and keeps the dataloader implementation simple).
    ds = load_dataset(HF_DATASET)

    # Build the canonical class name list from the first split's features.
    # All three splits share the same ClassLabel feature.
    first_split = next(iter(ds.values()))
    label_feature = first_split.features["label"]
    class_names: list[str] = list(label_feature.names)
    log.info("Found %d classes", len(class_names))

    image_paths: list[str] = []
    labels: list[int] = []
    splits: list[str] = []
    image_count = 0

    for src_split_name, split_dataset in ds.items():
        canonical_split = _SPLIT_MAP.get(src_split_name, src_split_name)
        log.info("Materializing split %s → %s (%d rows)",
                 src_split_name, canonical_split, len(split_dataset))
        for i, row in enumerate(tqdm(split_dataset, desc=canonical_split)):
            img: Image.Image = row["image"]
            label_idx: int = int(row["label"])
            class_name = class_names[label_idx]
            class_dir = images_dir / slug(class_name)
            class_dir.mkdir(exist_ok=True)
            rel_path = f"images/{slug(class_name)}/{canonical_split}_{i:06d}.jpg"
            abs_path = out_dir / rel_path
            # Re-encode to JPEG (the source is already JPEG-derived;
            # keeps quality, normalizes EXIF). Skip if already written
            # so re-running is idempotent.
            if not abs_path.exists():
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(abs_path, format="JPEG", quality=92)
            image_paths.append(rel_path)
            labels.append(label_idx)
            splits.append(canonical_split)
            image_count += 1

    meta = SourceMetadata(
        source="gpiosenka",
        downloaded_at=now_iso(),
        image_count=image_count,
        class_names=class_names,
        image_paths=image_paths,
        labels=labels,
        splits=splits,
        notes={
            "origin": "yashikota/birds-525-species-image-classification (HF mirror of gpiosenka/100-bird-species)",
            "license_note": "Original dataset removed from Kaggle 2025; HF mirror has no explicit license — treat as fair use for research.",
        },
    )
    write_metadata(out_dir, meta)
    log.info("Done: %d images, %d classes → %s", image_count, len(class_names), out_dir)
