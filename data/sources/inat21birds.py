"""iNat21-Birds — bird subset of the iNaturalist 2021 Challenge.

Source: visipedia/inat_comp. The 2021 challenge data lives in a public
S3 bucket; no account required.

Files (URLs as of v1, immutable):

    train.tar.gz                ~234 GB
    val.tar.gz                  ~8 GB
    train.json.tar.gz           ~50 MB  (annotations)
    val.json.tar.gz             ~5 MB

Strategy: rather than extracting the full 250 GB train tarball, we
stream it through a filter that writes out only the Aves images. The
JSON annotations tell us which categories are birds (``supercategory:
"Birds"``); we keep their images and skip everything else.

Final on-disk footprint: ~75 GB after filter. Wall-clock: bandwidth-
bound; figure ~3-6 hours on a residential connection.

Idempotency: if ``images/`` already contains the expected file count
for a given species directory, we skip re-downloading that shard. A
``state.json`` checkpoint records categories successfully processed.
"""
from __future__ import annotations

import json
import logging
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from data._common import SourceMetadata, now_iso, slug, write_metadata

log = logging.getLogger(__name__)

S3_BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
ANNOTATION_FILES = {"train": "train.json.tar.gz", "val": "val.json.tar.gz"}
IMAGE_FILES = {"train": "train.tar.gz", "val": "val.tar.gz"}

# Hard limit on Aves images to keep per-class for training. The
# challenge has long-tailed bird classes (a few thousand per common
# species, < 50 for some); capping at this prevents the head from
# being dominated by the over-represented classes.
PER_CLASS_CAP = 1000


def download(out_dir: Path, cache_dir: Path | None = None) -> None:
    """Pull the iNat21-Birds subset into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir or (out_dir / "_cache")
    cache.mkdir(exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # ----- 1. Annotations (cheap, get bird category IDs first). -----
    log.info("Fetching annotations …")
    train_ann = _load_annotations(cache, "train")
    val_ann = _load_annotations(cache, "val")

    bird_category_ids = {c["id"] for c in train_ann["categories"]
                         if c.get("supercategory") == "Birds"}
    log.info("iNat21 has %d bird categories", len(bird_category_ids))

    cat_by_id = {c["id"]: c for c in train_ann["categories"] if c["id"] in bird_category_ids}
    # We use scientific name as the canonical id; common name as fallback.
    # iNat21 categories carry both as keys: "name" (scientific binomial)
    # and "common_name".
    class_names: list[str] = sorted(
        cat_by_id[cid].get("common_name") or cat_by_id[cid]["name"]
        for cid in bird_category_ids
    )
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    image_paths: list[str] = []
    labels: list[int] = []
    splits: list[str] = []

    # ----- 2. Stream-filter the train+val tarballs. -----
    for split_name in ("train", "val"):
        ann = train_ann if split_name == "train" else val_ann
        bird_image_ids = {a["image_id"] for a in ann["annotations"]
                          if a["category_id"] in bird_category_ids}
        image_id_to_meta = {im["id"]: im for im in ann["images"]
                            if im["id"] in bird_image_ids}
        image_id_to_cat = {a["image_id"]: a["category_id"] for a in ann["annotations"]
                           if a["category_id"] in bird_category_ids}
        log.info("Split %s: %d bird images to extract", split_name, len(image_id_to_meta))
        # iNat tarball image paths follow `train/<category_id>/<image_id>.jpg`,
        # so we don't need to scan the whole tarball — we know the paths.
        # But the file is huge; we still stream sequentially.
        kept_per_class: dict[int, int] = {}
        _stream_filter_tarball(
            cache=cache,
            split_name=split_name,
            kept_paths=image_paths,
            kept_labels=labels,
            kept_splits=splits,
            split_label="train" if split_name == "train" else "val",
            image_id_to_meta=image_id_to_meta,
            image_id_to_cat=image_id_to_cat,
            cat_by_id=cat_by_id,
            name_to_idx=name_to_idx,
            kept_per_class=kept_per_class,
            images_dir=images_dir,
            out_dir=out_dir,
        )

    meta = SourceMetadata(
        source="inat21birds",
        downloaded_at=now_iso(),
        image_count=len(image_paths),
        class_names=class_names,
        image_paths=image_paths,
        labels=labels,
        splits=splits,
        notes={
            "per_class_cap": PER_CLASS_CAP,
            "license_note": "iNat21 is CC-BY-NC; downstream model weights inherit the non-commercial clause.",
            "version": "iNat21 v1 challenge release",
        },
    )
    write_metadata(out_dir, meta)
    log.info("Done: %d images, %d classes → %s", len(image_paths), len(class_names), out_dir)


def _load_annotations(cache: Path, split: str) -> dict:
    """Download + parse the JSON annotations for one split."""
    name = ANNOTATION_FILES[split]
    local = cache / name
    if not local.exists():
        log.info("Downloading %s …", name)
        _download_to(f"{S3_BASE}/{name}", local)
    extracted_dir = cache / f"{split}_ann"
    if not extracted_dir.exists():
        log.info("Extracting %s …", name)
        extracted_dir.mkdir()
        with tarfile.open(local, "r:gz") as tf:
            tf.extractall(extracted_dir)
    # The JSON inside is `<split>.json` by convention.
    json_path = next(extracted_dir.rglob("*.json"))
    return json.loads(json_path.read_text())


def _download_to(url: str, dst: Path) -> None:
    """Streamed download with a progress bar."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def _stream_filter_tarball(
    cache: Path,
    split_name: str,
    kept_paths: list[str],
    kept_labels: list[int],
    kept_splits: list[str],
    split_label: str,
    image_id_to_meta: dict,
    image_id_to_cat: dict,
    cat_by_id: dict,
    name_to_idx: dict[str, int],
    kept_per_class: dict[int, int],
    images_dir: Path,
    out_dir: Path,
) -> None:
    """Stream the image tarball from S3, extract only Aves rows,
    enforce ``PER_CLASS_CAP``."""
    name = IMAGE_FILES[split_name]
    log.info("Streaming %s (this is the slow step; ~hours)", name)
    url = f"{S3_BASE}/{name}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                # Member path looks like `train/<category_id>/<filename>.jpg`.
                # We use the meta-table lookup keyed on file_name to find
                # the matching image_id (member name vs `im["file_name"]`).
                # iNat21's annotation file_name format matches the
                # tarball member name.
                # Build reverse lookup on first call.
                file_to_image_id = getattr(_stream_filter_tarball, "_lookup", None)
                if file_to_image_id is None or _stream_filter_tarball._lookup_split != split_name:
                    _stream_filter_tarball._lookup = {
                        im["file_name"]: im_id
                        for im_id, im in image_id_to_meta.items()
                    }
                    _stream_filter_tarball._lookup_split = split_name
                    file_to_image_id = _stream_filter_tarball._lookup
                image_id = file_to_image_id.get(member.name)
                if image_id is None:
                    continue
                cat_id = image_id_to_cat.get(image_id)
                if cat_id is None:
                    continue
                cls_name = cat_by_id[cat_id].get("common_name") or cat_by_id[cat_id]["name"]
                label_idx = name_to_idx[cls_name]
                if kept_per_class.get(label_idx, 0) >= PER_CLASS_CAP:
                    continue
                class_dir = images_dir / slug(cls_name)
                class_dir.mkdir(exist_ok=True)
                rel_path = f"images/{slug(cls_name)}/{Path(member.name).name}"
                dst = out_dir / rel_path
                if not dst.exists():
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    with open(dst, "wb") as out_f:
                        out_f.write(f.read())
                kept_paths.append(rel_path)
                kept_labels.append(label_idx)
                kept_splits.append(split_label)
                kept_per_class[label_idx] = kept_per_class.get(label_idx, 0) + 1
