"""NABirds v1 — expert-labeled NA bird imagery from Cornell Lab.

User downloads ``nabirds.tar.gz`` (~3 GB) from
https://dl.allaboutbirds.org/nabirds via the request-access form, then
passes the local path here.

NABirds layout (after extraction):

    nabirds/
      images/<class_id>/<image_id>.jpg
      classes.txt                # class_id → display name (2-level hierarchy)
      hierarchy.txt              # parent_class_id child_class_id
      image_class_labels.txt     # image_id class_id
      images.txt                 # image_id relative_path
      train_test_split.txt       # image_id is_training (1=train, 0=test)

Children of the hierarchy are leaf species (used for training); the
parents are higher-level groupings (e.g. ``Sparrows``) that we drop —
the unified taxonomy needs concrete species, not category labels.
"""
from __future__ import annotations

import logging
import shutil
import tarfile
from pathlib import Path

from tqdm import tqdm

from data._common import SourceMetadata, now_iso, slug, write_metadata

log = logging.getLogger(__name__)


def download(out_dir: Path, tarball: Path) -> None:
    """Extract a local NABirds tarball and emit metadata.json.

    Args:
        out_dir: where to populate ``images/`` and ``metadata.json``.
        tarball: path to the user-downloaded ``nabirds.tar.gz``.
    """
    if not tarball.exists():
        raise FileNotFoundError(f"NABirds tarball not found: {tarball}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract under a tmp dir first so we can inspect the inner layout
    # before promoting to the canonical structure.
    tmp_extract = out_dir / "_extract"
    if tmp_extract.exists():
        log.info("Re-using existing extracted copy at %s", tmp_extract)
    else:
        log.info("Extracting %s → %s (~3 GB, several minutes) …", tarball, tmp_extract)
        tmp_extract.mkdir()
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(tmp_extract)

    # NABirds tarball unpacks as `nabirds/...`. Find it.
    candidates = [p for p in tmp_extract.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"Extracted tarball is empty: {tmp_extract}")
    root = candidates[0]

    classes = _parse_classes(root / "classes.txt")
    parents = _parse_hierarchy(root / "hierarchy.txt")
    images = _parse_images(root / "images.txt")
    image_labels = _parse_image_class_labels(root / "image_class_labels.txt")
    train_split = _parse_train_test_split(root / "train_test_split.txt")

    # Leaf species: any class_id that doesn't appear as a parent.
    parent_ids = set(parents.values())
    leaf_class_ids = sorted(cid for cid in classes if cid not in parent_ids)
    log.info("NABirds: %d total classes, %d leaf species", len(classes), len(leaf_class_ids))

    # Build a leaf-class index for our metadata.
    leaf_idx_by_id = {cid: i for i, cid in enumerate(leaf_class_ids)}
    class_names = [classes[cid] for cid in leaf_class_ids]

    images_out_dir = out_dir / "images"
    images_out_dir.mkdir(exist_ok=True)

    image_paths: list[str] = []
    labels: list[int] = []
    splits: list[str] = []
    n_skipped_nonleaf = 0

    for image_id, src_rel in tqdm(images.items(), desc="indexing images"):
        cls_id = image_labels.get(image_id)
        if cls_id is None or cls_id not in leaf_idx_by_id:
            n_skipped_nonleaf += 1
            continue
        label = leaf_idx_by_id[cls_id]
        class_dir = images_out_dir / slug(classes[cls_id])
        class_dir.mkdir(exist_ok=True)
        dst = class_dir / Path(src_rel).name
        rel_path = f"images/{slug(classes[cls_id])}/{Path(src_rel).name}"
        if not dst.exists():
            src = root / "images" / src_rel
            shutil.copy(src, dst)
        image_paths.append(rel_path)
        labels.append(label)
        # NABirds has no explicit val split — use 90/10 of train split
        # carved off deterministically by image_id hash so re-runs are
        # stable. Test images stay as test.
        if train_split.get(image_id, 0) == 0:
            splits.append("test")
        else:
            # 1-in-10 hash bucket → val.
            splits.append("val" if (hash(image_id) % 10 == 0) else "train")

    log.info("Skipped %d non-leaf-class images", n_skipped_nonleaf)
    log.info("Promoted %d images across %d leaf species", len(image_paths), len(class_names))

    meta = SourceMetadata(
        source="nabirds",
        downloaded_at=now_iso(),
        image_count=len(image_paths),
        class_names=class_names,
        image_paths=image_paths,
        labels=labels,
        splits=splits,
        notes={
            "version": "v1",
            "license_note": "Cornell Lab academic research use — see https://dl.allaboutbirds.org/nabirds",
            "leaf_only": True,
        },
    )
    write_metadata(out_dir, meta)

    # Best-effort cleanup of the temp extract once we've copied everything
    # into the canonical images/ tree. Saves ~3 GB.
    try:
        shutil.rmtree(tmp_extract)
        log.info("Removed temp extract %s", tmp_extract)
    except OSError as e:
        log.warning("Couldn't remove temp extract %s: %s (you can rm it manually)", tmp_extract, e)


def _parse_classes(path: Path) -> dict[int, str]:
    """class_id (int) → display name."""
    out: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        # Format: "<id> <name with spaces>"
        parts = line.split(maxsplit=1)
        out[int(parts[0])] = parts[1] if len(parts) > 1 else ""
    return out


def _parse_hierarchy(path: Path) -> dict[int, int]:
    """child_class_id → parent_class_id. Roots aren't in this map."""
    out: dict[int, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        child, parent = line.split()
        out[int(child)] = int(parent)
    return out


def _parse_images(path: Path) -> dict[str, str]:
    """image_id → relative path under images/."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        image_id, rel_path = line.split()
        out[image_id] = rel_path
    return out


def _parse_image_class_labels(path: Path) -> dict[str, int]:
    """image_id → class_id."""
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        image_id, cls_id = line.split()
        out[image_id] = int(cls_id)
    return out


def _parse_train_test_split(path: Path) -> dict[str, int]:
    """image_id → 1 (train) / 0 (test)."""
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        image_id, flag = line.split()
        out[image_id] = int(flag)
    return out
