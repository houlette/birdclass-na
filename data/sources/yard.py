"""BirdWatcher yard data — labeled crops from the production deployment.

Pulls two things:

1. The crop JPEGs themselves, via rsync from the BirdWatcher VM at
   ``ryan@birdwatcher.ryanhoulette.com:BirdWatcher/backend/data/crops/``.
2. The labels, by running a tiny SQLAlchemy query inside the container
   over SSH (the DB isn't externally accessible). Emits a CSV with one
   row per detection: ``(crop_path, species, source_tier)``.

Trust tiers (matches BirdWatcher's Correction.source taxonomy):

- ``gold``: user-verified — Correction.source ∈ {NULL, user-confirmed,
  llm-claude-confirmed}.
- ``high``: Claude HIGH auto-committed, unreviewed.
- ``medium``: Claude MEDIUM, awaiting user review.

For the unified-classifier training set we include ``gold`` and
``high`` only. ``medium`` is too noisy at the species level for
training even if the binary (bird/not-bird) classifier could tolerate
it. Tier choice is configurable via ``--tiers``.

All labels are filtered through the canonical taxonomy so non-mappable
labels (e.g. family-level catch-alls like ``Sparrow``) become
``OTHER`` — they still contribute "is a bird" signal but don't dilute
the species classifier head.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from data._common import SourceMetadata, now_iso, slug, write_metadata

log = logging.getLogger(__name__)

# Trust-tier sets used by the Correction.source filter below. Mirrors
# the GOLD/HIGH/MED definitions used in BirdWatcher/backend/scripts/
# train/export_classifier_dataset.py so the two stay in sync.
TIER_TO_SOURCES = {
    "gold": (None, "user-confirmed", "llm-claude-confirmed"),
    "high": ("llm-claude",),
    "medium": ("llm-claude-medium",),
}


def download(
    out_dir: Path,
    vm_host: str = "ryan@birdwatcher.ryanhoulette.com",
    vm_repo: str = "BirdWatcher",
    tiers: tuple[str, ...] = ("gold", "high"),
) -> None:
    """rsync crops + dump labels via SSH."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # ----- 1. Pull labels first (cheap) so we know which crops to want. -----
    log.info("Querying BirdWatcher DB on %s …", vm_host)
    labels_csv = _query_labels(out_dir, vm_host, vm_repo, tiers)
    log.info("Wrote labels → %s", labels_csv)

    # ----- 2. Build the list of crop_path values we actually need. -----
    wanted: list[tuple[str, str, str]] = []  # (crop_path, species, tier)
    for line in labels_csv.read_text().splitlines()[1:]:
        crop_path, species, tier = line.split(",", 2)
        wanted.append((crop_path, species, tier))
    log.info("DB returned %d labels across requested tiers", len(wanted))

    # ----- 3. Rsync only the crops we need. -----
    # Build an --files-from list to avoid pulling the full ~5GB crops/
    # directory (most of which isn't labeled).
    listing = out_dir / "_rsync_list.txt"
    listing.write_text("\n".join(crop_path for crop_path, _, _ in wanted))
    crops_local = out_dir / "_crops_raw"
    crops_local.mkdir(exist_ok=True)
    log.info("rsync %d crop files …", len(wanted))
    subprocess.run(
        [
            "rsync", "-a", "--info=stats2",
            f"--files-from={listing}",
            f"{vm_host}:{vm_repo}/backend/data/",
            str(crops_local),
        ],
        check=True,
    )
    listing.unlink(missing_ok=True)

    # ----- 4. Promote rsynced crops into the canonical ImageFolder layout. -----
    class_names: list[str] = sorted({sp for _, sp, _ in wanted})
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    image_paths: list[str] = []
    labels: list[int] = []
    splits: list[str] = []
    notes_skipped = 0
    for crop_path, species, _tier in wanted:
        src = crops_local / crop_path
        if not src.exists():
            notes_skipped += 1
            continue
        class_dir = images_dir / slug(species)
        class_dir.mkdir(exist_ok=True)
        dst = class_dir / Path(crop_path).name
        rel_path = f"images/{slug(species)}/{Path(crop_path).name}"
        if not dst.exists():
            shutil.move(src, dst)
        image_paths.append(rel_path)
        labels.append(label_to_idx[species])
        # Yard data has no canonical split — let manifest.py decide.
        splits.append("unknown")
    if notes_skipped:
        log.warning("Skipped %d crops missing on the host (likely pruned)", notes_skipped)
    # Drop the now-empty rsync staging tree.
    shutil.rmtree(crops_local, ignore_errors=True)

    meta = SourceMetadata(
        source="yard",
        downloaded_at=now_iso(),
        image_count=len(image_paths),
        class_names=class_names,
        image_paths=image_paths,
        labels=labels,
        splits=splits,
        notes={
            "vm_host": vm_host,
            "tiers_included": list(tiers),
            "license_note": "BirdWatcher private dataset — Apache-2.0 release contingent on yard data being non-PII (crops only, no metadata).",
        },
    )
    write_metadata(out_dir, meta)
    log.info("Done: %d crops, %d species → %s", len(image_paths), len(class_names), out_dir)


def _query_labels(out_dir: Path, vm_host: str, vm_repo: str, tiers: tuple[str, ...]) -> Path:
    """Dump (crop_path, species, tier) rows via SSH+docker exec into a CSV."""
    sources_per_tier_lines = "\n".join(
        f"TIER_SOURCES[{name!r}] = {list(TIER_TO_SOURCES[name])!r}"
        for name in tiers
    )
    py_script = f"""
from sqlalchemy import func
from db.session import SessionLocal
from db.models import Correction, Detection, Species

TIER_SOURCES = {{}}
{sources_per_tier_lines}

db = SessionLocal()
try:
    sub = (db.query(Correction.detection_id, func.max(Correction.id).label('latest'))
             .group_by(Correction.detection_id).subquery())
    rows = (db.query(Detection.crop_path, Species.common_name, Correction.source)
              .join(sub, sub.c.detection_id == Detection.id)
              .join(Correction, Correction.id == sub.c.latest)
              .join(Species, Species.id == Correction.correct_species_id).all())
    print('crop_path,species,tier')
    for crop_path, species, source in rows:
        if not crop_path or not species:
            continue
        for tier, sources in TIER_SOURCES.items():
            if source in sources:
                # Quote the species in case it contains a comma.
                print(f'{{crop_path}},\"{{species}}\",{{tier}}')
                break
finally:
    db.close()
"""
    csv_path = out_dir / "labels.csv"
    log.info("Running label dump on VM …")
    result = subprocess.run(
        [
            "ssh", vm_host,
            "docker", "compose", "-f", f"{vm_repo}/docker-compose.yml",
            "exec", "-T", "api", "python", "-c", py_script,
        ],
        capture_output=True, text=True, check=True,
    )
    csv_path.write_text(result.stdout)
    return csv_path
