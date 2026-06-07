"""Dataset downloaders.

One sub-command per source. Each fetches the dataset into a uniform
on-disk layout under ``raw_data/<source>/`` and writes a per-source
``metadata.json`` consumed by ``manifest.py``.

Implemented sources:

- ``gpiosenka``: 525-species bird classification dataset. Original
  Kaggle upload was removed by gpiosenka in 2025; we pull from the
  HuggingFace mirror at ``yashikota/birds-525-species-image-classification``
  which is bit-identical to the original (89,885 imgs, 525 classes).
  No auth needed. ~2 GB.
- ``nabirds``: Cornell's NABirds v1. User pre-downloads
  ``nabirds.tar.gz`` from https://dl.allaboutbirds.org/nabirds and
  passes the local path via ``--nabirds-tar``. ~3 GB.
- ``inat21birds``: bird subset of iNat21. Streams the official tarballs
  from the visipedia S3 bucket (no account needed), extracts only the
  Aves category. ~75 GB after filter.
- ``yard``: BirdWatcher's labeled crops, rsynced from the BirdWatcher
  VM, plus the DB-side label dump (one CSV row per detection).

Each downloader is idempotent — re-running skips already-downloaded
files (presence-checked on disk, since each source's "shard" granularity
differs).
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("data.download")


@click.command()
@click.option(
    "--datasets",
    required=True,
    help="Comma-separated subset of: gpiosenka,nabirds,inat21birds,yard",
)
@click.option(
    "--raw-data-dir",
    default="raw_data",
    type=click.Path(file_okay=False, path_type=Path),
    help="Where to download to. Defaults to ./raw_data/ (gitignored).",
)
@click.option(
    "--nabirds-tar",
    default=None,
    type=click.Path(dir_okay=False, exists=False, path_type=Path),
    help="Path to a pre-downloaded nabirds.tar.gz. Required if --datasets includes nabirds.",
)
@click.option(
    "--vm-host",
    default="ryan@birdwatcher.ryanhoulette.com",
    help="SSH target for the yard-data downloader.",
)
@click.option(
    "--vm-repo",
    default="BirdWatcher",
    help="Remote path to the BirdWatcher repo on the VM.",
)
@click.option(
    "--yard-tiers",
    default="gold,high",
    help="Comma-separated yard-data trust tiers to include (gold,high,medium).",
)
def main(
    datasets: str,
    raw_data_dir: Path,
    nabirds_tar: Path | None,
    vm_host: str,
    vm_repo: str,
    yard_tiers: str,
) -> None:
    """Fetch one or more training datasets."""
    # Lazy imports so unused sources don't pay the dep cost at --help time.
    requested = {s.strip() for s in datasets.split(",") if s.strip()}
    known = {"gpiosenka", "nabirds", "inat21birds", "yard"}
    unknown = requested - known
    if unknown:
        raise click.UsageError(f"Unknown datasets: {sorted(unknown)}")
    if "nabirds" in requested and not nabirds_tar:
        raise click.UsageError("--nabirds-tar is required when --datasets includes nabirds")

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    for name in sorted(requested):
        log.info("=== %s ===", name)
        target = raw_data_dir / name
        if name == "gpiosenka":
            from data.sources import gpiosenka
            gpiosenka.download(target)
        elif name == "nabirds":
            from data.sources import nabirds
            nabirds.download(target, tarball=nabirds_tar)
        elif name == "inat21birds":
            from data.sources import inat21birds
            inat21birds.download(target)
        elif name == "yard":
            from data.sources import yard
            tiers = tuple(t.strip() for t in yard_tiers.split(",") if t.strip())
            yard.download(target, vm_host=vm_host, vm_repo=vm_repo, tiers=tiers)


if __name__ == "__main__":
    main()
