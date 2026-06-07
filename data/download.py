"""Dataset downloaders.

One sub-command per source. Each fetches the dataset into a uniform
on-disk layout under ``raw_data/<source>/`` and writes a per-source
``metadata.json`` consumed by ``manifest.py``.

Implemented sources (Phase 1, next task):

- ``gpiosenka``: Kaggle's 525-species set. Requires a Kaggle API token
  at ``~/.kaggle/kaggle.json``. ~5 GB.
- ``nabirds``: Cornell's NABirds v1. Requires manual TOS click-through
  to obtain the download URL; the script prompts.
- ``inat21birds``: bird subset of iNat21. Requires accepting iNat21's
  TOS on their site; the script then extracts the Aves images from the
  full iNat21 tarball (~250 GB) producing ~75 GB on disk.
- ``yard``: BirdWatcher's labeled crops, rsynced from the VM, plus the
  DB-side label dump (one CSV row per detection).

Each downloader is idempotent — re-running skips already-downloaded
shards via the per-source ``state.json`` checkpoint.
"""
from __future__ import annotations

import click


@click.command()
@click.option(
    "--datasets",
    required=True,
    help="Comma-separated subset of: gpiosenka,nabirds,inat21birds,yard",
)
@click.option(
    "--raw-data-dir",
    default="raw_data",
    help="Where to download to. Defaults to ./raw_data/ (gitignored).",
)
def main(datasets: str, raw_data_dir: str) -> None:
    """Fetch one or more training datasets."""
    requested = {s.strip() for s in datasets.split(",") if s.strip()}
    handlers = {
        "gpiosenka": _download_gpiosenka,
        "nabirds": _download_nabirds,
        "inat21birds": _download_inat21birds,
        "yard": _download_yard,
    }
    unknown = requested - set(handlers)
    if unknown:
        raise click.UsageError(f"Unknown datasets: {sorted(unknown)}")
    for name in sorted(requested):
        click.echo(f"=== {name} ===")
        handlers[name](raw_data_dir)


def _download_gpiosenka(raw_data_dir: str) -> None:
    raise NotImplementedError("Phase 1 task #2 — implement gpiosenka Kaggle fetch.")


def _download_nabirds(raw_data_dir: str) -> None:
    raise NotImplementedError("Phase 1 task #2 — implement NABirds Cornell fetch.")


def _download_inat21birds(raw_data_dir: str) -> None:
    raise NotImplementedError("Phase 1 task #2 — implement iNat21-Birds subset extract.")


def _download_yard(raw_data_dir: str) -> None:
    raise NotImplementedError("Phase 1 task #2 — implement BirdWatcher yard data rsync.")


if __name__ == "__main__":
    main()
