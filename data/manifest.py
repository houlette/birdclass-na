"""Emit per-split CSV manifests for the training loop.

Reads ``taxonomy.json`` + each source's ``metadata.json`` and writes
``manifests/{train,val,test}.csv`` with the columns:

    path, label_idx, source_dataset, original_label

The training loop is pure manifest-driven — it never re-walks the
dataset trees. Splits are deterministic via a seeded hash on a stable
sample identifier (file path or detection_id), so re-running produces
the same splits unless the underlying data changes.

Sampling caps for the linear-probe gate:

- ``--probe-cap``: total images included when ``--split=probe`` (default
  200k, drawn proportionally per source/class with a per-class floor of
  10 so the long tail isn't lost).

The full fine-tune uses all available data and ignores ``--probe-cap``.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--raw-data-dir", default="raw_data")
@click.option("--taxonomy", default="taxonomy.json")
@click.option("--out", default="manifests/")
@click.option(
    "--split",
    type=click.Choice(["full", "probe"]),
    default="full",
    help="`probe` emits a 200k-image subsample for linear-probe training.",
)
@click.option("--probe-cap", default=200_000, show_default=True)
def main(raw_data_dir: str, taxonomy: str, out: str, split: str, probe_cap: int) -> None:
    raise NotImplementedError("Phase 1 task #3 — implement manifest builder.")


if __name__ == "__main__":
    main()
