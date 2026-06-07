"""Merge per-dataset label spaces into one canonical taxonomy.

Reads each source's ``raw_data/<source>/metadata.json`` (produced by
``download.py``) and emits a single ``taxonomy.json`` mapping every
source label → canonical Cornell common name (or sentinel ``OTHER``
for non-NA species).

Why this is non-trivial:

- gpiosenka labels: capitalized common names (``Northern Cardinal``).
- NABirds labels: a 2-level hierarchy with parent classes (``Sparrows``)
  and child species (``Song Sparrow``); leaves only for training.
- iNat21 labels: scientific names (``Cardinalis cardinalis``).
- Yard labels: capitalized common names that mostly already match
  gpiosenka, with a few family-level catch-alls (``Sparrow``,
  ``Warbler``) we collapse into ``OTHER`` for the unified head.

Strategy:

1. Pick gpiosenka's common-name vocabulary as the canonical surface
   form (it's the most consistent and matches BirdWatcher's UI).
2. For NABirds, map child species via a hand-curated alias table.
3. For iNat21, map scientific → common via a static lookup derived
   from the Cornell Birds of the World API (offline JSON shipped in
   the repo so we don't depend on Cornell at training time).
4. Anything that doesn't resolve (rare non-NA species, hybrids,
   unclear sub-species) → ``OTHER``.

Output: ``taxonomy.json`` with shape
``{ "canonical": ["Northern Cardinal", ...], "aliases": { "<source>": { ... } } }``.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--raw-data-dir", default="raw_data")
@click.option("--out", default="taxonomy.json")
def main(raw_data_dir: str, out: str) -> None:
    raise NotImplementedError("Phase 1 task #3 — implement taxonomy merge.")


if __name__ == "__main__":
    main()
