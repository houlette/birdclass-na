"""Merge per-dataset label spaces into one canonical taxonomy.

Reads each source's ``raw_data/<source>/metadata.json`` (produced by
``data/download.py``) and emits a single ``taxonomy.json`` mapping
every source label → canonical name (or sentinel ``OTHER`` for
non-NA species).

Design choice — NA-focused canonical list:

  The canonical class list is built from **NABirds' leaf species**
  (Cornell's expert-curated list of NA-occurring species, ~555),
  augmented by yard-only species not yet in NABirds (e.g. family-
  level catch-alls like "Sparrow" that BirdWatcher uses when the
  user is unsure of species). Every non-NA bird image in iNat21 /
  gpiosenka maps to ``OTHER``.

  Why: the project goal is the best NA classifier, not a global one.
  A focused head (~570 classes) trains faster and generalizes better
  on NA species than a 1,500-way head with sparse non-NA data. Non-NA
  bird images still contribute "this is a bird, not NAB" training
  signal via the ``OTHER`` class.

Output ``taxonomy.json`` shape:

  {
    "canonical": ["American Robin", "Northern Cardinal", ..., "OTHER"],
    "aliases": {
      "gpiosenka": { "AMERICAN ROBIN": "American Robin", ... },
      "nabirds":   { "American Robin": "American Robin", ... },
      "inat21birds": { "American Robin": "American Robin", ... },
      "yard":      { "American Robin": "American Robin", "Sparrow": "Sparrow", ... }
    },
    "na_class_count": 568,         # canonical excluding OTHER
    "other_index": 568             # canonical[other_index] == "OTHER"
  }

The training loop / manifest builder uses ``aliases[source][label]``
to translate any source's label to a canonical class index.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("data.build_unified_taxonomy")

OTHER_LABEL = "OTHER"

# Hand-curated alias table for known variants. Mostly handles:
#   - American/Common prefix differences (e.g. "American Robin" vs "Robin")
#   - Spelling variants from BirdWatcher's calibration set
#   - Family-level catch-alls the yard uses (kept as their own canonical
#     entries; the model can predict at family granularity for ambiguous
#     crops).
HAND_ALIASES: dict[str, str] = {
    # gpiosenka uses "FINCHES" plural for some classes — singular it.
    "Finches": "Finch",
    # NABirds uses "Black-capped Chickadee" but some sources just say "Chickadee".
    # Map the family back to the canonical NABirds species.
    "Chickadee": "Black-capped Chickadee",
}


def _normalize(s: str) -> str:
    """Title-case canonical form used for matching across datasets.

    - Strips parenthetical suffixes (NABirds leaves carry morph / age
      annotations like "Red-tailed Hawk (Light morph adult)" — we
      collapse all morphs to species level so they map to a single
      canonical class). gpiosenka / iNat21 / yard labels don't have
      parentheticals, so stripping is a no-op on those.
    - Lower-cases, removes accents/apostrophes, collapses whitespace,
      then title-cases. ``"ABBOTT'S BABBLER"`` → ``"Abbotts Babbler"``.
    """
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("’", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.title()


@click.command()
@click.option(
    "--raw-data-dir",
    default="raw_data",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--out",
    default="taxonomy.json",
    type=click.Path(dir_okay=False, path_type=Path),
)
def main(raw_data_dir: Path, out: Path) -> None:
    """Build the unified taxonomy from whichever sources are downloaded."""
    available_sources = [d.name for d in raw_data_dir.iterdir()
                        if (d / "metadata.json").exists()] if raw_data_dir.exists() else []
    if not available_sources:
        raise click.UsageError(f"No metadata.json files under {raw_data_dir}/. "
                               "Run `python -m data.download …` first.")
    log.info("Sources available: %s", ", ".join(sorted(available_sources)))

    # NABirds is the anchor — its species list defines the canonical NA set.
    if "nabirds" not in available_sources:
        raise click.UsageError(
            "NABirds is the canonical-species anchor and must be downloaded first. "
            "Run `python -m data.download --datasets nabirds --nabirds-tar …` "
            "before building the taxonomy."
        )

    metas: dict[str, dict] = {}
    for src in available_sources:
        metas[src] = json.loads((raw_data_dir / src / "metadata.json").read_text())

    # ----- Build the canonical name list --------------------------------
    # Start from NABirds; normalize and dedupe.
    canonical_set: set[str] = set()
    nabirds_canonical: dict[str, str] = {}   # original_label → canonical
    for label in metas["nabirds"]["class_names"]:
        canon = _normalize(label)
        canon = HAND_ALIASES.get(canon, canon)
        canonical_set.add(canon)
        nabirds_canonical[label] = canon

    # Augment with yard-only species (e.g. "Sparrow" family catch-all)
    # so the model can predict at family granularity when uncertain.
    yard_canonical: dict[str, str] = {}
    if "yard" in metas:
        for label in metas["yard"]["class_names"]:
            canon = _normalize(label)
            canon = HAND_ALIASES.get(canon, canon)
            yard_canonical[label] = canon
            # Only adopt the yard label as a new canonical if it doesn't
            # already resolve to a known NA species (e.g. yard's "Sparrow"
            # is a family-level catch-all NABirds doesn't have).
            if canon not in canonical_set:
                canonical_set.add(canon)

    # Finalize canonical ordering: NA species alphabetical, then OTHER last.
    canonical: list[str] = sorted(canonical_set) + [OTHER_LABEL]
    log.info("Canonical: %d NA classes + OTHER", len(canonical) - 1)

    # ----- Build per-source alias maps ----------------------------------
    aliases: dict[str, dict[str, str]] = {}
    # NABirds: already canonicalized above.
    aliases["nabirds"] = nabirds_canonical
    if "yard" in metas:
        aliases["yard"] = yard_canonical

    # gpiosenka + iNat21: anything that normalizes to a known NA class
    # stays; everything else → OTHER.
    canonical_lookup = set(canonical) - {OTHER_LABEL}
    for src in ("gpiosenka", "inat21birds"):
        if src not in metas:
            continue
        src_aliases: dict[str, str] = {}
        n_kept = 0
        n_other = 0
        for label in metas[src]["class_names"]:
            canon = _normalize(label)
            canon = HAND_ALIASES.get(canon, canon)
            if canon in canonical_lookup:
                src_aliases[label] = canon
                n_kept += 1
            else:
                src_aliases[label] = OTHER_LABEL
                n_other += 1
        log.info("%s: %d labels → %d NA-matched, %d → OTHER",
                 src, len(metas[src]["class_names"]), n_kept, n_other)
        aliases[src] = src_aliases

    payload = {
        "canonical": canonical,
        "aliases": aliases,
        "na_class_count": len(canonical) - 1,
        "other_index": canonical.index(OTHER_LABEL),
    }
    out.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s (%d canonical classes)", out, len(canonical))


if __name__ == "__main__":
    main()
