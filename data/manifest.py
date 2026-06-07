"""Emit per-split CSV manifests for the training loop.

Reads ``taxonomy.json`` (from ``build_unified_taxonomy.py``) + each
source's ``metadata.json`` and writes ``manifests/{train,val,test}.csv``
with columns:

    path, label_idx, source_dataset, original_label

The training loop is pure manifest-driven — it never re-walks the
dataset trees. Splits are deterministic via a seeded hash on the file
path, so re-running produces the same splits unless the underlying
data changes.

Two output modes:

- ``--split=full`` — every available sample, honoring each source's
  declared train/val/test where present. Yard data (which has no
  canonical split) is hash-split 80/10/10.
- ``--split=probe`` — same split assignment but capped at
  ``--probe-cap`` total images, drawn proportionally per (source,
  class) with a per-class floor of 10 so the long tail isn't lost.

Manifest paths are RELATIVE to ``--raw-data-dir`` so the training loop
constructs absolute paths via ``raw_data_dir / row['path']`` and the
manifests stay portable between machines.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("data.manifest")


@click.command()
@click.option("--raw-data-dir", default="raw_data", type=click.Path(file_okay=False, path_type=Path))
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--out", default="manifests/", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--split",
    type=click.Choice(["full", "probe"]),
    default="full",
    help="`probe` emits a capped subsample for linear-probe training.",
)
@click.option("--probe-cap", default=200_000, show_default=True)
@click.option("--seed", default=42, show_default=True)
def main(
    raw_data_dir: Path,
    taxonomy: Path,
    out: Path,
    split: str,
    probe_cap: int,
    seed: int,
) -> None:
    """Build per-split manifests from each source's metadata.json."""
    tax = json.loads(taxonomy.read_text())
    canonical: list[str] = tax["canonical"]
    aliases: dict[str, dict[str, str]] = tax["aliases"]
    label_to_idx = {name: i for i, name in enumerate(canonical)}
    log.info("Canonical taxonomy: %d classes", len(canonical))

    # Discover source metas.
    metas: dict[str, dict] = {}
    for d in raw_data_dir.iterdir():
        meta_p = d / "metadata.json"
        if meta_p.exists():
            metas[d.name] = json.loads(meta_p.read_text())

    # Flatten into a single list of (canonical_label_idx, abs_rel_path,
    # source, original_label, split_hint). The training loader joins
    # raw_data_dir/<path> at load time.
    rows: list[tuple[str, int, str, str, str]] = []
    for src, m in metas.items():
        src_class_names = m["class_names"]
        src_aliases = aliases.get(src, {})
        for i in range(m["image_count"]):
            rel_path = m["image_paths"][i]   # already relative to raw_data/<src>/
            src_label_idx = m["labels"][i]
            src_label = src_class_names[src_label_idx]
            canon_label = src_aliases.get(src_label)
            if canon_label is None:
                # Source label not aliased — drop. Should only happen
                # for "Unknown bird" or NAB sentinels we deliberately
                # exclude before metadata is written, but be defensive.
                continue
            canon_idx = label_to_idx[canon_label]
            split_hint = m["splits"][i]
            # Path stored in manifest is relative to raw_data_dir (not raw_data/<src>/).
            rel_to_root = f"{src}/{rel_path}"
            rows.append((rel_to_root, canon_idx, src, src_label, split_hint))
    log.info("Pulled %d total rows across %d sources", len(rows), len(metas))

    # ----- Resolve splits -----------------------------------------------
    # Sources that declared a split: honor it.
    # Sources whose split is "unknown" (yard): hash-split 80/10/10 by path.
    finalized: list[tuple[str, int, str, str, str]] = []
    for path, idx, src, src_label, split_hint in rows:
        if split_hint in ("train", "val", "test"):
            finalized.append((path, idx, src, src_label, split_hint))
        else:
            bucket = _hash_bucket(path)
            if bucket < 80:
                s = "train"
            elif bucket < 90:
                s = "val"
            else:
                s = "test"
            finalized.append((path, idx, src, src_label, s))

    # ----- Probe-mode subsampling ---------------------------------------
    if split == "probe":
        finalized = _probe_subsample(finalized, probe_cap, seed)
        log.info("After probe-cap=%d subsample: %d rows", probe_cap, len(finalized))

    # ----- Emit CSVs ----------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    per_split: dict[str, list] = defaultdict(list)
    for row in finalized:
        per_split[row[4]].append(row)
    for split_name in ("train", "val", "test"):
        rows_split = per_split.get(split_name, [])
        csv_path = out / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label_idx", "source_dataset", "original_label"])
            for path, idx, src, src_label, _ in rows_split:
                w.writerow([path, idx, src, src_label])
        log.info("Wrote %s (%d rows)", csv_path, len(rows_split))


def _hash_bucket(s: str) -> int:
    """Stable 0-99 bucket for hash-based splitting."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def _probe_subsample(
    rows: list[tuple[str, int, str, str, str]],
    cap: int,
    seed: int,
) -> list[tuple[str, int, str, str, str]]:
    """Cap total samples at ``cap``, stratified per (source, class) with
    a per-bucket floor so rare classes aren't dropped entirely."""
    rng = random.Random(seed)
    # Group by (source, label_idx).
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for r in rows:
        groups[(r[2], r[1])].append(r)

    # Compute per-group target sizes: proportional to original size,
    # then floored at min(10, group_size) so rare classes survive.
    total = len(rows)
    floor = 10
    target_groups: dict[tuple[str, int], int] = {}
    for k, items in groups.items():
        proportional = max(1, int(cap * len(items) / total))
        target_groups[k] = min(len(items), max(proportional, min(floor, len(items))))

    # If the floor pushes total above cap, scale-down the large groups.
    total_target = sum(target_groups.values())
    if total_target > cap:
        # Scale down groups that exceed their floor proportionally to
        # absorb the overshoot.
        overshoot = total_target - cap
        scalable = [(k, target_groups[k] - min(floor, len(groups[k])))
                    for k in target_groups
                    if target_groups[k] > min(floor, len(groups[k]))]
        scalable_total = sum(extra for _, extra in scalable)
        if scalable_total > 0:
            for k, extra in scalable:
                share = int(overshoot * extra / scalable_total)
                target_groups[k] -= share

    sampled: list = []
    for k, items in groups.items():
        rng.shuffle(items)
        sampled.extend(items[: target_groups[k]])
    return sampled


if __name__ == "__main__":
    main()
