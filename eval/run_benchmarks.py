"""Phase 4 — Benchmark harness.

Three comparators:

1. **vs denisjooo**: load both models, score gpiosenka's held-out test
   split, report top-1 / top-5 / per-class accuracy with 95% bootstrap
   CIs.
2. **vs published NABirds SOTA**: report our model's top-1 / top-5 /
   mean-per-class on NABirds test; the model card cites the relevant
   paper's number alongside.
3. **vs Merlin on yard data**: 100 manually-curated yard crops; the
   user submits them via the Merlin Bird ID app and exports the CSV
   (~30-60 min one-time). We report top-1 agreement, where each model
   counts as "correct" iff it matches the user's ground-truth label.

Output: ``BENCHMARK.md`` with all three tables and the 95% CIs.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--model", required=True, help="Path to a fine-tuned model directory.")
@click.option("--out", default="BENCHMARK.md")
@click.option(
    "--skip-merlin",
    is_flag=True,
    help="Skip the Merlin comparison (it requires a hand-curated CSV).",
)
@click.option(
    "--merlin-csv",
    default=None,
    help="CSV with columns (image_path, merlin_top1, ground_truth). Required unless --skip-merlin.",
)
def main(model: str, out: str, skip_merlin: bool, merlin_csv: str | None) -> None:
    raise NotImplementedError("Phase 4 task #6 — implement benchmark harness.")


if __name__ == "__main__":
    main()
