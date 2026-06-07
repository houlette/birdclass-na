"""Phase 2 — Linear probe gate.

Frozen DINOv2-B backbone, linear classification head, trained on a
200k-image subsample of the unified manifest. Cheap (~$15 on a rented
A100) and quick (~4 hr). Result decides whether we proceed to the full
fine-tune.

Gate criterion: top-1 on gpiosenka test split must beat
``dennisjooo/Birds-Classifier-EfficientNetB2`` by at least 3 pp. If it
doesn't, the foundation backbone story isn't pulling its weight and
we should stop before spending the rest of the budget.

The code pattern (class-weighted CE, AdamW differential LR, cosine
schedule, save best by val) follows ``BirdWatcher/backend/scripts/
train/finetune_binary.py`` — same conventions so the two repos stay
mentally aligned.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--manifest", default="manifests/", help="Directory with train/val/test CSVs.")
@click.option("--out", default="runs/probe/")
@click.option("--epochs", default=12, show_default=True)
@click.option("--batch-size", default=128, show_default=True)
@click.option("--lr", default=3e-3, show_default=True)
@click.option("--features-cache", default="features_cache/dinov2-base/")
def main(
    manifest: str, out: str, epochs: int, batch_size: int, lr: float, features_cache: str,
) -> None:
    raise NotImplementedError("Phase 2 task #4 — implement linear-probe training.")


if __name__ == "__main__":
    main()
