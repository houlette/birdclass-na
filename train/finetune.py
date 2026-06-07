"""Phase 3 — Full DINOv2-B fine-tune.

Unfrozen backbone (differential LR: backbone 5e-5, head 5e-4),
bf16 mixed precision, batch 64 on an A100 40 GB. Cosine schedule over
3 epochs.

Two-stage training:

1. General stage: gpiosenka + NABirds + iNat21-Birds. 1-2 epochs.
2. Domain stage: continue on yard data only. 1 epoch.

Pushes the best-by-val checkpoint to ``runs/finetune/`` in HuggingFace
format so ``scripts/publish.py`` can hand it directly to
``HfApi().upload_folder``.

Reuses differential-LR / AdamW pattern from BirdWatcher's
``finetune_binary.py``. New here: gradient accumulation, bf16 autocast,
``--resume-from`` to initialize the head from the linear probe.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--manifest", default="manifests/")
@click.option("--out", default="runs/finetune/")
@click.option(
    "--resume-from",
    default=None,
    help="Path to a previous run (e.g. runs/probe/) whose head we initialize from.",
)
@click.option("--epochs-general", default=2, show_default=True)
@click.option("--epochs-domain", default=1, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--lr-backbone", default=5e-5, show_default=True)
@click.option("--lr-head", default=5e-4, show_default=True)
def main(
    manifest: str,
    out: str,
    resume_from: str | None,
    epochs_general: int,
    epochs_domain: int,
    batch_size: int,
    lr_backbone: float,
    lr_head: float,
) -> None:
    raise NotImplementedError("Phase 3 task #5 — implement full fine-tune.")


if __name__ == "__main__":
    main()
