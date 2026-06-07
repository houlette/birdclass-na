"""Phase 3 — Full DINOv2-B fine-tune.

Two-stage training:

1. **General stage**: train on the union of gpiosenka + NABirds +
   iNat21-Birds (`general_sources`) for ``--epochs-general`` epochs.
   This is where the model learns the broad NA taxonomy.
2. **Domain stage**: continue training on yard data only for
   ``--epochs-domain`` epochs. This is where the model adapts to
   feeder-camera conditions (partial occlusion, motion blur, fence
   clutter) that the public datasets don't have.

Key implementation choices:

- **Differential LR**: backbone lr=5e-5, head lr=5e-4. The fresh head
  needs ~10× more learning than the pre-trained backbone.
- **bf16 autocast**: A100s have native bf16 support; cuts memory ~half
  with no quality penalty on this size of model.
- **Resume from probe**: ``--resume-from runs/probe/`` initializes the
  head from the linear probe's weights so we start with a working
  classifier rather than a randomly-initialized one.
- **Gradient accumulation**: ``--grad-accum-steps`` lets us hit
  effective batch=64 even if real batch must be smaller for GPU RAM.
- **HF-format save**: the best-by-val checkpoint is saved via
  ``save_pretrained()`` so ``scripts/publish.py`` can push it directly.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train._common import class_weights, device, load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train.finetune")

DINOV2_REPO = "facebook/dinov2-base"
EMBED_DIM = 768


class ManifestImageDataset(Dataset):
    """Loads images from disk per row. Applies the given transform."""

    def __init__(
        self,
        raw_data_dir: Path,
        paths: list[str],
        labels: list[int],
        transform,
    ):
        self.raw_data_dir = raw_data_dir
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.raw_data_dir / self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            log.warning("Unreadable image %s — substituting black frame", path)
            img = Image.new("RGB", (224, 224))
        return self.transform(img), self.labels[idx]


def _train_transform():
    """Heavy augmentation for the fine-tune: random resize-crop, flip,
    color jitter. We don't go too aggressive because birds are partly
    color-cued (cardinals, finches, jays)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _val_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DINOv2WithHead(nn.Module):
    """DINOv2 backbone + linear classification head."""

    def __init__(self, n_classes: int):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(DINOV2_REPO)
        self.head = nn.Linear(EMBED_DIM, n_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.head(feats)


def _evaluate(model: DINOv2WithHead, loader: DataLoader, sources: list[str], dev) -> dict:
    model.eval()
    n = 0
    n_correct = 0
    n_correct_top5 = 0
    by_src_correct: dict[str, int] = {}
    by_src_total: dict[str, int] = {}
    cursor = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="eval", leave=False):
            imgs = imgs.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            top5 = logits.topk(5, dim=1).indices
            for i in range(labels.shape[0]):
                src = sources[cursor + i]
                by_src_total[src] = by_src_total.get(src, 0) + 1
                if preds[i] == labels[i]:
                    n_correct += 1
                    by_src_correct[src] = by_src_correct.get(src, 0) + 1
                if labels[i] in top5[i]:
                    n_correct_top5 += 1
            n += labels.shape[0]
            cursor += labels.shape[0]
    return {
        "top1": n_correct / max(n, 1),
        "top5": n_correct_top5 / max(n, 1),
        "per_source_top1": {
            s: by_src_correct.get(s, 0) / by_src_total[s] for s in by_src_total
        },
        "n_examples": n,
    }


@click.command()
@click.option("--manifest", default="manifests/", type=click.Path(file_okay=False, path_type=Path))
@click.option("--raw-data-dir", default="raw_data", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", default="runs/finetune/", type=click.Path(file_okay=False, path_type=Path))
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--resume-from", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Path to a previous run (e.g. runs/probe/) whose head we initialize from.")
@click.option("--epochs-general", default=2, show_default=True)
@click.option("--epochs-domain", default=1, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--grad-accum-steps", default=1, show_default=True,
              help="Virtual batch multiplier (steps before each .step()).")
@click.option("--lr-backbone", default=5e-5, show_default=True)
@click.option("--lr-head", default=5e-4, show_default=True)
@click.option("--num-workers", default=4, show_default=True)
def main(
    manifest: Path,
    raw_data_dir: Path,
    out: Path,
    taxonomy: Path,
    resume_from: Path | None,
    epochs_general: int,
    epochs_domain: int,
    batch_size: int,
    grad_accum_steps: int,
    lr_backbone: float,
    lr_head: float,
    num_workers: int,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    log.info("Using device: %s", dev)

    tax = json.loads(taxonomy.read_text())
    n_classes = len(tax["canonical"])
    log.info("Taxonomy: %d canonical classes", n_classes)

    train_paths, train_labels, train_sources = load_manifest(manifest / "train.csv")
    val_paths, val_labels, val_sources = load_manifest(manifest / "val.csv")
    log.info("Train: %d  Val: %d", len(train_paths), len(val_paths))

    # ----- Build model ----------
    model = DINOv2WithHead(n_classes).to(dev)
    if resume_from is not None:
        probe_ckpt = torch.load(resume_from / "best_head.pt", map_location=dev)
        log.info("Initializing head from %s (val_top1 was %.3f at probe stage)",
                 resume_from, _read_metric(resume_from, "best_val_top1"))
        model.head.load_state_dict(probe_ckpt["state_dict"])

    # ----- Optimizer w/ differential LR ----------
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=0.01,
    )

    # ----- Loss: class-weighted cross-entropy ----------
    weights = class_weights(train_labels, n_classes, dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ----- Val loader (used between every stage / epoch) ----------
    val_ds = ManifestImageDataset(raw_data_dir, val_paths, val_labels, _val_transform())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    history: list[dict] = []
    best_val_top1 = 0.0

    # ----- Stage 1: GENERAL ----------
    general_idx = [i for i, s in enumerate(train_sources)
                   if s in ("gpiosenka", "nabirds", "inat21birds")]
    log.info("=== STAGE 1: general (n=%d, %d epochs) ===", len(general_idx), epochs_general)
    if general_idx:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_general)
        best_val_top1 = _run_stage(
            stage="general", train_idx=general_idx,
            model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
            train_paths=train_paths, train_labels=train_labels,
            raw_data_dir=raw_data_dir, val_loader=val_loader, val_sources=val_sources,
            epochs=epochs_general, batch_size=batch_size, grad_accum_steps=grad_accum_steps,
            num_workers=num_workers, dev=dev, out=out, tax=tax,
            history=history, best_val_top1=best_val_top1,
        )

    # ----- Stage 2: DOMAIN (yard only) ----------
    domain_idx = [i for i, s in enumerate(train_sources) if s == "yard"]
    if domain_idx and epochs_domain > 0:
        log.info("=== STAGE 2: domain (yard n=%d, %d epochs) ===", len(domain_idx), epochs_domain)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_domain)
        best_val_top1 = _run_stage(
            stage="domain", train_idx=domain_idx,
            model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
            train_paths=train_paths, train_labels=train_labels,
            raw_data_dir=raw_data_dir, val_loader=val_loader, val_sources=val_sources,
            epochs=epochs_domain, batch_size=batch_size, grad_accum_steps=grad_accum_steps,
            num_workers=num_workers, dev=dev, out=out, tax=tax,
            history=history, best_val_top1=best_val_top1,
        )
    elif not domain_idx:
        log.warning("No yard data in train manifest — skipping domain stage. "
                    "Did you run `python -m data.download --datasets yard`?")

    # ----- Final metrics dump ----------
    (out / "history.json").write_text(json.dumps(history, indent=2))
    log.info("Done. Best val_top1=%.3f  → %s", best_val_top1, out)


def _read_metric(run_dir: Path, key: str) -> float:
    p = run_dir / "metrics.json"
    if not p.exists():
        return float("nan")
    try:
        return float(json.loads(p.read_text()).get(key, float("nan")))
    except (ValueError, json.JSONDecodeError):
        return float("nan")


def _run_stage(
    *, stage: str,
    train_idx: list[int],
    model: DINOv2WithHead,
    optimizer: optim.Optimizer,
    scheduler,
    loss_fn,
    train_paths: list[str],
    train_labels: list[int],
    raw_data_dir: Path,
    val_loader: DataLoader,
    val_sources: list[str],
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    num_workers: int,
    dev: torch.device,
    out: Path,
    tax: dict,
    history: list[dict],
    best_val_top1: float,
) -> float:
    """Run one training stage. Returns updated best_val_top1."""
    paths = [train_paths[i] for i in train_idx]
    labels = [train_labels[i] for i in train_idx]
    train_ds = ManifestImageDataset(raw_data_dir, paths, labels, _train_transform())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    log.info("[%s] class counts (top 10): %s",
             stage, Counter(labels).most_common(10))

    use_amp = dev.type == "cuda"
    autocast_dtype = torch.bfloat16

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_acc = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (imgs, lbls) in enumerate(tqdm(train_loader, desc=f"[{stage}] epoch {epoch}")):
            imgs = imgs.to(dev, non_blocking=True)
            lbls = lbls.to(dev, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(imgs)
                    loss = loss_fn(logits, lbls) / grad_accum_steps
            else:
                logits = model(imgs)
                loss = loss_fn(logits, lbls) / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_acc += float(loss.item()) * grad_accum_steps
            steps += 1
        scheduler.step()
        train_loss = loss_acc / max(steps, 1)

        # Eval
        val_metrics = _evaluate(model, val_loader, val_sources, dev)
        elapsed = time.time() - t0
        log.info("[%s] epoch %d/%d  train_loss=%.3f  val_top1=%.3f  val_top5=%.3f  %.1fs",
                 stage, epoch, epochs, train_loss,
                 val_metrics["top1"], val_metrics["top5"], elapsed)
        for src, acc in sorted(val_metrics["per_source_top1"].items()):
            log.info("    per-source %-12s val_top1: %.3f", src, acc)

        history.append({
            "stage": stage, "epoch": epoch, "train_loss": train_loss,
            "val_top1": val_metrics["top1"], "val_top5": val_metrics["top5"],
            "per_source_val_top1": val_metrics["per_source_top1"],
        })

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            _save_hf_format(model, out, tax)
            (out / "metrics.json").write_text(json.dumps({
                "best_val_top1": best_val_top1,
                "best_stage": stage,
                "best_epoch": epoch,
            }, indent=2))
            log.info("    ↑ saved best (val_top1=%.3f) to %s", best_val_top1, out)
    return best_val_top1


def _save_hf_format(model: DINOv2WithHead, out: Path, tax: dict) -> None:
    """Save backbone + head as a single HF-compatible classifier directory
    so scripts/publish.py can hand it directly to upload_folder."""
    # The DINOv2 image classification model in transformers wraps the
    # backbone+head exactly as we've assembled it. We export by writing
    # a state_dict directly + the config; the publish script repackages
    # it into a proper AutoModelForImageClassification on the destination.
    torch.save({
        "backbone_state": model.backbone.state_dict(),
        "head_state": model.head.state_dict(),
        "canonical": tax["canonical"],
        "n_classes": len(tax["canonical"]),
        "backbone_repo": DINOV2_REPO,
    }, out / "model.pt")


if __name__ == "__main__":
    main()
