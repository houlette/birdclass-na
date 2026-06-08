"""Phase 3 — Full DINOv2-B fine-tune (single-stage, modern ViT recipe).

Earlier iteration used a two-stage general-then-domain layout; the
domain stage on 1,617 yard samples wrecked the model (train_loss
exploded 1.5 → 8.8 from rare-class weights blowing up gradient norm).
This version uses the **mixed-source manifest as one training set**
and adds the modern fine-tuning machinery that closes most of the gap
between "naive AdamW for 2 epochs" and a well-tuned ViT fine-tune:

- **Linear warmup → cosine decay**: 5 % of training steps for warmup
  prevents the first step's big gradient from wrecking the pretrained
  backbone. Cosine decay over the remaining 95 %.
- **Layer-wise LR decay (LLRD)**: each transformer block has its own
  LR, decaying by `LLRD_FACTOR` per layer going down the stack. Top
  blocks (close to the head) train near the base LR; bottom blocks
  (close to the input embeddings) train at much lower LR so we don't
  disturb DINOv2's well-learned low-level features.
- **Class-weight capping**: inverse-frequency weights clamped at
  10× so rare classes don't blow up the loss. Without this a single
  rare-class sample dominates the batch gradient.
- **Label smoothing 0.1**: standard for fine-grained tasks; trades
  a sliver of training accuracy for a meaningful generalization gain.
- **bf16 autocast** on CUDA.
- **Best-by-val saved as HF Dinov2ForImageClassification format** so
  the publish script can hand it directly to `upload_folder`.
"""
from __future__ import annotations

import json
import logging
import math
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

from train._common import device, load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train.finetune")

DINOV2_REPO = "facebook/dinov2-base"
EMBED_DIM = 768
# Per-layer LR multiplier (going from top to bottom of the encoder).
# 0.75 is the value the original BEiT/DINOv2 fine-tuning papers report
# works best for ViT-B; smaller (0.65) protects pretrained features
# more aggressively but trains slower.
LLRD_FACTOR = 0.75
# Cap on inverse-frequency class weights — without this a single
# 1-sample-per-class rare bird gets 50,000× the gradient weight of
# OTHER and the batch loss is fully dominated by it.
CLASS_WEIGHT_CAP = 10.0
LABEL_SMOOTHING = 0.1


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


# ---------- LR scheduling ----------------------------------------------------

def _capped_class_weights(labels: list[int], n_classes: int,
                          dev: torch.device, cap: float = CLASS_WEIGHT_CAP) -> torch.Tensor:
    """Inverse-frequency weights, clamped so rare classes can't blow
    up the loss. Mean of weights is kept ≈ 1 for stable LR scaling."""
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.zeros(n_classes, dtype=torch.float32, device=dev)
    for i in range(n_classes):
        n = counts.get(i, 1)
        weights[i] = total / (n_classes * n)
    weights = weights.clamp(max=cap)
    # Re-normalize so mean ≈ 1 → loss scale doesn't drift vs uniform.
    weights = weights / weights.mean()
    return weights


def _llrd_param_groups(model: DINOv2WithHead, lr_backbone: float, lr_head: float,
                       decay: float = LLRD_FACTOR) -> list[dict]:
    """Layer-wise LR decay for the DINOv2 backbone.

    Group structure (top-of-stack → bottom-of-stack):
      - head            → lr_head             (full)
      - encoder.layer.{N-1, N-2, …, 0}  →  lr_backbone × decay^k
      - embeddings      → lr_backbone × decay^N   (deepest, smallest LR)
    """
    groups: list[dict] = []
    # Head: full LR.
    groups.append({"params": list(model.head.parameters()), "lr": lr_head, "name": "head"})

    # Encoder blocks: layer 11 (top) → layer 0 (bottom).
    blocks = list(model.backbone.encoder.layer)
    n_blocks = len(blocks)
    # Also collect the final LayerNorm if present.
    extra_top = []
    if hasattr(model.backbone, "layernorm"):
        extra_top.extend(model.backbone.layernorm.parameters())
    if extra_top:
        groups.append({"params": extra_top, "lr": lr_backbone,
                       "name": "backbone_layernorm"})

    for k, block in enumerate(reversed(blocks)):
        lr = lr_backbone * (decay ** k)
        groups.append({"params": list(block.parameters()), "lr": lr,
                       "name": f"backbone_block_{n_blocks - 1 - k}"})

    # Embeddings: deepest layer.
    groups.append({"params": list(model.backbone.embeddings.parameters()),
                   "lr": lr_backbone * (decay ** (n_blocks + 1)),
                   "name": "backbone_embeddings"})

    return groups


class _WarmupCosineLR(optim.lr_scheduler._LRScheduler):
    """Linear warmup over `warmup_steps`, then cosine decay to 0 over
    the remaining steps."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return [base_lr * scale for base_lr in self.base_lrs]


# ---------- Eval -------------------------------------------------------------

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


# ---------- CLI --------------------------------------------------------------

@click.command()
@click.option("--manifest", default="manifests/", type=click.Path(file_okay=False, path_type=Path))
@click.option("--raw-data-dir", default="raw_data", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", default="runs/finetune/", type=click.Path(file_okay=False, path_type=Path))
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--resume-from", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Probe-run directory; head weights initialized from runs/probe/best_head.pt.")
@click.option("--epochs", default=15, show_default=True,
              help="Single-stage training over the full train manifest (gpiosenka + nabirds + "
                   "inat21birds + yard, mixed). No separate domain stage.")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--grad-accum-steps", default=1, show_default=True)
@click.option("--lr-backbone", default=2e-5, show_default=True,
              help="Peak backbone LR. With LLRD, deeper layers train at this × decay^k.")
@click.option("--lr-head", default=5e-4, show_default=True,
              help="Peak head LR. Always full (no LLRD).")
@click.option("--warmup-frac", default=0.05, show_default=True,
              help="Fraction of total steps used as linear warmup.")
@click.option("--weight-decay", default=0.05, show_default=True)
@click.option("--num-workers", default=8, show_default=True)
def main(
    manifest: Path,
    raw_data_dir: Path,
    out: Path,
    taxonomy: Path,
    resume_from: Path | None,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr_backbone: float,
    lr_head: float,
    warmup_frac: float,
    weight_decay: float,
    num_workers: int,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    log.info("Using device: %s", dev)

    tax = json.loads(taxonomy.read_text())
    n_classes = len(tax["canonical"])
    canonical = tax["canonical"]
    log.info("Taxonomy: %d canonical classes", n_classes)

    train_paths, train_labels, train_sources = load_manifest(manifest / "train.csv")
    val_paths, val_labels, val_sources = load_manifest(manifest / "val.csv")
    log.info("Train: %d  Val: %d", len(train_paths), len(val_paths))

    # ----- Build model -----
    model = DINOv2WithHead(n_classes).to(dev)
    if resume_from is not None:
        probe_ckpt = torch.load(resume_from / "best_head.pt", map_location=dev, weights_only=False)
        log.info("Initializing head from probe at %s", resume_from)
        model.head.load_state_dict(probe_ckpt["state_dict"])

    # ----- Optimizer with LLRD param groups -----
    param_groups = _llrd_param_groups(model, lr_backbone, lr_head, decay=LLRD_FACTOR)
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    log.info("Built %d AdamW param groups; smallest LR=%.2e, largest LR=%.2e",
             len(param_groups),
             min(g["lr"] for g in param_groups),
             max(g["lr"] for g in param_groups))

    # ----- Loss with capped weights + label smoothing -----
    weights = _capped_class_weights(train_labels, n_classes, dev, cap=CLASS_WEIGHT_CAP)
    log.info("Class weights: min=%.3f median=%.3f max=%.3f (cap=%.1f)",
             weights.min().item(), weights.median().item(), weights.max().item(),
             CLASS_WEIGHT_CAP)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    # ----- Train + val loaders -----
    train_ds = ManifestImageDataset(raw_data_dir, train_paths, train_labels, _train_transform())
    val_ds = ManifestImageDataset(raw_data_dir, val_paths, val_labels, _val_transform())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    log.info("Top-10 train classes by count: %s",
             [(canonical[c], n) for c, n in Counter(train_labels).most_common(10)])

    # ----- Scheduler: linear warmup → cosine -----
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(100, int(warmup_frac * total_steps))
    log.info("Total optimizer steps: %d  Warmup: %d (%.0f%%)",
             total_steps, warmup_steps, 100 * warmup_steps / max(total_steps, 1))
    scheduler = _WarmupCosineLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # ----- Train loop -----
    use_amp = dev.type == "cuda"
    history: list[dict] = []
    best_val_top1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_acc = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (imgs, lbls) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")):
            imgs = imgs.to(dev, non_blocking=True)
            lbls = lbls.to(dev, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(imgs)
                    loss = loss_fn(logits, lbls) / grad_accum_steps
            else:
                logits = model(imgs)
                loss = loss_fn(logits, lbls) / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_acc += float(loss.item()) * grad_accum_steps
            steps += 1
        train_loss = loss_acc / max(steps, 1)

        # Eval
        val_metrics = _evaluate(model, val_loader, val_sources, dev)
        elapsed = time.time() - t0
        current_lr_head = optimizer.param_groups[0]["lr"]
        log.info("epoch %d/%d  train_loss=%.3f  val_top1=%.3f  val_top5=%.3f  "
                 "lr_head=%.2e  %.0fs",
                 epoch, epochs, train_loss,
                 val_metrics["top1"], val_metrics["top5"], current_lr_head, elapsed)
        for src, acc in sorted(val_metrics["per_source_top1"].items()):
            log.info("    per-source %-12s val_top1: %.3f", src, acc)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_top1": val_metrics["top1"], "val_top5": val_metrics["top5"],
            "per_source_val_top1": val_metrics["per_source_top1"],
            "lr_head": current_lr_head,
        })

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            _save_hf_format(model, out, tax)
            (out / "metrics.json").write_text(json.dumps({
                "best_val_top1": best_val_top1,
                "best_epoch": epoch,
                "best_val_top5": val_metrics["top5"],
                "best_per_source_val_top1": val_metrics["per_source_top1"],
            }, indent=2))
            log.info("    ↑ saved best (val_top1=%.3f) to %s", best_val_top1, out)

    (out / "history.json").write_text(json.dumps(history, indent=2))
    log.info("Done. Best val_top1=%.3f  → %s", best_val_top1, out)


def _save_hf_format(model: DINOv2WithHead, out: Path, tax: dict) -> None:
    """Save backbone + head as a single raw checkpoint. The publish
    script (scripts/publish.py) repackages this into an HF
    Dinov2ForImageClassification directory."""
    torch.save({
        "backbone_state": model.backbone.state_dict(),
        "head_state": model.head.state_dict(),
        "canonical": tax["canonical"],
        "n_classes": len(tax["canonical"]),
        "backbone_repo": DINOV2_REPO,
    }, out / "model.pt")


if __name__ == "__main__":
    main()
