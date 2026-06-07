"""Phase 2 — Linear probe gate.

Frozen DINOv2-B backbone, fresh linear head, trained on a 200k-image
subsample. Cheap and quick: ~$15 / ~4 hr on a rented A100.

Two passes:

1. **Feature extraction** (one-time): run DINOv2-B over every image in
   the manifest, save 768-dim float16 vectors to a single .npy file
   per split. Caches to ``features_cache/dinov2-base/{split}.npy`` and
   matching ``labels.npy`` / ``sources.npy``. Subsequent epochs are
   pure feature → linear → loss, no image I/O.
2. **Head training**: ``Linear(768, n_classes)`` with class-weighted
   cross-entropy, AdamW, cosine LR schedule. Train for many epochs
   cheaply since each epoch is just a few seconds of FP ops.

Gate decision happens after training: report top-1 on gpiosenka test
split vs denisjooo (manual lookup of published number, or rerun
denisjooo on the same test split if budget allows). ≥ +3 pp = proceed.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train._common import class_weights, device, load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train.linear_probe")

DINOV2_REPO = "facebook/dinov2-base"
EMBED_DIM = 768   # DINOv2-B's CLS feature width


class ImagePathDataset(Dataset):
    """Maps manifest rows → preprocessed image tensors. Used only during
    feature extraction; head training uses the cached feature .npy."""

    def __init__(self, raw_data_dir: Path, paths: list[str], transform):
        self.raw_data_dir = raw_data_dir
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.raw_data_dir / self.paths[idx]
        # Defensive: occasionally an image is truncated / unreadable.
        # Return a black tensor in that case so the batch shape stays
        # valid; the corresponding row will get a near-uniform feature
        # vector and won't dominate the loss.
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            log.warning("Unreadable image %s — returning black frame", path)
            return torch.zeros(3, 224, 224)
        return self.transform(img)


def _build_transform():
    """DINOv2 standard preprocessing: resize, crop, ImageNet-normalize."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _extract_features(
    raw_data_dir: Path,
    paths: list[str],
    batch_size: int,
    backbone: nn.Module,
    dev: torch.device,
) -> np.ndarray:
    """Run DINOv2-B once and return (N, 768) float16 features."""
    backbone.eval()
    ds = ImagePathDataset(raw_data_dir, paths, _build_transform())
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    out = np.zeros((len(ds), EMBED_DIM), dtype=np.float16)
    cursor = 0
    with torch.no_grad():
        for imgs in tqdm(loader, desc="extract"):
            imgs = imgs.to(dev, non_blocking=True)
            # DINOv2's HF API returns a BaseModelOutput; the CLS token is
            # last_hidden_state[:, 0].
            out_b = backbone(pixel_values=imgs).last_hidden_state[:, 0]
            out[cursor : cursor + out_b.shape[0]] = out_b.cpu().numpy().astype(np.float16)
            cursor += out_b.shape[0]
    return out


def _load_or_extract_features(
    cache_dir: Path,
    raw_data_dir: Path,
    split_name: str,
    paths: list[str],
    labels: list[int],
    sources: list[str],
    batch_size: int,
    backbone,
    dev: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (features, labels, sources) for a split, cached on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_p = cache_dir / f"{split_name}_features.npy"
    label_p = cache_dir / f"{split_name}_labels.npy"
    src_p = cache_dir / f"{split_name}_sources.json"
    if feat_p.exists() and label_p.exists() and src_p.exists():
        log.info("Using cached features for %s (%s)", split_name, feat_p)
        return (
            np.load(feat_p),
            np.load(label_p),
            json.loads(src_p.read_text()),
        )
    log.info("Extracting %s features (%d images) …", split_name, len(paths))
    features = _extract_features(raw_data_dir, paths, batch_size, backbone, dev)
    np.save(feat_p, features)
    np.save(label_p, np.asarray(labels, dtype=np.int64))
    src_p.write_text(json.dumps(sources))
    return features, np.asarray(labels, dtype=np.int64), sources


def _evaluate(
    head: nn.Linear,
    features: np.ndarray,
    labels: np.ndarray,
    sources: list[str],
    n_classes: int,
    dev: torch.device,
) -> dict:
    """Top-1, top-5, and per-source top-1 accuracy."""
    head.eval()
    feats = torch.from_numpy(features.astype(np.float32)).to(dev)
    with torch.no_grad():
        logits = head(feats)
        preds_top1 = logits.argmax(dim=1).cpu().numpy()
        preds_top5 = logits.topk(5, dim=1).indices.cpu().numpy()
    correct_top1 = (preds_top1 == labels).sum()
    correct_top5 = sum(labels[i] in preds_top5[i] for i in range(len(labels)))
    by_src_correct: dict[str, int] = {}
    by_src_total: dict[str, int] = {}
    for i, s in enumerate(sources):
        by_src_total[s] = by_src_total.get(s, 0) + 1
        if preds_top1[i] == labels[i]:
            by_src_correct[s] = by_src_correct.get(s, 0) + 1
    return {
        "top1": correct_top1 / len(labels),
        "top5": correct_top5 / len(labels),
        "per_source_top1": {
            s: by_src_correct.get(s, 0) / n for s, n in by_src_total.items()
        },
        "n_examples": len(labels),
    }


@click.command()
@click.option("--manifest", default="manifests/", type=click.Path(file_okay=False, path_type=Path),
              help="Directory with train/val/test CSVs from data.manifest (use --split=probe).")
@click.option("--raw-data-dir", default="raw_data", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", default="runs/probe/", type=click.Path(file_okay=False, path_type=Path))
@click.option("--features-cache", default="features_cache/dinov2-base/",
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--epochs", default=15, show_default=True)
@click.option("--batch-size", default=256, show_default=True,
              help="Per-step minibatch size for head training (feature-space, cheap).")
@click.option("--extract-batch-size", default=64, show_default=True,
              help="DINOv2 feature-extraction batch size (image-space, GPU-RAM bound).")
@click.option("--lr", default=3e-3, show_default=True)
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
def main(
    manifest: Path,
    raw_data_dir: Path,
    out: Path,
    features_cache: Path,
    epochs: int,
    batch_size: int,
    extract_batch_size: int,
    lr: float,
    taxonomy: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    log.info("Using device: %s", dev)

    tax = json.loads(taxonomy.read_text())
    n_classes = len(tax["canonical"])
    log.info("Taxonomy: %d canonical classes", n_classes)

    # Load all three split manifests.
    splits = {}
    for split in ("train", "val", "test"):
        paths, labels, sources = load_manifest(manifest / f"{split}.csv")
        log.info("Manifest %s: %d rows", split, len(paths))
        splits[split] = (paths, labels, sources)

    # ----- DINOv2 backbone (frozen) ----------
    from transformers import AutoModel
    log.info("Loading DINOv2-B backbone …")
    backbone = AutoModel.from_pretrained(DINOV2_REPO).to(dev).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # ----- Extract / load features for each split ----------
    feats = {}
    for split in ("train", "val", "test"):
        paths, labels, sources = splits[split]
        feats[split] = _load_or_extract_features(
            features_cache, raw_data_dir, split,
            paths, labels, sources,
            extract_batch_size, backbone, dev,
        )

    # Free the backbone — we don't need it again.
    del backbone
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ----- Linear head ----------
    head = nn.Linear(EMBED_DIM, n_classes).to(dev)
    weights = class_weights(splits["train"][1], n_classes, dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_feats, train_labels, _ = feats["train"]
    n_train = train_feats.shape[0]
    indices = np.arange(n_train)

    history = []
    best_val_top1 = 0.0
    for epoch in range(1, epochs + 1):
        head.train()
        np.random.shuffle(indices)
        t0 = time.time()
        loss_acc = 0.0
        steps = 0
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            x = torch.from_numpy(train_feats[batch_idx].astype(np.float32)).to(dev)
            y = torch.from_numpy(train_labels[batch_idx]).to(dev)
            optimizer.zero_grad()
            logits = head(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            loss_acc += loss.item()
            steps += 1
        scheduler.step()
        train_loss = loss_acc / max(steps, 1)

        # Validation
        val_metrics = _evaluate(head, *feats["val"], n_classes=n_classes, dev=dev)
        elapsed = time.time() - t0
        log.info("epoch %d/%d  train_loss=%.3f  val_top1=%.3f  val_top5=%.3f  %.1fs",
                 epoch, epochs, train_loss,
                 val_metrics["top1"], val_metrics["top5"], elapsed)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_top1": float(val_metrics["top1"]), "val_top5": float(val_metrics["top5"]),
            "per_source_val_top1": val_metrics["per_source_top1"],
        })

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            torch.save({
                "state_dict": head.state_dict(),
                "n_classes": n_classes,
                "embed_dim": EMBED_DIM,
                "canonical": tax["canonical"],
                "backbone_repo": DINOV2_REPO,
            }, out / "best_head.pt")
            log.info("    ↑ saved best (val_top1=%.3f)", best_val_top1)

    # ----- Final test eval ----------
    log.info("=== final test eval ===")
    head.load_state_dict(torch.load(out / "best_head.pt")["state_dict"])
    test_metrics = _evaluate(head, *feats["test"], n_classes=n_classes, dev=dev)
    log.info("test top-1: %.3f  top-5: %.3f", test_metrics["top1"], test_metrics["top5"])
    for src, acc in sorted(test_metrics["per_source_top1"].items()):
        log.info("  per-source %-12s top-1: %.3f", src, acc)

    (out / "metrics.json").write_text(json.dumps({
        "best_val_top1": best_val_top1,
        "test": {
            "top1": float(test_metrics["top1"]),
            "top5": float(test_metrics["top5"]),
            "per_source_top1": test_metrics["per_source_top1"],
            "n_examples": test_metrics["n_examples"],
        },
        "history": history,
    }, indent=2))
    log.info("Wrote metrics to %s", out / "metrics.json")


if __name__ == "__main__":
    main()
