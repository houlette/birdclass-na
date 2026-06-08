"""Apples-to-apples baseline: run dennisjooo's model through our taxonomy.

The linear-probe gate compares our top-1 to denisjooo's published ~85%
on the gpiosenka 525-way test split. But our model predicts in a
**407-way NA-focused taxonomy** (406 NA species + OTHER bucket), so
direct comparison is broken: a gpiosenka test image of a SCARLET_MACAW
has true label "Scarlet Macaw" for denisjooo but "OTHER" for us.

This script runs denisjooo on our test manifest, maps each of its
525-way predictions through the same normalization our taxonomy
builder uses (hyphen-strip + alias table + non-NA → OTHER), and
scores top-1 against our canonical labels. The output is denisjooo's
top-1 on the EXACT same evaluation problem the probe was solving,
making the two numbers directly comparable.

Usage:
    python -m eval.baseline_denisjooo \\
        --manifest manifests/probe/test.csv \\
        --raw-data-dir raw_data \\
        --taxonomy taxonomy.json
"""
from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import click
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Re-use the manifest loader from train/_common.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train._common import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval.baseline_denisjooo")

DENNISJOOO_REPO = "dennisjooo/Birds-Classifier-EfficientNetB2"


def _normalize_match(s: str) -> str:
    """MUST match data.build_unified_taxonomy._normalize_match exactly,
    so denisjooo's "ROCK DOVE" and our canonical "Rock Pigeon" resolve
    consistently via the same alias table."""
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("’", "").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.title()


# Alias subset relevant for mapping denisjooo's labels.
# Mirror of HAND_ALIASES in build_unified_taxonomy.
HAND_ALIASES = {
    "Rock Dove": "Rock Pigeon",
    "Common Starling": "European Starling",
    "Common Pheasant": "Ring Necked Pheasant",
    "Common Pochard": "Pochard",
    "Common Tern": "Common Tern",
    "Finches": "Finch",
}


class _ImageDataset(Dataset):
    def __init__(self, raw_data_dir: Path, paths: list[str], transform):
        self.raw_data_dir = raw_data_dir
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.raw_data_dir / self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            log.warning("Unreadable image %s — returning black frame", path)
            return torch.zeros(3, 260, 260)
        return self.transform(img)


@click.command()
@click.option("--manifest", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--raw-data-dir", default="raw_data", type=click.Path(file_okay=False, path_type=Path))
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--batch-size", default=64, show_default=True)
@click.option("--num-workers", default=4, show_default=True)
def main(manifest: Path, raw_data_dir: Path, taxonomy: Path,
         batch_size: int, num_workers: int) -> None:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    # ----- Load model -----
    log.info("Loading %s …", DENNISJOOO_REPO)
    proc = AutoImageProcessor.from_pretrained(DENNISJOOO_REPO)
    model = AutoModelForImageClassification.from_pretrained(DENNISJOOO_REPO)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    log.info("Device: %s", dev)
    model.to(dev).eval()

    # denisjooo's 525 species labels (screaming-case).
    dj_classes: list[str] = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    log.info("denisjooo has %d classes", len(dj_classes))

    # ----- Load our taxonomy + manifest -----
    tax = json.loads(taxonomy.read_text())
    canonical: list[str] = tax["canonical"]
    canon_lookup = {name: i for i, name in enumerate(canonical)}
    other_idx = tax["other_index"]
    log.info("Our taxonomy: %d canonical classes (OTHER=%d)", len(canonical), other_idx)

    paths, labels, sources = load_manifest(manifest)
    log.info("Test manifest: %d rows", len(paths))

    # ----- Build denisjooo-class-idx → our-canonical-idx mapping -----
    dj_to_canon: list[int] = []
    n_matched_na = 0
    for cls in dj_classes:
        canon = _normalize_match(cls)
        canon = HAND_ALIASES.get(canon, canon)
        if canon in canon_lookup and canon != "OTHER":
            dj_to_canon.append(canon_lookup[canon])
            n_matched_na += 1
        else:
            dj_to_canon.append(other_idx)
    log.info("denisjooo NA-matched classes: %d / %d", n_matched_na, len(dj_classes))

    # ----- Preprocess transform matching denisjooo's processor -----
    img_size = proc.size["height"] if isinstance(proc.size, dict) else 260
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
    ])

    # ----- Run inference -----
    ds = _ImageDataset(raw_data_dir, paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    preds_canonical: list[int] = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="eval"):
            imgs = imgs.to(dev, non_blocking=True)
            logits = model(pixel_values=imgs).logits
            preds = logits.argmax(dim=1).cpu().numpy()
            for p in preds:
                preds_canonical.append(dj_to_canon[int(p)])

    # ----- Score top-1 overall + per-source -----
    correct = sum(1 for p, t in zip(preds_canonical, labels) if p == t)
    n = len(labels)
    top1 = correct / max(n, 1)
    by_src_correct: Counter = Counter()
    by_src_total: Counter = Counter()
    for p, t, s in zip(preds_canonical, labels, sources):
        by_src_total[s] += 1
        if p == t:
            by_src_correct[s] += 1
    log.info("=" * 50)
    log.info("dennisjooo on our 407-way taxonomy:")
    log.info("  overall top-1: %.3f  (%d / %d)", top1, correct, n)
    for src in sorted(by_src_total):
        acc = by_src_correct[src] / by_src_total[src]
        log.info("  per-source %-12s top-1: %.3f  (%d / %d)",
                 src, acc, by_src_correct[src], by_src_total[src])

    # ----- "Predict OTHER always" baseline for context -----
    other_baseline_correct = sum(1 for t in labels if t == other_idx)
    log.info("  baseline (predict OTHER always): %.3f",
             other_baseline_correct / max(n, 1))


if __name__ == "__main__":
    main()
