"""Phase 4 — Benchmark harness.

Compares our trained model against three external reference points:

1. ``dennisjooo/Birds-Classifier-EfficientNetB2`` on the gpiosenka 525
   test split. Apples-to-apples: same training data + test split, we
   compare top-1 / top-5 / per-class. Computed by running both models
   over the test split end-to-end.
2. **Published NABirds top-1/5**: we report our top-1/5/mean-per-class
   on the NABirds test split. The model card cites the relevant
   published number alongside (we don't have a runnable competitor
   here — the published SOTA models aren't always released).
3. **Merlin Bird ID on a 100-crop yard test set**. The user submits
   the same 100 crops via the Merlin app and exports the top-1 picks
   as a CSV with columns ``(image_id, merlin_top1)``; we score top-1
   against the user's ground-truth label.

Output: ``BENCHMARK.md`` with all three tables + 95% bootstrap CIs
suitable for embedding directly into the model card.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train._common import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval.run_benchmarks")


DENNISJOOO_REPO = "dennisjooo/Birds-Classifier-EfficientNetB2"
N_BOOTSTRAP = 1000


# ---------- model loaders ----------

def _load_our_model(model_dir: Path, n_classes: int, dev: torch.device):
    """Load our trained DINOv2-based model from runs/finetune/."""
    from transformers import AutoModel

    ckpt = torch.load(model_dir / "model.pt", map_location=dev)
    backbone = AutoModel.from_pretrained(ckpt["backbone_repo"])
    backbone.load_state_dict(ckpt["backbone_state"])
    head = torch.nn.Linear(backbone.config.hidden_size, n_classes)
    head.load_state_dict(ckpt["head_state"])
    backbone.to(dev).eval()
    head.to(dev).eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def predict(img_tensor_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = backbone(pixel_values=img_tensor_batch).last_hidden_state[:, 0]
            return head(feats)

    return predict, transform, ckpt["canonical"]


def _load_dennisjooo(dev: torch.device):
    """Load denisjooo's published model from HuggingFace Hub."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    proc = AutoImageProcessor.from_pretrained(DENNISJOOO_REPO)
    model = AutoModelForImageClassification.from_pretrained(DENNISJOOO_REPO)
    model.to(dev).eval()
    # Class names in denisjooo's id2label (525 species, screaming-case).
    classes: list[str] = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    img_size = proc.size["height"] if isinstance(proc.size, dict) else 260
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
    ])

    def predict(img_tensor_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(pixel_values=img_tensor_batch).logits

    return predict, transform, classes


# ---------- data loader ----------

class _ManifestImageDataset(Dataset):
    def __init__(self, raw_data_dir: Path, paths: list[str], labels: list[int], transform):
        self.raw_data_dir = raw_data_dir
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Skip missing / unreadable images gracefully — return a black frame
        # rather than crashing. Same fallback used by train/finetune.py.
        # A small fraction of images get this treatment; for accuracy
        # numbers this is a worst-case "always wrong" sample.
        path = self.raw_data_dir / self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError, FileNotFoundError):
            log.warning("Unreadable / missing image %s — black-frame fallback", path)
            img = Image.new("RGB", (224, 224))
        return self.transform(img), self.labels[idx]


# ---------- metrics ----------

def _bootstrap_ci(correct: list[int], n_bootstrap: int = N_BOOTSTRAP, alpha: float = 0.05
                  ) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of a binary correctness array."""
    arr = np.array(correct, dtype=np.int8)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed=42)
    means = np.empty(n_bootstrap, dtype=np.float32)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _evaluate(
    predict_fn,
    transform,
    raw_data_dir: Path,
    paths: list[str],
    labels: list[int],
    dev: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Run a model over the given (path, label) pairs and return top-1 /
    top-5 / per-class metrics + the correctness array (for bootstrap CIs)."""
    ds = _ManifestImageDataset(raw_data_dir, paths, labels, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    correct1: list[int] = []
    correct5: list[int] = []
    per_class_correct: Counter = Counter()
    per_class_total: Counter = Counter()
    for imgs, lbls in tqdm(loader, desc="eval"):
        imgs = imgs.to(dev, non_blocking=True)
        lbls_np = lbls.numpy()
        logits = predict_fn(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu().numpy()
        for i in range(len(lbls_np)):
            ok1 = int(preds[i] == lbls_np[i])
            ok5 = int(lbls_np[i] in top5[i])
            correct1.append(ok1)
            correct5.append(ok5)
            per_class_total[lbls_np[i]] += 1
            if ok1:
                per_class_correct[lbls_np[i]] += 1

    top1 = sum(correct1) / max(len(correct1), 1)
    top5 = sum(correct5) / max(len(correct5), 1)
    lo1, hi1 = _bootstrap_ci(correct1)
    mean_per_class = (
        sum(per_class_correct[c] / per_class_total[c] for c in per_class_total)
        / max(len(per_class_total), 1)
    )
    return {
        "top1": top1, "top1_ci": (lo1, hi1),
        "top5": top5,
        "mean_per_class_top1": mean_per_class,
        "correct1": correct1,
        "n": len(correct1),
    }


# ---------- comparison strategies ----------

def _filter_to_source(paths, labels, sources, target_source: str):
    keep = [i for i, s in enumerate(sources) if s == target_source]
    return [paths[i] for i in keep], [labels[i] for i in keep]


def _remap_labels_to_dennisjooo(
    paths: list[str],
    labels: list[int],
    canonical: list[str],
    dennisjooo_classes: list[str],
) -> tuple[list[str], list[int]]:
    """For the gpiosenka head-to-head: only keep test samples whose
    canonical label exists in denisjooo's 525-class label space. The
    label index has to be re-mapped to denisjooo's vocabulary.
    """
    # denisjooo uses SCREAMING_SNAKE labels. Build canonical → denisjooo index.
    dj_lookup = {name.upper().replace("'", "").replace("-", " ").strip(): i
                 for i, name in enumerate(dennisjooo_classes)}
    out_paths, out_labels = [], []
    n_dropped = 0
    for p, label_idx in zip(paths, labels):
        canon = canonical[label_idx]
        key = canon.upper().replace("'", "").replace("-", " ").strip()
        if key in dj_lookup:
            out_paths.append(p)
            out_labels.append(dj_lookup[key])
        else:
            n_dropped += 1
    log.info("Remapped %d → %d for denisjooo eval (%d dropped: not in 525-class space)",
             len(paths), len(out_paths), n_dropped)
    return out_paths, out_labels


# ---------- Merlin comparison ----------

def _eval_merlin_csv(
    merlin_csv: Path,
    raw_data_dir: Path,
    our_predict, our_transform, our_canonical: list[str],
    dev: torch.device,
) -> dict:
    """Load the Merlin CSV (image_path, merlin_top1, ground_truth) and
    score both Merlin and our model against the user's ground-truth.

    The CSV is the manual-curation artifact: the user runs 100 yard
    crops through the Merlin Bird ID app on their phone (~30 min) and
    exports their top-1 picks. The ground-truth column should be the
    user's verified label, separately from Merlin's guess.
    """
    rows = []
    with open(merlin_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    log.info("Loaded %d rows from %s", len(rows), merlin_csv)

    paths = [r["image_path"] for r in rows]
    truth = [r["ground_truth"] for r in rows]
    merlin = [r["merlin_top1"] for r in rows]

    # Score Merlin first (string-equal, normalized).
    def _norm(s: str) -> str:
        return s.strip().lower().replace("-", " ")
    merlin_correct = [int(_norm(m) == _norm(t)) for m, t in zip(merlin, truth)]

    # Score our model.
    canon_lookup = {_norm(name): i for i, name in enumerate(our_canonical)}
    label_indices = [canon_lookup.get(_norm(t)) for t in truth]
    # Drop samples whose ground truth isn't in our canonical (would
    # only happen if the user labeled a non-NA species).
    keep = [i for i, x in enumerate(label_indices) if x is not None]
    if len(keep) < len(rows):
        log.warning("%d samples dropped (ground truth not in canonical)", len(rows) - len(keep))

    ours_paths = [paths[i] for i in keep]
    ours_labels = [label_indices[i] for i in keep]
    ours_metrics = _evaluate(our_predict, our_transform, raw_data_dir,
                             ours_paths, ours_labels, dev)

    merlin_kept = [merlin_correct[i] for i in keep]
    merlin_top1 = sum(merlin_kept) / max(len(merlin_kept), 1)
    merlin_ci = _bootstrap_ci(merlin_kept)
    return {
        "n_scored": len(keep),
        "our_top1": ours_metrics["top1"],
        "our_top1_ci": ours_metrics["top1_ci"],
        "merlin_top1": merlin_top1,
        "merlin_top1_ci": merlin_ci,
    }


# ---------- Markdown report ----------

def _write_report(out: Path, results: dict) -> None:
    def fmt_ci(value, ci):
        lo, hi = ci
        return f"{value*100:.1f}% (95% CI {lo*100:.1f}–{hi*100:.1f})"

    lines = ["# BENCHMARK.md\n"]
    lines.append(f"_Generated for model at: `{results['model_dir']}`_\n")

    if "gpiosenka" in results:
        g = results["gpiosenka"]
        lines.append("\n## 1. vs `dennisjooo/Birds-Classifier-EfficientNetB2` on gpiosenka test split\n")
        lines.append("| Model | n | Top-1 | Top-5 | Mean per-class top-1 |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(f"| **ours** | {g['ours']['n']} | "
                     f"{fmt_ci(g['ours']['top1'], g['ours']['top1_ci'])} | "
                     f"{g['ours']['top5']*100:.1f}% | "
                     f"{g['ours']['mean_per_class_top1']*100:.1f}% |")
        lines.append(f"| denisjooo | {g['theirs']['n']} | "
                     f"{fmt_ci(g['theirs']['top1'], g['theirs']['top1_ci'])} | "
                     f"{g['theirs']['top5']*100:.1f}% | "
                     f"{g['theirs']['mean_per_class_top1']*100:.1f}% |")
        delta = g['ours']['top1'] - g['theirs']['top1']
        lines.append(f"\n**Δ top-1: {delta*100:+.1f} pp** "
                     f"({'meets' if delta >= 0.05 else 'falls short of'} the +5 pp ship gate)\n")

    if "nabirds" in results:
        n = results["nabirds"]
        lines.append("\n## 2. on NABirds test split\n")
        lines.append("| Model | n | Top-1 | Top-5 | Mean per-class top-1 |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(f"| **ours** | {n['n']} | "
                     f"{fmt_ci(n['top1'], n['top1_ci'])} | "
                     f"{n['top5']*100:.1f}% | "
                     f"{n['mean_per_class_top1']*100:.1f}% |")
        lines.append("| published NABirds SOTA | — | _(cite paper in model card)_ | — | — |")

    if "merlin" in results:
        m = results["merlin"]
        lines.append("\n## 3. vs Merlin Bird ID on yard test set\n")
        lines.append(f"_{m['n_scored']} hand-curated yard crops; user submitted to Merlin app + recorded top-1._\n")
        lines.append("| Model | Top-1 |")
        lines.append("|---|---:|")
        lines.append(f"| **ours** | {fmt_ci(m['our_top1'], m['our_top1_ci'])} |")
        lines.append(f"| Merlin Bird ID | {fmt_ci(m['merlin_top1'], m['merlin_top1_ci'])} |")
        delta = m['our_top1'] - m['merlin_top1']
        lines.append(f"\n**Δ top-1: {delta*100:+.1f} pp** on yard conditions.\n")

    out.write_text("\n".join(lines))
    log.info("Wrote %s", out)


# ---------- CLI ----------

@click.command()
@click.option("--model", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Trained-model directory (runs/finetune/).")
@click.option("--manifest", default="manifests/", type=click.Path(file_okay=False, path_type=Path))
@click.option("--raw-data-dir", default="raw_data", type=click.Path(file_okay=False, path_type=Path))
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--out", default="BENCHMARK.md", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--merlin-csv", default=None, type=click.Path(dir_okay=False, path_type=Path),
              help="CSV with columns image_path, merlin_top1, ground_truth.")
@click.option("--skip-merlin", is_flag=True)
@click.option("--skip-dennisjooo", is_flag=True,
              help="Skip the denisjooo head-to-head (saves ~5 min per run).")
def main(
    model: Path,
    manifest: Path,
    raw_data_dir: Path,
    taxonomy: Path,
    out: Path,
    merlin_csv: Path | None,
    skip_merlin: bool,
    skip_dennisjooo: bool,
) -> None:
    """Run all available benchmarks and write BENCHMARK.md."""
    from train._common import device
    dev = device()
    log.info("Using device: %s", dev)

    tax = json.loads(taxonomy.read_text())
    canonical: list[str] = tax["canonical"]
    log.info("Loading our model from %s …", model)
    our_predict, our_transform, model_canonical = _load_our_model(model, len(canonical), dev)

    test_paths, test_labels, test_sources = load_manifest(manifest / "test.csv")
    log.info("Test manifest: %d rows", len(test_paths))

    results: dict = {"model_dir": str(model)}

    # ----- Comparison 1: head-to-head with denisjooo on gpiosenka test -----
    if not skip_dennisjooo:
        log.info("Loading denisjooo's model …")
        dj_predict, dj_transform, dj_classes = _load_dennisjooo(dev)
        g_paths, g_labels = _filter_to_source(test_paths, test_labels, test_sources, "gpiosenka")
        log.info("gpiosenka test split: %d rows", len(g_paths))
        # Our model: use canonical labels directly.
        ours_g = _evaluate(our_predict, our_transform, raw_data_dir, g_paths, g_labels, dev)
        # denisjooo: remap labels into its 525-class space.
        dj_paths, dj_labels = _remap_labels_to_dennisjooo(g_paths, g_labels, canonical, dj_classes)
        theirs_g = _evaluate(dj_predict, dj_transform, raw_data_dir, dj_paths, dj_labels, dev)
        results["gpiosenka"] = {"ours": ours_g, "theirs": theirs_g}

    # ----- Comparison 2: NABirds test (no runnable competitor) -----
    n_paths, n_labels = _filter_to_source(test_paths, test_labels, test_sources, "nabirds")
    if n_paths:
        log.info("NABirds test split: %d rows", len(n_paths))
        results["nabirds"] = _evaluate(our_predict, our_transform, raw_data_dir, n_paths, n_labels, dev)

    # ----- Comparison 3: Merlin head-to-head on yard data -----
    if not skip_merlin:
        if merlin_csv is None:
            log.warning("--merlin-csv not provided; skipping Merlin comparison.")
        else:
            results["merlin"] = _eval_merlin_csv(
                merlin_csv, raw_data_dir, our_predict, our_transform, model_canonical, dev,
            )

    _write_report(out, results)
    (out.with_suffix(".json")).write_text(json.dumps(
        {k: v for k, v in results.items() if k != "correct1"},
        default=lambda o: "[non-serializable]", indent=2,
    ))


if __name__ == "__main__":
    main()
