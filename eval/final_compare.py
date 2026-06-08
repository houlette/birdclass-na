"""Final apples-to-apples comparison: our fine-tune vs denisjooo,
both evaluated on the full test split in our 407-way canonical
taxonomy. Per-source breakdown with 95 % bootstrap CIs.

Outputs:
- BENCHMARK.md  (drop-in for the HF model card)
- final_metrics.json
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
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train._common import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval.final_compare")

DENNISJOOO_REPO = "dennisjooo/Birds-Classifier-EfficientNetB2"
DINOV2_REPO = "facebook/dinov2-base"
N_BOOTSTRAP = 1000


# Reuse the same normalization as build_unified_taxonomy + baseline_denisjooo.
HAND_ALIASES = {
    "Rock Dove": "Rock Pigeon",
    "Common Starling": "European Starling",
    "Common Pheasant": "Ring Necked Pheasant",
    "Common Pochard": "Pochard",
}


def _normalize_match(s: str) -> str:
    s = re.sub(r"\s*\([^)]*\)", "", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("'", "").replace("’", "").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.title()


class _ImageDataset(Dataset):
    def __init__(self, raw_data_dir: Path, paths: list[str], labels: list[int],
                 transform):
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
        except (OSError, Image.UnidentifiedImageError, FileNotFoundError):
            img = Image.new("RGB", (224, 224))
        return self.transform(img), self.labels[idx]


def _bootstrap_ci(correct: np.ndarray, n: int = N_BOOTSTRAP) -> tuple[float, float]:
    if len(correct) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed=42)
    means = np.empty(n, dtype=np.float32)
    for i in range(n):
        idx = rng.integers(0, len(correct), len(correct))
        means[i] = correct[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@click.command()
@click.option("--model", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Our fine-tuned model dir (runs/finetune/).")
@click.option("--manifest", default="manifests/full/test.csv",
              type=click.Path(dir_okay=False, path_type=Path))
@click.option("--raw-data-dir", default="raw_data", type=click.Path(file_okay=False, path_type=Path))
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--out-md", default="BENCHMARK.md", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--batch-size", default=64, show_default=True)
@click.option("--num-workers", default=8, show_default=True)
def main(model: Path, manifest: Path, raw_data_dir: Path, taxonomy: Path,
         out_md: Path, batch_size: int, num_workers: int) -> None:
    from transformers import (
        AutoImageProcessor, AutoModel, AutoModelForImageClassification,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", dev)

    tax = json.loads(taxonomy.read_text())
    canonical: list[str] = tax["canonical"]
    n_classes = len(canonical)
    canon_lookup = {n: i for i, n in enumerate(canonical)}
    other_idx = tax["other_index"]

    paths, labels, sources = load_manifest(manifest)
    labels_arr = np.asarray(labels, dtype=np.int64)
    log.info("Test set: %d rows  canonical: %d classes", len(paths), n_classes)

    # ----- Our model -----
    log.info("Loading our fine-tuned model …")
    ckpt = torch.load(model / "model.pt", map_location=dev, weights_only=False)
    our_backbone = AutoModel.from_pretrained(DINOV2_REPO)
    our_backbone.load_state_dict(ckpt["backbone_state"])
    our_head = torch.nn.Linear(our_backbone.config.hidden_size, n_classes)
    our_head.load_state_dict(ckpt["head_state"])
    our_backbone.to(dev).eval()
    our_head.to(dev).eval()
    our_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ----- denisjooo -----
    log.info("Loading denisjooo …")
    dj_proc = AutoImageProcessor.from_pretrained(DENNISJOOO_REPO)
    dj_model = AutoModelForImageClassification.from_pretrained(DENNISJOOO_REPO).to(dev).eval()
    dj_classes = [dj_model.config.id2label[i] for i in range(len(dj_model.config.id2label))]
    dj_size = dj_proc.size["height"] if isinstance(dj_proc.size, dict) else 260
    dj_transform = transforms.Compose([
        transforms.Resize((dj_size, dj_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=dj_proc.image_mean, std=dj_proc.image_std),
    ])
    dj_to_canon: list[int] = []
    for cls in dj_classes:
        canon = HAND_ALIASES.get(_normalize_match(cls), _normalize_match(cls))
        if canon in canon_lookup and canon != "OTHER":
            dj_to_canon.append(canon_lookup[canon])
        else:
            dj_to_canon.append(other_idx)
    log.info("denisjooo NA-matched: %d / %d", sum(1 for x in dj_to_canon if x != other_idx), len(dj_classes))

    # ----- Evaluate both -----
    def _run(model_name, predict_fn, transform):
        log.info("Running %s …", model_name)
        ds = _ImageDataset(raw_data_dir, paths, labels, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        all_correct: list[int] = []
        all_correct_top5: list[int] = []
        cursor = 0
        for imgs, _ in tqdm(loader, desc=model_name):
            imgs = imgs.to(dev, non_blocking=True)
            with torch.no_grad():
                logits = predict_fn(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu().numpy()
            n_batch = preds.shape[0]
            tgt = labels_arr[cursor : cursor + n_batch]
            all_correct.extend((preds == tgt).astype(np.int8).tolist())
            for i in range(n_batch):
                all_correct_top5.append(int(tgt[i] in top5[i]))
            cursor += n_batch
        return np.asarray(all_correct, dtype=np.int8), np.asarray(all_correct_top5, dtype=np.int8)

    def _our_predict(imgs):
        feats = our_backbone(pixel_values=imgs).last_hidden_state[:, 0]
        return our_head(feats)

    def _dj_predict(imgs):
        logits = dj_model(pixel_values=imgs).logits   # (B, 525)
        # Remap to our 407-way: take max-prob class within each canonical bucket.
        B, _ = logits.shape
        out = torch.zeros((B, n_classes), device=logits.device)
        for src_idx in range(len(dj_classes)):
            dst = dj_to_canon[src_idx]
            torch.maximum(out[:, dst], logits[:, src_idx], out=out[:, dst])
        return out

    ours_c1, ours_c5 = _run("ours", _our_predict, our_transform)
    dj_c1, dj_c5 = _run("denisjooo", _dj_predict, dj_transform)

    # ----- Score per-source and overall -----
    def _metrics(correct: np.ndarray, correct5: np.ndarray, mask: np.ndarray) -> dict:
        sub = correct[mask]
        sub5 = correct5[mask]
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "top1": 0.0, "top1_ci": [0.0, 0.0], "top5": 0.0}
        lo, hi = _bootstrap_ci(sub)
        return {"n": n, "top1": float(sub.mean()), "top1_ci": [lo, hi],
                "top5": float(sub5.mean())}

    sources_set = sorted(set(sources))
    results: dict = {"ours": {}, "denisjooo": {}}
    sources_arr = np.asarray(sources)
    results["ours"]["overall"] = _metrics(ours_c1, ours_c5, np.ones(len(sources_arr), bool))
    results["denisjooo"]["overall"] = _metrics(dj_c1, dj_c5, np.ones(len(sources_arr), bool))
    for s in sources_set:
        mask = sources_arr == s
        results["ours"][s] = _metrics(ours_c1, ours_c5, mask)
        results["denisjooo"][s] = _metrics(dj_c1, dj_c5, mask)

    # ----- Print + write report -----
    log.info("=" * 60)
    log.info("Final apples-to-apples on our 407-way taxonomy:")

    def _fmt(m):
        return f"{m['top1']*100:.1f}% (CI {m['top1_ci'][0]*100:.1f}–{m['top1_ci'][1]*100:.1f})"

    log.info("  %-12s  ours %s   denisjooo %s", "OVERALL",
             _fmt(results["ours"]["overall"]), _fmt(results["denisjooo"]["overall"]))
    for s in sources_set:
        log.info("  %-12s  ours %s   denisjooo %s",
                 s, _fmt(results["ours"][s]), _fmt(results["denisjooo"][s]))

    # Markdown report.
    lines = [
        "# BENCHMARK.md\n",
        "Test split scored apples-to-apples in the **407-way canonical**",
        "taxonomy (406 NA species + OTHER). Both models map their native",
        "predictions through the alias table (gpiosenka's screaming-case,",
        "old-world common names like \"Rock Dove\", etc.) before scoring.\n",
        "## vs `dennisjooo/Birds-Classifier-EfficientNetB2`\n",
        "| Split | n | Ours top-1 | denisjooo top-1 | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["overall"] + sources_set:
        o = results["ours"][key]
        d = results["denisjooo"][key]
        delta = o["top1"] - d["top1"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| **{key}** | {o['n']:,} | "
            f"**{o['top1']*100:.1f}%** ({o['top1_ci'][0]*100:.1f}–{o['top1_ci'][1]*100:.1f}) "
            f"top-5 {o['top5']*100:.1f}% | "
            f"{d['top1']*100:.1f}% ({d['top1_ci'][0]*100:.1f}–{d['top1_ci'][1]*100:.1f}) "
            f"top-5 {d['top5']*100:.1f}% | "
            f"**{sign}{delta*100:.1f} pp** |"
        )
    lines.append("\n_95 % bootstrap CIs over 1,000 resamples._\n")
    out_md.write_text("\n".join(lines))
    log.info("Wrote %s", out_md)

    json_path = out_md.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
