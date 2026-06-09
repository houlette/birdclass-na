"""Apples-to-apples baseline: run birder-project's
`hieradet_d_small_dino-v2-inat21` through our 407-way NA taxonomy.

Their model predicts in iNat21's 10,000-class space using full
taxonomic paths (`<idx>_<Kingdom>_<Phylum>_<Class>_<Order>_<Family>_
<Genus>_<species>`). To make their numbers comparable to ours, we:

1. Parse each of their 10k class names into a scientific binomial.
2. Look up the binomial → common-name via iNat21's own categories.json.
3. Apply our build_unified_taxonomy._normalize_match + HAND_ALIASES.
4. If the resolved common name is in our 406 NA species, predict that
   canonical index. Otherwise predict OTHER (~9,600 of their 10k classes
   collapse into OTHER, mostly plants / fungi / insects / fish / etc).

Outputs BENCHMARK_birder.md and BENCHMARK_birder.json with per-source
top-1 and 95 % bootstrap CIs, mirroring eval/final_compare.py.

Usage:
    python -m eval.baseline_birder \\
        --manifest manifests/full/test.csv \\
        --raw-data-dir raw_data \\
        --taxonomy taxonomy.json \\
        --inat21-train-json /tmp/train.json
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
log = logging.getLogger("eval.baseline_birder")

BIRDER_MODEL = "hieradet_d_small_dino-v2-inat21"

# Mirror of build_unified_taxonomy.HAND_ALIASES so European↔NA common-
# name pairs (Rock Dove → Rock Pigeon etc.) get aligned consistently.
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


def _parse_birder_class_to_binomial(class_label: str) -> str | None:
    """`00001_Animalia_Chordata_Aves_…_Cardinalis_cardinalis` → 'Cardinalis cardinalis'.

    The genus + species are the last two underscore-separated tokens.
    Returns None for non-bird classes (filter on `_Aves_` substring).
    """
    if "_Aves_" not in class_label:
        return None
    parts = class_label.split("_")
    if len(parts) < 3:
        return None
    return f"{parts[-2]} {parts[-1]}"


class _ImageDataset(Dataset):
    def __init__(self, raw_data_dir: Path, paths: list[str], labels: list[int], transform):
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
            img = Image.new("RGB", (384, 384))
        return self.transform(img), self.labels[idx]


def _bootstrap_ci(correct: np.ndarray, n: int = 1000) -> tuple[float, float]:
    if len(correct) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed=42)
    means = np.empty(n, dtype=np.float32)
    for i in range(n):
        idx = rng.integers(0, len(correct), len(correct))
        means[i] = correct[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@click.command()
@click.option("--manifest", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--raw-data-dir", default="raw_data", type=click.Path(file_okay=False, path_type=Path))
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--inat21-train-json", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Path to iNat21's extracted train.json (for sci → common-name).")
@click.option("--out-md", default="BENCHMARK_birder.md", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--batch-size", default=32, show_default=True)
@click.option("--num-workers", default=4, show_default=True)
def main(manifest: Path, raw_data_dir: Path, taxonomy: Path, inat21_train_json: Path,
         out_md: Path, batch_size: int, num_workers: int) -> None:
    import birder

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    log.info("Device: %s", dev)

    # ----- Load taxonomy + manifest -----
    tax = json.loads(taxonomy.read_text())
    canonical = tax["canonical"]
    canon_lookup = {n: i for i, n in enumerate(canonical)}
    other_idx = tax["other_index"]
    n_classes = len(canonical)

    paths, labels, sources = load_manifest(manifest)
    labels_arr = np.asarray(labels, dtype=np.int64)
    log.info("Test set: %d rows  canonical: %d classes", len(paths), n_classes)

    # ----- iNat21 sci → common -----
    log.info("Loading iNat21 categories from %s …", inat21_train_json)
    inat = json.loads(inat21_train_json.read_text())
    sci_to_common: dict[str, str] = {}
    for c in inat["categories"]:
        if c.get("supercategory") == "Birds":
            sci_to_common[c["name"]] = c.get("common_name", "")
    log.info("iNat21 bird sci→common map: %d entries", len(sci_to_common))

    # ----- Load birder model -----
    log.info("Loading birder model: %s …", BIRDER_MODEL)
    net, info = birder.load_pretrained_model(BIRDER_MODEL, inference=True)
    net.to(dev).eval()
    img_size = info.signature["inputs"][0]["data_shape"][-1]
    log.info("birder input size: %d×%d  rgb_stats: %s", img_size, img_size, info.rgb_stats)
    birder_classes = sorted(info.class_to_idx.keys(), key=lambda k: info.class_to_idx[k])
    log.info("birder classes: %d", len(birder_classes))

    # ----- Build birder-idx → our-canonical-idx mapping -----
    birder_to_canon: list[int] = []
    n_bird = 0
    n_na_match = 0
    for cls in birder_classes:
        binomial = _parse_birder_class_to_binomial(cls)
        if binomial is None:
            birder_to_canon.append(other_idx)
            continue
        n_bird += 1
        common = sci_to_common.get(binomial, "")
        if not common:
            birder_to_canon.append(other_idx)
            continue
        canon = _normalize_match(common)
        canon = HAND_ALIASES.get(canon, canon)
        if canon in canon_lookup and canon != "OTHER":
            birder_to_canon.append(canon_lookup[canon])
            n_na_match += 1
        else:
            birder_to_canon.append(other_idx)
    log.info("birder taxonomy alignment: %d bird classes, %d NA-matched into our taxonomy",
             n_bird, n_na_match)

    # ----- Transform matching birder's RGB stats -----
    mean = info.rgb_stats["mean"]
    std = info.rgb_stats["std"]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])

    # ----- Run inference -----
    log.info("Running birder on %d test images …", len(paths))
    ds = _ImageDataset(raw_data_dir, paths, labels, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    correct1: list[int] = []
    correct5: list[int] = []
    birder_to_canon_t = torch.tensor(birder_to_canon, device=dev, dtype=torch.long)
    cursor = 0
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="birder"):
            imgs = imgs.to(dev, non_blocking=True)
            out = net(imgs)
            # Birder may return (logits, ...) or just logits; handle both.
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            # Max-pool birder's 10k logits into our 407 canonical buckets.
            # For each canonical bucket k, take the max over all birder
            # classes that map to k. This is the same trick we used for
            # denisjooo and reflects "the model's top guess among classes
            # that resolve to canonical k."
            B = logits.shape[0]
            our_logits = torch.full((B, n_classes), -1e9, device=dev)
            for src in range(len(birder_classes)):
                dst = birder_to_canon[src]
                torch.maximum(our_logits[:, dst], logits[:, src], out=our_logits[:, dst])
            preds = our_logits.argmax(dim=1).cpu().numpy()
            top5 = our_logits.topk(min(5, n_classes), dim=1).indices.cpu().numpy()
            tgt = labels_arr[cursor : cursor + B]
            correct1.extend((preds == tgt).astype(np.int8).tolist())
            for i in range(B):
                correct5.append(int(tgt[i] in top5[i]))
            cursor += B

    c1 = np.asarray(correct1, dtype=np.int8)
    c5 = np.asarray(correct5, dtype=np.int8)
    sources_arr = np.asarray(sources)

    def _metrics(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "top1": 0.0, "top1_ci": [0.0, 0.0], "top5": 0.0}
        sub = c1[mask]
        sub5 = c5[mask]
        lo, hi = _bootstrap_ci(sub)
        return {"n": n, "top1": float(sub.mean()),
                "top1_ci": [lo, hi], "top5": float(sub5.mean())}

    results: dict = {"overall": _metrics(np.ones(len(sources_arr), bool))}
    for s in sorted(set(sources)):
        results[s] = _metrics(sources_arr == s)

    log.info("=" * 60)
    log.info("birder on our 407-way taxonomy:")
    for k, m in results.items():
        log.info("  %-12s  n=%-6d  top-1=%.3f (%.3f–%.3f)  top-5=%.3f",
                 k, m["n"], m["top1"], m["top1_ci"][0], m["top1_ci"][1], m["top5"])

    # Markdown report.
    lines = [
        f"# BENCHMARK_birder.md\n",
        f"`birder-project/{BIRDER_MODEL}` evaluated on our 407-way",
        "NA-focused canonical taxonomy. Their 10,000-class logits are",
        "max-pooled into our 407 canonical buckets via the same alias",
        "table used for the denisjooo baseline.\n",
        "| Split | n | Top-1 | Top-5 |",
        "|---|---:|---:|---:|",
    ]
    for k, m in results.items():
        lines.append(
            f"| **{k}** | {m['n']:,} | "
            f"{m['top1']*100:.1f}% ({m['top1_ci'][0]*100:.1f}–{m['top1_ci'][1]*100:.1f}) | "
            f"{m['top5']*100:.1f}% |"
        )
    out_md.write_text("\n".join(lines))
    out_md.with_suffix(".json").write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_md)


if __name__ == "__main__":
    main()
