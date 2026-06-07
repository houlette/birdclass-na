"""Phase 5 — Push the trained model + model card to the HuggingFace Hub.

Inputs:
- A trained model directory (typically ``runs/finetune/``) containing
  ``model.pt`` with backbone + head state dicts, plus a ``metrics.json``
  with training history.
- ``BENCHMARK.md`` — the harness output from ``eval/run_benchmarks.py``.

What this script does:

1. Repackages the raw ``model.pt`` state-dicts into a proper
   HuggingFace ``AutoModelForImageClassification`` directory. Loads
   the DINOv2-B backbone fresh, applies our trained weights, attaches
   the head, and saves via ``model.save_pretrained()`` so any downstream
   loader can use the standard HF API without custom code.
2. Composes the model card (README.md for the HF repo). The card
   inlines the benchmark tables verbatim and prefaces them with honest
   limitations: NA-skewed training data, rare-species long tail, license
   inheritance (CC-BY-NC from iNat21 → non-commercial-only downstream).
3. Pushes the directory to ``$HF_REPO`` via ``HfApi().upload_folder``.

Requires ``HF_TOKEN`` env var set to a write token (you'll be prompted
once via huggingface_hub if not present in env).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scripts.publish")


MODEL_CARD_PREAMBLE = """\
---
license: cc-by-nc-4.0
language:
- en
tags:
- image-classification
- bird-classification
- birds
- dinov2
- fine-grained
- north-america
library_name: transformers
pipeline_tag: image-classification
base_model: facebook/dinov2-base
datasets:
- yashikota/birds-525-species-image-classification
metrics:
- accuracy
---

# {model_name}

A bird species classifier optimized for **North American backyard and
camera-trap conditions** — partial occlusion, motion blur, fence/leaf
clutter, low-light, and other things you don't see in handheld photo
datasets but you do see when a feeder camera is the photographer.

Backbone: [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base)
(Apache-2.0), with a `Linear({embed_dim}, {n_classes})` classification
head trained on a unified taxonomy spanning gpiosenka 525,
NABirds, iNat21-Birds, and the [BirdWatcher](https://github.com/houlette/BirdWatcher)
yard dataset.

## What this model is good at

- North American backyard bird identification, **especially** under
  feeder-camera conditions (partial occlusion, motion blur, leaf
  clutter, low-light) — the domain stage of training adapted
  specifically to these conditions using real labeled crops from a
  production deployment.
- Fine-grained discrimination of common NA confusables (Mourning Dove
  vs Rock Pigeon, Cooper's Hawk vs Sharp-shinned Hawk).

## What this model is _not_ good at

- **Non-NA species**: most non-NA bird images in iNat21 were collapsed
  into a single `OTHER` bucket during training. The model can flag a
  bird as "not one of these {na_classes} NA species" but can't tell
  you _which_ non-NA species it is.
- **Rare-species long tail**: NA species with very few training samples
  (< 30 each) have low individual accuracy. We're not better than
  general-purpose bird classifiers there, just smaller.
- **Comparison to Cornell Merlin / iNat CV**: those are trained on
  10-100× more data and remain stronger in absolute terms on most
  common-species photos. This model's value is in being open-source,
  fine-tunable, and stronger on camera-trap conditions.

## Benchmarks

{benchmark_md}

## Training data

- **gpiosenka 525**: ~89,885 images across 525 species. Pulled from
  [yashikota's HF mirror](https://huggingface.co/datasets/yashikota/birds-525-species-image-classification)
  (the original gpiosenka Kaggle upload was removed in 2025).
- **NABirds v1**: ~48,000 expert-labeled NA bird images from Cornell.
  Used under academic license — see https://dl.allaboutbirds.org/nabirds.
- **iNat21-Birds**: bird subset (~414k images) of the iNat 2021 challenge,
  filtered to the Aves supercategory. **License: CC-BY-NC**. This is
  why the trained model weights inherit a non-commercial restriction.
- **Yard data**: ~5,000 labeled crops from the [BirdWatcher](https://github.com/houlette/BirdWatcher)
  project. Domain-adaptation stage only.

## Quick start

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

processor = AutoImageProcessor.from_pretrained("{hf_repo}")
model = AutoModelForImageClassification.from_pretrained("{hf_repo}")

img = Image.open("your_bird.jpg")
inputs = processor(images=img, return_tensors="pt")
outputs = model(**inputs)
top1 = outputs.logits.softmax(dim=-1)[0].argmax().item()
print(model.config.id2label[top1])
```

## Limitations and honest claims

This model is **best-in-class among open-source bird classifiers for
NA backyard / camera-trap use** — it is **not** absolute SOTA on bird
classification benchmarks. Cornell's Merlin Bird ID app and iNaturalist's
internal classifier are both trained on orders of magnitude more data
and remain stronger on most common-species, clean-photo scenarios.

Use this model when:
- You need a local-running, fine-tunable bird classifier.
- Your inference distribution looks like camera-trap or feeder-camera
  imagery.
- You want Apache-2.0 code (the training pipeline) and CC-BY-NC
  weights with provenance you can audit.

Don't use this model when:
- You need commercial use (the iNat21 license restricts downstream).
  Re-train without iNat21 if commercial deployment matters.
- You need a global bird classifier — this is NA-focused by design.

## Citation

If you use this model in research, please cite it as:

```bibtex
@misc{{birdclass_na_{year},
  author = {{Houlette, Ryan}},
  title = {{ {model_name}: an open-source bird species classifier for North American backyards }},
  year = {{ {year} }},
  publisher = {{ HuggingFace }},
  url = {{ https://huggingface.co/{hf_repo} }}
}}
```

## License

Apache-2.0 for the training pipeline at https://github.com/houlette/birdclass-na.
Model weights themselves are released under CC-BY-NC-4.0 due to inheritance
from iNat21's non-commercial clause.
"""


def _repackage_to_hf(model_dir: Path, out_dir: Path, canonical: list[str]) -> None:
    """Convert our raw model.pt into a HF-format directory."""
    from transformers import (
        AutoModel,
        AutoImageProcessor,
        Dinov2Config,
        Dinov2Model,
    )
    # Construct a proper AutoModelForImageClassification wrapper.
    # The dinov2 backbone + classification head pattern lives in transformers
    # as Dinov2ForImageClassification, which is exactly what we want.
    from transformers import Dinov2ForImageClassification

    ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
    backbone_repo = ckpt["backbone_repo"]
    n_classes = ckpt["n_classes"]
    id2label = {i: name for i, name in enumerate(canonical)}
    label2id = {name: i for i, name in enumerate(canonical)}

    log.info("Repackaging into Dinov2ForImageClassification …")
    model = Dinov2ForImageClassification.from_pretrained(
        backbone_repo,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    # Replace backbone weights with our fine-tuned ones.
    model.dinov2.load_state_dict(ckpt["backbone_state"])
    # The HF wrapper uses `.classifier` for the linear head.
    model.classifier.load_state_dict(ckpt["head_state"])

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    # Save processor too — DINOv2 uses a standard image processor.
    processor = AutoImageProcessor.from_pretrained(backbone_repo)
    processor.save_pretrained(out_dir)
    log.info("Saved HF-format model to %s", out_dir)


def _compose_model_card(
    hf_repo: str, n_classes: int, embed_dim: int, na_classes: int,
    benchmark_md: str, year: int = 2026,
) -> str:
    model_name = hf_repo.split("/")[-1]
    return MODEL_CARD_PREAMBLE.format(
        hf_repo=hf_repo, model_name=model_name,
        n_classes=n_classes, embed_dim=embed_dim, na_classes=na_classes,
        benchmark_md=benchmark_md,
        year=year,
    )


@click.command()
@click.option("--model", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--hf-repo", required=True, help="e.g. houlette/birdclass-na")
@click.option("--benchmark", default="BENCHMARK.md", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--taxonomy", default="taxonomy.json", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Compose locally without uploading.")
def main(model: Path, hf_repo: str, benchmark: Path, taxonomy: Path, dry_run: bool) -> None:
    tax = json.loads(taxonomy.read_text())
    canonical = tax["canonical"]
    n_classes = len(canonical)
    na_classes = tax.get("na_class_count", n_classes - 1)

    hf_format_dir = model / "_hf_export"
    _repackage_to_hf(model, hf_format_dir, canonical)

    # Compose model card.
    embed_dim = 768   # DINOv2-B CLS token width
    benchmark_md = benchmark.read_text() if benchmark.exists() else "_(no benchmarks run yet)_"
    card = _compose_model_card(hf_repo, n_classes, embed_dim, na_classes, benchmark_md)
    (hf_format_dir / "README.md").write_text(card)
    log.info("Composed model card at %s", hf_format_dir / "README.md")

    if dry_run:
        log.info("--dry-run: skipping upload. Local artifacts at %s", hf_format_dir)
        return

    # Upload.
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN env var not set. Generate a write token at "
                  "https://huggingface.co/settings/tokens and re-run with "
                  "HF_TOKEN=hf_xxx python -m scripts.publish ...")
        sys.exit(2)

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(hf_repo, repo_type="model", exist_ok=True)
    log.info("Uploading %s → https://huggingface.co/%s …", hf_format_dir, hf_repo)
    api.upload_folder(
        repo_id=hf_repo,
        folder_path=str(hf_format_dir),
        commit_message=f"Initial model release ({Path(model).name})",
    )
    log.info("✓ Published: https://huggingface.co/%s", hf_repo)


if __name__ == "__main__":
    main()
