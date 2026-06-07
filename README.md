# birdclass-na

A bird-species image classifier optimized for **North American backyard
and camera-trap conditions** — partial occlusion, motion blur, fence /
leaf clutter, low-light, and other things you don't see in handheld
photo datasets but you do see when a feeder camera is the photographer.

Status: **early development.** No model has been published yet.

## Why this exists

The best open-source bird classifiers on HuggingFace today are weekend
Kaggle efforts trained on a single dataset (e.g. gpiosenka's 525
species). They work OK on clean handheld photos but degrade sharply on
the kind of imagery a feeder camera actually produces. The proprietary
alternatives (Cornell's Merlin Bird ID, iNaturalist's CV) are
API-only or app-only — you can't run them locally, you can't fine-tune
them, you can't audit their training data.

The aim of this project is to fill a real gap: an Apache-2.0 bird
classifier that is

1. **Trained on a unified taxonomy** spanning gpiosenka 525, NABirds,
   and iNat21-Birds — ~3.8M images across ~1,500 NA-eligible species.
2. **Built on a modern foundation backbone** (DINOv2-B) rather than a
   2019 ConvNet.
3. **Domain-adapted** to real feeder-camera conditions via the yard
   data from the [BirdWatcher](https://github.com/houlette/BirdWatcher)
   project (~6,000 labeled crops, real false-positive distractors).
4. **Honest about its limits** — model card publishes benchmarks vs
   denisjooo's EfficientNet-B2, published NABirds SOTA, and Merlin on
   real-world yard crops.

If those benchmarks land in the right place, the goal is to be the
best open-source bird classifier on HuggingFace for NA backyard and
camera-trap use. Not best in absolute terms (Merlin and iNat have
orders of magnitude more data) — best in the *open* niche, which is
currently empty.

## Project structure

```
birdclass-na/
├── data/                       # dataset acquisition + manifest building
│   ├── download.py             # fetch each public dataset + yard data
│   ├── build_unified_taxonomy.py
│   └── manifest.py             # emit per-split CSVs
├── train/                      # training pipelines
│   ├── linear_probe.py         # Phase 2 gate (frozen DINOv2 + linear head)
│   └── finetune.py             # Phase 3 full fine-tune
├── eval/
│   └── run_benchmarks.py       # vs denisjooo / NABirds / Merlin
├── scripts/
│   └── publish.py              # push to HuggingFace Hub
├── docs/
│   └── blog_post.md
├── manifests/                  # gitignored — train/val/test CSVs
├── runs/                       # gitignored — training outputs + checkpoints
├── raw_data/                   # gitignored — downloaded datasets
├── pyproject.toml
└── LICENSE                     # Apache-2.0
```

## Quick start

```bash
# Install (editable so scripts can pick up local changes)
pip install -e ".[dev,yard]"

# Phase 1: build the dataset manifests (Mac, ~30 min including downloads
#         once you've accepted the dataset TOSes interactively)
python -m data.download --datasets gpiosenka,nabirds,inat21birds,yard
python -m data.build_unified_taxonomy
python -m data.manifest --out manifests/

# Phase 2: linear-probe gate (rented A100, ~$15)
python -m train.linear_probe --manifest manifests/ --out runs/probe/

# Phase 3 (only if Phase 2 passes its gate): full fine-tune (~$50)
python -m train.finetune --manifest manifests/ --resume-from runs/probe/

# Phase 4: benchmark harness
python -m eval.run_benchmarks --model runs/finetune/

# Phase 5: publish to HF Hub
python -m scripts.publish --model runs/finetune/ --hf-repo houlette/birdclass-na
```

## Acknowledgements

Training data sources:

- **gpiosenka 525 BIRD SPECIES** (Kaggle, CC0/CC-BY) — the broadest
  per-species coverage in any public bird classification dataset.
- **NABirds v1** (Cornell Lab, research use) — expert-labeled NA bird
  imagery with rich annotations.
- **iNat21-Birds** (CC-BY-NC subset of iNaturalist 2021 challenge) —
  the bird subset of the iNat21 fine-grained classification challenge.
- **BirdWatcher yard data** — ~6,000 user- and Claude-labeled crops
  from a private feeder-camera dataset, used only for the domain
  adaptation stage.

Backbone: [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base),
licensed Apache-2.0.

## License

Apache-2.0. See [LICENSE](./LICENSE).

Trained model weights are released separately under the same license,
subject to the *non-commercial* clause inherited from iNat21 — i.e.,
non-commercial use is unrestricted, commercial use needs you to
re-train without iNat21 in the mix.
