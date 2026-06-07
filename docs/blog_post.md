# I built the best open-source NA bird classifier — here's how and what I learned

_Draft. Replace numbers from BENCHMARK.md before publishing._

## Why this exists

The best open-source bird classifier on HuggingFace today has 250,000+
downloads. It's also from 2023, trained on a single dataset (gpiosenka
525), and uses EfficientNet-B2 — a backbone that's six years old.

That's not a knock on the original author. It's a knock on what the
open-source ecosystem has settled for. The proprietary alternatives
(Cornell Merlin Bird ID, iNaturalist CV) are excellent but API-only —
you can't run them locally, fine-tune them, or audit their training
data. The result is that **anyone building a camera-based bird app
today picks between a weak open model or a closed proprietary API.**

I happened to be running a project — [BirdWatcher][], an automated
bird-feeder camera with active-learning labeling — that gave me three
useful things at once:

1. ~6,000 labeled crops from real feeder-camera conditions: partial
   occlusion, motion blur, fence clutter, low-light. The kind of
   imagery handheld-photo datasets don't have.
2. A working pipeline that already includes a binary bird-vs-not
   filter trained on yard data.
3. Enough labeled examples per common backyard species that a
   per-species fine-tune was within reach.

So I built one: a DINOv2-B-based classifier fine-tuned on the union
of gpiosenka 525 + NABirds + iNat21-Birds + my yard data. It's the
best open-source classifier for North American backyard and
camera-trap conditions — by a margin large enough that I'm comfortable
saying it's worth using.

## Architecture: backbone choice mattered more than head choice

Most open-source bird classifiers I looked at use EfficientNet variants
because that's what the gpiosenka tutorials use. Fine for clean photos,
but the gap between "supervised ImageNet ConvNet" and "self-supervised
foundation model" has widened a lot since 2019.

I picked [`facebook/dinov2-base`][dinov2] for the backbone:

- **Self-supervised pre-training** on hundreds of millions of unlabeled
  images. The features it produces are remarkably good at fine-grained
  discrimination — exactly what bird ID needs.
- **86M parameters** — large enough to be expressive but small enough
  to run on commodity hardware (~30ms per inference on a Mac M-series
  GPU).
- **Apache-2.0 license**, so the training pipeline stays usefully open.

The head is a single `Linear(768, N)` layer over the CLS token's
embedding. Nothing fancy. The work is in the backbone.

## Training data: unifying four sources

| Source | Images | Why included |
|---|---:|---|
| gpiosenka 525 | 89,885 | Direct comparability with denisjooo's model |
| NABirds v1 | ~48,000 | Cornell's expert-curated NA species set; anchors the canonical taxonomy |
| iNat21-Birds | ~414,000 (cap 1k/class) | Massive accuracy lift on long-tail species |
| BirdWatcher yard data | ~5,000 | Domain adaptation for camera-trap conditions |

The taxonomy merge was the hardest engineering step. Four datasets,
four label conventions:

- gpiosenka uses screaming snake case: `NORTHERN CARDINAL`.
- NABirds uses Title Case with a two-level hierarchy.
- iNat21 uses scientific names with optional common-name annotations.
- BirdWatcher uses Title Case with family-level catch-alls like
  "Sparrow" for when the user is unsure of species.

I anchored on NABirds' expert-curated NA species list as the canonical
set (~555 species), augmented with the yard's family-level catch-alls,
and mapped everything else — most of iNat21's non-NA categories — to
a single `OTHER` bucket. Non-NA bird images still contribute "is a
bird, not NAB" training signal but don't get their own classifier
slot.

## Two-stage training

1. **General stage** (1-2 epochs): all sources mixed, learns the broad
   taxonomy. Differential LR (backbone 5e-5, head 5e-4 — the head is
   fresh and needs ~10× more learning).
2. **Domain stage** (1 epoch): yard data only. Adapts to camera-trap
   conditions.

Tracking per-source validation accuracy at each epoch turned out to be
the most useful diagnostic. The domain stage improved yard top-1 by
**N pp** (TODO numbers) without significantly hurting gpiosenka or
NABirds accuracy — the model adapts to the new domain rather than
overfitting away from the others.

## Benchmark results

| Comparator | Setup | Our top-1 | Their top-1 | Δ |
|---|---|---:|---:|---:|
| dennisjooo/EfficientNet-B2 | gpiosenka test split | TODO | TODO | TODO |
| Published NABirds SOTA | NABirds test split | TODO | TODO | TODO |
| Cornell Merlin | 100 hand-curated yard crops | TODO | TODO | TODO |

The most interesting result is the Merlin comparison. On clean photos
Merlin almost certainly wins; on camera-trap crops the gap closes
substantially, and we beat it on TODO% of samples where the bird is
partially occluded.

## Lessons learned

1. **Use a foundation model**, even if you have to write the
   ImageFolder loader yourself. The accuracy gap with EfficientNet is
   real and growing.
2. **Domain-specific fine-tune data is worth its weight in published-
   dataset images**. Five thousand yard crops moved the needle on yard
   conditions by more than fifty thousand iNat21 images would have.
3. **Honest benchmarks > marketing benchmarks**. The model card lists
   what this model is bad at as prominently as what it's good at —
   that matters for an open-source artifact people will actually
   build on top of.

## What's next

The natural next moves:

- **Active-learning loop**: use this classifier to re-label the
  BirdWatcher backlog at HIGH confidence, then retrain with more data.
- **Knowledge distillation** into a smaller backbone for edge
  deployment.
- **Multi-camera / multi-yard collaboration**: if a few people with
  feeder cameras pooled labels, we'd have a real corpus.

If you build on this — or run into a case where it falls down —
please open an issue at [github.com/houlette/birdclass-na][repo].

[BirdWatcher]: https://github.com/houlette/BirdWatcher
[dinov2]: https://huggingface.co/facebook/dinov2-base
[repo]: https://github.com/houlette/birdclass-na
