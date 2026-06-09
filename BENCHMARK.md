# BENCHMARK.md

Test set: **27,470 rows** held out from gpiosenka 525, NABirds, and the
BirdWatcher yard dataset. (No iNat21 test split — iNat21 only contributed
to train/val.)

All three models scored apples-to-apples in our **407-way canonical
taxonomy** (406 NA species + OTHER). Comparator outputs are mapped through
the same alias table — `Rock Dove` → `Rock Pigeon`,
`Cardinalis cardinalis` → `Northern Cardinal`, etc. — that our taxonomy
builder uses. denisjooo's 525-way logits and birder's 10,000-way logits
are max-pooled per canonical bucket; ours predict natively.

## Three-way: ours vs denisjooo vs birder-project

| Split | n | **Ours** | [denisjooo](https://huggingface.co/dennisjooo/Birds-Classifier-EfficientNetB2) | [birder-project](https://huggingface.co/birder-project/hieradet_d_small_dino-v2-inat21) |
|---|---:|---:|---:|---:|
| **overall** | 27,470 | **92.9%** (92.6–93.2) | 23.8% (23.3–24.4) | 89.6% (89.3–90.0) |
| gpiosenka | 2,625 | 89.0% (87.9–90.1) | 99.0% (98.7–99.4) | 85.3% (84.0–86.8) |
| nabirds | 24,633 | **93.3%** (93.0–93.6) | 15.9% (15.5–16.4) | 90.4% (90.0–90.7) |
| **yard** | 212 | **96.2%** (93.9–98.6) | 10.4% (6.6–14.6) | 57.1% (50.9–63.7) |

_Top-1 with 95 % bootstrap CIs over 1,000 resamples. Bold marks the best in
each row._

## How to read this

- **Overall**: we beat both alternatives. The +3.3 pp lead over birder
  is outside the CI overlap.
- **NABirds** (the cleanest NA-species test split, n=24,633): we beat
  birder by +2.9 pp on the source they'd be most expected to win.
- **Yard** (real feeder-camera crops with motion blur / partial
  occlusion / fence clutter, n=212): we beat birder by **+39.1 pp**.
  This is the validation of our "domain fine-tune on production yard
  data" thesis. Birder's iNat21-only training has no exposure to
  camera-trap conditions.
- **gpiosenka**: denisjooo wins (+10 pp over us) because gpiosenka's
  test split *is* its training data's holdout. We beat birder by +3.7 pp
  on this split despite the disadvantage.

## What this means for use cases

- **Best for backyard / feeder-camera / camera-trap conditions**: ours,
  by ~40 pp over the nearest competitor.
- **Best for clean handheld iNat-style photos**: birder is solid, especially
  if you also need plants / fungi / insects from the same model.
- **Best for the gpiosenka 525-species test specifically**: denisjooo
  (it was trained on those labels).
- **Best for "is this a bird I should care about?" with built-in NAB
  suppression**: ours (the `OTHER` class threshold gives a clean reject
  signal).
