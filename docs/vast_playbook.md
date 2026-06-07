# vast.ai playbook: rent → train → ship

A start-to-finish guide for running the training pipeline on a rented
GPU. Walks through account setup, picking an instance, transferring
data, running the linear probe gate, then (conditionally) the full
fine-tune. Estimated wall-clock: **~6-12 hours of GPU time** for the
full path; **~30-60 minutes of your time** spread across that window.

Estimated cost at vast.ai spot pricing (as of 2026): **$30-80 total**
for a single A100 40 GB. Linear probe alone is ~$10-15.

## 1. Account setup (one-time, ~10 min)

1. Create an account at https://cloud.vast.ai/
2. Verify your email
3. Add $50-100 in credits via Account → Billing → Add Credit. Stripe
   accepts credit cards.
4. **Generate an SSH key** at Account → Keys → Add Key. Paste in your
   `~/.ssh/id_ed25519.pub` (or generate a new pair if you don't have
   one). Vast.ai uses this for SSH access to your instances.
5. **Generate an API key** at Account → API Keys. Save it locally —
   we'll use it for the CLI:
   ```
   pip install vastai
   vastai set api-key YOUR_KEY
   ```

## 2. Pick an instance (~5 min)

Use the web UI at https://cloud.vast.ai/create/ — filter for:

- **GPU type**: A100 PCIE 40 GB (cheapest workable option) or A100
  SXM4 80 GB (faster but ~2× the price). H100 is overkill but
  sometimes spot-priced low.
- **Disk space**: at least 200 GB. The full dataset is ~80 GB and
  training caches add another ~10-20 GB.
- **Internet speed**: 1+ Gbps down for fast iNat21 download. Filter
  on "DLPerf > 100" as a rough proxy.
- **Reliability**: prefer instances with ≥ 99% reliability score.
  Spot instances can be reclaimed; for a fine-tune you want stability.
- **Docker image**: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
  (matches our pinned versions).

A reasonable filter: A100 40GB, ≥ 200 GB disk, 1+ Gbps, $0.50-1.50/hr.
You'll usually find 10-30 candidates.

When you find one, click **Rent**. Wait for the status to go from
"creating" → "running" (~1-2 min).

## 3. SSH in (~2 min)

vast.ai gives you SSH details under the instance's **Connect** menu.
It looks like:

```
ssh -p 12345 root@123.45.67.89 -L 8080:localhost:8080
```

That `-L` is for port-forwarding Jupyter if you want it; we don't need
it for our pipeline.

Copy that SSH command. Run it locally. You should land in
`/root/` on a CUDA-enabled Linux box.

## 4. Bootstrap the environment (~5 min)

From within the SSH session:

```bash
# Clone our repo
cd /workspace
git clone https://github.com/houlette/birdclass-na.git
cd birdclass-na

# Install deps. The base image already has torch + CUDA; we just
# add transformers, datasets, etc.
pip install -e ".[yard]"

# Confirm GPU is visible
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0))"
```

You should see: `cuda: True device: NVIDIA A100-PCIE-40GB` or similar.

## 5. Transfer the small datasets (~10 min, from your Mac)

The three smaller datasets — gpiosenka (1.8 GB), NABirds (3 GB), yard
(50 MB) — go up via rsync from your Mac. Open a **new terminal** on
your Mac (not the SSH one):

```bash
# Replace with your instance's actual SSH command details.
RSYNC_RSH="ssh -p 12345" rsync -a --info=progress2 \
    ~/Documents/Projects/birdclass-na/raw_data/{gpiosenka,nabirds,yard}/ \
    root@123.45.67.89:/workspace/birdclass-na/raw_data/

# About 5 GB total at residential upload speeds ≈ 5-15 min.
```

Alternative if you'd rather skip the upload: the gpiosenka and yard
downloaders work on the remote instance too. Run:

```bash
# Inside the SSH session:
python -m data.download --datasets gpiosenka,yard --raw-data-dir raw_data
```

(NABirds requires a tarball, so unless you can re-fetch the Cornell
download URL from your email, it's easier to rsync that one.)

## 6. Download iNat21 on the remote (~30-60 min)

vast.ai instances have multi-Gbps internet, so iNat21 downloads
**~10-20× faster** than on your Mac:

```bash
# Inside the SSH session:
python -m data.download --datasets inat21birds --raw-data-dir raw_data
```

Expect ~30-60 min for the full stream + filter. The script extracts
only Aves images on the fly, so the disk footprint stays at ~75 GB.

## 7. Build taxonomy + manifests (~30 sec)

```bash
python -m data.build_unified_taxonomy --raw-data-dir raw_data
python -m data.manifest --raw-data-dir raw_data --split probe \
    --probe-cap 200000 --out manifests/probe/
python -m data.manifest --raw-data-dir raw_data --split full \
    --out manifests/full/
```

## 8. Linear probe gate (~3-4 hours)

This is the cheap experiment — frozen DINOv2-B + linear head over
200k subsampled images:

```bash
mkdir -p runs/probe/
nohup python -m train.linear_probe \
    --manifest manifests/probe/ \
    --raw-data-dir raw_data \
    --out runs/probe/ \
    > runs/probe/log 2>&1 &
disown
```

Tail the log to watch progress:

```bash
tail -f runs/probe/log
```

You'll see two stages: feature extraction (~2 hr) then head training
(~30 min for 15 epochs since each epoch is just dot products).

**Gate decision**: open `runs/probe/metrics.json` when done. Compare
the gpiosenka per-source top-1 to denisjooo's published ~85%. We
need ≥ 90% to justify the full fine-tune.

If it fails: stop here. You've spent ~$15, document why, shelve.
If it passes: proceed to Phase 3.

## 9. Full fine-tune (~5-8 hours, only if Phase 2 gated through)

```bash
mkdir -p runs/finetune/
nohup python -m train.finetune \
    --manifest manifests/full/ \
    --raw-data-dir raw_data \
    --out runs/finetune/ \
    --resume-from runs/probe/ \
    --epochs-general 2 \
    --epochs-domain 1 \
    --batch-size 64 \
    > runs/finetune/log 2>&1 &
disown
```

Estimated: 3-4 hr per general epoch + 5-10 min per domain epoch ≈
6-8 hours total. Best-by-val checkpoint saved to `runs/finetune/model.pt`.

## 10. Pull the trained model back (~5 min)

From your Mac (a new terminal again):

```bash
RSYNC_RSH="ssh -p 12345" rsync -a --info=progress2 \
    root@123.45.67.89:/workspace/birdclass-na/runs/ \
    ~/Documents/Projects/birdclass-na/runs/
```

The fine-tune checkpoint is ~350 MB; pulls in 1-2 min.

## 11. Destroy the instance (~1 min)

Crucial: **vast.ai keeps billing until you destroy the instance**.

In the web UI: go to your instances list, click the **trash can**
icon, confirm.

## 12. Run benchmarks locally (~30 min on your Mac)

You don't need GPU for the benchmark harness — it's CPU-tolerable on
the modest dataset sizes involved:

```bash
# From your Mac
cd ~/Documents/Projects/birdclass-na

# We need raw_data to score against. Pull the small parts back from
# vast.ai before destroying it, or use what's already local.
python -m eval.run_benchmarks \
    --model runs/finetune/ \
    --manifest manifests/full/ \
    --raw-data-dir raw_data \
    --out BENCHMARK.md \
    --skip-merlin   # add --merlin-csv after you've done the 100-crop captures
```

## 13. Publish to HF Hub (~5 min)

After the Merlin captures + BENCHMARK.md is updated:

```bash
# Get a write token from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_xxx
python -m scripts.publish \
    --model runs/finetune/ \
    --hf-repo houlette/birdclass-na \
    --benchmark BENCHMARK.md
```

## Total cost summary

| Phase | Wall-clock | A100 cost (at $0.80/hr spot) |
|---|---:|---:|
| iNat21 download on remote | 30-60 min | ~$1 |
| Linear probe gate | 3-4 hr | $3-4 |
| Full fine-tune | 6-8 hr | $5-7 |
| Buffer for restarts | 2 hr | $2 |
| **Total** | **~12-15 hr** | **~$10-15** |

Comfortably inside the $50-100 budget. The actual money risk is
**forgetting to destroy the instance** — set a phone alarm if you
walk away from the SSH session.

## Common failure modes

- **Instance preempted mid-train**: the linear probe is short enough
  to just restart from scratch; the fine-tune saves best-by-val
  every epoch so worst-case restart loses < 1 epoch.
- **Disk full during iNat21**: the per-class cap of 1000 keeps disk
  usage bounded, but if the transient stream cache fills up, drop
  `_cache/` between attempts.
- **CUDA OOM on full fine-tune**: drop `--batch-size` to 32 and add
  `--grad-accum-steps 2` to keep effective batch at 64.
- **HF token rejected during publish**: regenerate at
  https://huggingface.co/settings/tokens with `Write` scope, not
  just `Read`.

## Open questions to think about while it runs

- Should we publish the model under `houlette/birdclass-na` (matches
  the repo) or a more descriptive name like
  `houlette/birdclass-na-dinov2-base`?
- Do we want to publish the intermediate checkpoints (general-stage
  only, vs general-then-domain) so users can pick which behavior they
  want?
