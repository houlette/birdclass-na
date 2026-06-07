#!/usr/bin/env bash
# Bootstraps a fresh vast.ai instance after the user clones the repo.
#
# Run from inside the SSH session:
#   cd /workspace/birdclass-na
#   bash scripts/remote_bootstrap.sh
#
# Idempotent. Re-running is safe.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
echo "Bootstrapping in $REPO_ROOT …"

# ----- 1. Install Python deps -----
# vast.ai PyTorch images already include torch + CUDA. We just add
# the rest.
echo "Installing Python dependencies …"
pip install --quiet -e ".[yard]"

# ----- 2. Verify CUDA + GPU -----
echo ""
echo "GPU check:"
python - <<'PY'
import torch
print(f"  torch: {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0: {torch.cuda.get_device_name(0)}")
    print(f"  memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
else:
    print("  WARNING: CUDA not available. Training will be very slow.")
PY

# ----- 3. Pre-download model weights -----
# Pre-fetch DINOv2-B + denisjooo so subsequent training/benchmark scripts
# don't spend the first 1-2 minutes downloading them under throttled
# conditions.
echo ""
echo "Pre-downloading model weights …"
python - <<'PY'
from transformers import AutoModel, AutoImageProcessor
AutoImageProcessor.from_pretrained("facebook/dinov2-base")
AutoModel.from_pretrained("facebook/dinov2-base")
print("  DINOv2-B cached.")
# denisjooo for benchmark; comment out if you skip the head-to-head
from transformers import AutoModelForImageClassification
AutoImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
AutoModelForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
print("  denisjooo cached (for benchmark harness).")
PY

# ----- 4. Print next steps -----
echo ""
echo "==================================================="
echo "Bootstrap complete. Next steps:"
echo ""
echo "If raw_data/ wasn't rsynced up, run the downloaders:"
echo "  python -m data.download --datasets gpiosenka,inat21birds --raw-data-dir raw_data"
echo "  (NABirds requires a tarball, easier to rsync from your Mac)"
echo ""
echo "Once data is in place:"
echo "  python -m data.build_unified_taxonomy --raw-data-dir raw_data"
echo "  python -m data.manifest --raw-data-dir raw_data --split probe --probe-cap 200000 --out manifests/probe/"
echo "  python -m data.manifest --raw-data-dir raw_data --split full --out manifests/full/"
echo ""
echo "Linear probe (Phase 2 gate, ~3-4 hr):"
echo "  mkdir -p runs/probe/ && nohup python -m train.linear_probe \\"
echo "      --manifest manifests/probe/ --raw-data-dir raw_data --out runs/probe/ \\"
echo "      > runs/probe/log 2>&1 & disown"
echo "  tail -f runs/probe/log"
echo ""
echo "Full fine-tune (Phase 3, only if probe gate passes; ~6-8 hr):"
echo "  mkdir -p runs/finetune/ && nohup python -m train.finetune \\"
echo "      --manifest manifests/full/ --raw-data-dir raw_data --out runs/finetune/ \\"
echo "      --resume-from runs/probe/ > runs/finetune/log 2>&1 & disown"
echo ""
echo "After training, pull runs/ back to your Mac with:"
echo "  scripts/sync_to_remote.sh <user@host> <port> /workspace/birdclass-na --pull-back"
echo ""
echo "DON'T FORGET to destroy the vast.ai instance when finished."
echo "==================================================="
