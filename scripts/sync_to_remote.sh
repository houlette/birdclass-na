#!/usr/bin/env bash
# Local helper: sync the small datasets from your Mac to a vast.ai instance.
#
# Usage:
#   scripts/sync_to_remote.sh <ssh_user@host> <ssh_port> [<remote_repo_path>]
#
# Example:
#   scripts/sync_to_remote.sh root@123.45.67.89 12345 /workspace/birdclass-na
#
# Skips iNat21 entirely (huge; re-download on the remote with its faster
# bandwidth instead). Pulls the trained-model artifacts back at the end
# if you ran the script with --pull-back.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <ssh_user@host> <ssh_port> [<remote_repo_path>] [--pull-back]" >&2
    exit 2
fi

REMOTE_HOST="$1"
REMOTE_PORT="$2"
REMOTE_REPO="${3:-/workspace/birdclass-na}"
PULL_BACK="${4:-}"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"

RSYNC_RSH="ssh -p $REMOTE_PORT"

echo "Pushing small datasets to $REMOTE_HOST:$REMOTE_REPO/raw_data/ …"
RSYNC_RSH="$RSYNC_RSH" rsync -a --info=progress2 \
    "$LOCAL_REPO/raw_data/gpiosenka/" \
    "$REMOTE_HOST:$REMOTE_REPO/raw_data/gpiosenka/"
RSYNC_RSH="$RSYNC_RSH" rsync -a --info=progress2 \
    "$LOCAL_REPO/raw_data/nabirds/" \
    "$REMOTE_HOST:$REMOTE_REPO/raw_data/nabirds/"
RSYNC_RSH="$RSYNC_RSH" rsync -a --info=progress2 \
    "$LOCAL_REPO/raw_data/yard/" \
    "$REMOTE_HOST:$REMOTE_REPO/raw_data/yard/"

echo "Pushed gpiosenka + nabirds + yard. Next on the remote:"
echo "  cd $REMOTE_REPO"
echo "  python -m data.download --datasets inat21birds --raw-data-dir raw_data"
echo "  python -m data.build_unified_taxonomy --raw-data-dir raw_data"
echo "  python -m data.manifest --raw-data-dir raw_data --split probe --probe-cap 200000 --out manifests/probe/"
echo "  python -m train.linear_probe --manifest manifests/probe/ --raw-data-dir raw_data --out runs/probe/"

if [ "$PULL_BACK" = "--pull-back" ]; then
    echo ""
    echo "Waiting for runs/ on the remote. Press Ctrl-C if you want to pull back later."
    read -p "Press Enter when training has finished to pull runs/ back … "
    RSYNC_RSH="$RSYNC_RSH" rsync -a --info=progress2 \
        "$REMOTE_HOST:$REMOTE_REPO/runs/" \
        "$LOCAL_REPO/runs/"
    echo "Pulled. Don't forget to destroy the vast.ai instance."
fi
