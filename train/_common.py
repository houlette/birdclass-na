"""Shared helpers for both train pipelines (linear_probe + finetune).

Things that genuinely live in both files: device selection, manifest
loading, class-weighted loss prep, AdamW + cosine LR setup.
"""
from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def device() -> torch.device:
    """Pick best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(csv_path: Path) -> tuple[list[str], list[int], list[str]]:
    """Returns (relative_paths, label_indices, source_datasets)."""
    paths, labels, sources = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["path"])
            labels.append(int(row["label_idx"]))
            sources.append(row["source_dataset"])
    return paths, labels, sources


def class_weights(labels: list[int], n_classes: int, device_: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy.

    A class appearing 1000x more than another gets ~1000x lower weight,
    so the loss doesn't get dominated by ``OTHER`` (which can easily be
    >40% of the training set).
    """
    counts = Counter(labels)
    weights = torch.zeros(n_classes, dtype=torch.float32, device=device_)
    total = sum(counts.values())
    for i in range(n_classes):
        n = counts.get(i, 1)  # guard against zero — class with no samples
        weights[i] = total / (n_classes * n)
    return weights
