"""Phase 5 — Push the trained model + model card to the HuggingFace Hub.

Reads:
- A fine-tuned model directory (containing config.json, model.safetensors,
  preprocessor_config.json, training_meta.json).
- The latest ``BENCHMARK.md`` (inlined into the model card).
- A hand-written model-card preface (training data + license details +
  known limitations).

Composes a model card honest about limitations:
- NA-skewed training data.
- Long-tail rare species have far fewer samples (top-1 ≤ 50%).
- License inherits iNat21's CC-BY-NC for the bird subset, so the model
  is "non-commercial use only" downstream.

Requires the ``HF_TOKEN`` env var set to a write token.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--model", required=True, help="Trained-model directory.")
@click.option("--hf-repo", required=True, help="e.g. houlette/birdclass-na")
@click.option("--benchmark", default="BENCHMARK.md")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Compose the model card locally without pushing.",
)
def main(model: str, hf_repo: str, benchmark: str, dry_run: bool) -> None:
    raise NotImplementedError("Phase 5 task #7 — implement HF Hub push.")


if __name__ == "__main__":
    main()
