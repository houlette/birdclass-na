"""Training pipelines.

Two entry points:

- ``linear_probe.py`` — Phase 2 gate (frozen DINOv2 backbone, linear head).
- ``finetune.py`` — Phase 3 full fine-tune (unfrozen, differential LR).
"""
