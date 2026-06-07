"""Dataset acquisition + manifest building.

Each downloader produces a uniform on-disk layout under ``raw_data/<source>/``
and emits a per-source metadata JSON that ``manifest.py`` consumes to
build the unified train/val/test CSVs.
"""
