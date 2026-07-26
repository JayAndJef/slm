"""Filesystem locations for the project, resolved relative to the repo root.

Everything is derived from this file's own location (``__file__``) so paths are
independent of the current working directory — ``uv run main.py`` works the same
from anywhere. The trained tokenizer deliberately stays under ``notebooks/`` so the
(untouched) notebook keeps loading it from the same place.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # .../slm/slm
REPO_ROOT = PACKAGE_DIR.parent                         # .../slm

TOKENIZER_PATH = REPO_ROOT / "notebooks" / "tokenizer.json"
DATA_DIR = REPO_ROOT / "data"                          # cached encoded corpora (*.npy)
CKPT_DIR = REPO_ROOT / "checkpoints"                   # model checkpoints (*.pt)

# Machine-specific HuggingFace cache; override via TrainConfig / CLI if needed.
HF_CACHE_DIR = "/data/zejiaqi/huggingface-cache"
