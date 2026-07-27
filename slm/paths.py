"""Filesystem locations for the project, resolved relative to the repo root.

Everything is derived from this file's own location (``__file__``) so paths are
independent of the current working directory — ``uv run main.py`` works the same
from anywhere.

The default tokenizer lives in ``data/`` rather than a ``tokenizers/`` directory: the repo
root is on ``sys.path``, so that name would shadow the installed ``tokenizers`` package on
every import. ``data/`` is gitignored *except* for ``*.json``, since a checkpoint is
unusable without the tokenizer that produced its ids.
"""
import os
import sys
import warnings
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # .../slm/slm
REPO_ROOT = PACKAGE_DIR.parent                         # .../slm

TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer-32k.json"
SIMPLE_TOKENIZER_PATH = REPO_ROOT / "notebooks" / "tokenizer.json"
DATA_DIR = REPO_ROOT / "data"                          # cached encoded corpora (*.npy)
CKPT_DIR = REPO_ROOT / "checkpoints"                   # model checkpoints (*.pt)

# Machine-specific HuggingFace cache, feeding both HF_HOME below and
# TrainConfig.hf_cache_dir. Deferring to a pre-set HF_HOME keeps those two in agreement.
HF_CACHE_DIR = os.environ.get("HF_HOME", "/data/zejiaqi/huggingface-cache")

os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", str(Path(HF_CACHE_DIR) / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(Path(HF_CACHE_DIR) / "xet"))

if "huggingface_hub" in sys.modules:            # too late to matter — say so
    warnings.warn(
        f"huggingface_hub was imported before slm.paths, so it resolved its cache from the "
        f"environment and not {HF_CACHE_DIR} — downloads will land in ~/.cache. Import slm "
        f"(or slm.paths) first.", RuntimeWarning, stacklevel=2)
