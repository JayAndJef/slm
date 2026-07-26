"""slm — a from-scratch small language model (byte-level BPE + GPT-style transformer).

Public API, organized by concern:

- config    : :class:`ModelConfig`, :class:`TrainConfig`, :func:`default_configs`
- tokenizer : :class:`SimpleTokenizer`
- model     : :class:`JLM`
- data      : :func:`load_docs`, :func:`build_corpus`, :func:`get_batch`
- train     : :func:`train` (Lightning driver); :mod:`slm.lit` has LitJLM/SLMDataModule
- generate  : :func:`generate`, :func:`load_model`

See ``main.py`` at the repo root for the CLI entrypoint.
"""
from slm import paths
from slm.config import ModelConfig, TrainConfig, default_configs
from slm.data import build_corpus, get_batch, load_docs
from slm.generate import generate, load_model
from slm.model import JLM
from slm.tokenizer import SimpleTokenizer
from slm.train import train

__all__ = [
    "paths",
    "ModelConfig", "TrainConfig", "default_configs",
    "SimpleTokenizer",
    "JLM",
    "load_docs", "build_corpus", "get_batch",
    "train",
    "generate", "load_model",
]
