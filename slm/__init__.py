"""slm — a from-scratch small language model (byte-level BPE + GPT-style transformer).

Public API, organized by concern:

- config    : :class:`ModelConfig`, :class:`TrainConfig`, :func:`default_configs`
- tokenizer : :func:`load_tokenizer` (picks a backend from the file), :class:`HFTokenizer`
              (rust, the default), :class:`SimpleTokenizer` (from scratch, the fallback)
- model     : :class:`JLM`
- corpus    : :func:`load_docs`, :func:`locate`, :func:`build`, :class:`Corpus`
- render    : :class:`Renderer`, :func:`build_renderer` (records -> tokens, per objective)
- checkpoint: the compact ``{model, step, val, config}`` format — save/load
- train     : :func:`train` (Lightning driver); :mod:`slm.lit` has LitJLM/SLMDataModule
- generate  : :func:`generate` (whole string), :func:`stream` (incremental), :func:`load_model`

See ``main.py`` at the repo root for the CLI entrypoint.
"""
from slm import chat, checkpoint, paths
from slm.config import (CorpusSpec, ModelConfig, RenderSpec, SourcePart, SourceSpec,
                        TrainConfig, corpus_preset, default_configs)
from slm.corpus import Corpus, build, load_docs, locate
from slm.generate import generate, load_model, stream
from slm.model import JLM
from slm.render import Renderer, build_renderer
from slm.tokenizer import HFTokenizer, SimpleTokenizer, Tokenizer, load_tokenizer
from slm.train import train

__all__ = [
    "paths", "checkpoint", "chat",
    "ModelConfig", "TrainConfig", "default_configs",
    "SourcePart", "SourceSpec", "RenderSpec", "CorpusSpec", "corpus_preset",
    "Tokenizer", "load_tokenizer", "HFTokenizer", "SimpleTokenizer",
    "JLM",
    "Corpus", "build", "locate", "load_docs",
    "Renderer", "build_renderer",
    "train",
    "generate", "stream", "load_model",
]
