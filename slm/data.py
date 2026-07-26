"""Corpus preparation and batching.

Three responsibilities, all filesystem/data — no model or training logic:

- :func:`load_docs` pulls a random sample of documents from fineweb-edu.
- :func:`build_corpus` encodes documents to a flat ``uint16`` token stream in parallel
  and caches it to disk as a memmap (re-runs are instant).
- :func:`get_batch` samples random fixed-length ``(x, y)`` windows for training.

The worker functions live at module top level (not closures/lambdas) so they pickle by
qualified name for :class:`~concurrent.futures.ProcessPoolExecutor`. The tokenizer is
torch-free, so forked workers stay light and never inherit a CUDA context.
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset

from slm.config import TrainConfig
from slm.tokenizer import SimpleTokenizer

# Per-worker state, set by the pool initializer.
_WORKER_TOK: SimpleTokenizer | None = None
_WORKER_SEP: str = ""


def _worker_init(tokenizer_path, sep: str) -> None:
    global _WORKER_TOK, _WORKER_SEP
    _WORKER_TOK = SimpleTokenizer.load(tokenizer_path)
    _WORKER_SEP = sep


def _worker_encode(docs: list[str], chunk: int = 2_000) -> np.ndarray:
    """Encode a slice of documents to a ``uint16`` array.

    Every document is prefixed with the separator, so concatenating slices keeps a
    boundary token before each doc.

    Encodes in chunks and narrows to ``uint16`` as it goes, rather than building one
    Python ``list[int]`` for the whole slice. A CPython int costs ~36 bytes with its list
    slot against 2 bytes here — at billions of tokens that is the difference between
    ~14 GiB and more RAM than the machine has. It also keeps the value returned across
    the process boundary small, since the list would otherwise be pickled whole.
    """
    parts = [np.array(_WORKER_TOK.encode("".join(_WORKER_SEP + d for d in docs[i:i + chunk])),
                      dtype=np.uint16)
             for i in range(0, len(docs), chunk)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)


def load_docs(cfg: TrainConfig) -> tuple[list[str], list[str]]:
    """Return (train_docs, val_docs) as disjoint random samples of ``cfg.dataset_mix``.

    Each HF config named in the mix is shuffled and trimmed to its row cap, the trimmed
    parts are concatenated and shuffled again (so configs interleave rather than sitting
    in blocks), and only the ``text`` column is kept — the rest are provenance metadata.
    For Cosmopedia in particular, ``prompt`` is the instruction given to Mixtral to
    *generate* the text, not something to train on.

    **Validation is taken from the front of the pool**, training from what follows. That
    ordering is deliberate: raising ``n_train_docs`` then cannot swallow the val documents,
    which is exactly how a train/val overlap gets introduced silently. Changing the *mix*
    does change the pool, and therefore the val set — so losses are only comparable across
    runs that share a mix.
    """
    os.environ["HF_HOME"] = cfg.hf_cache_dir
    parts = []
    for name, cap in cfg.dataset_mix.items():
        ds = load_dataset(cfg.dataset_name, name=name, split="train",
                          cache_dir=cfg.hf_cache_dir)
        ds = ds.select_columns(["text"]).shuffle(seed=cfg.seed)
        if cap is not None:
            ds = ds.select(range(min(cap, len(ds))))   # random sample, not file order
        parts.append(ds)
    pool = concatenate_datasets(parts).shuffle(seed=cfg.seed)

    val_docs = pool.select(range(min(cfg.n_val_docs, len(pool))))["text"]
    end = len(pool) if cfg.n_train_docs is None else min(
        len(pool), cfg.n_val_docs + cfg.n_train_docs)
    train_docs = pool.select(range(len(val_docs), end))["text"]
    return train_docs, val_docs


def build_corpus(docs: list[str], cache_path, tokenizer_path, *, sep: str,
                 n_workers: int = 8, logger=None) -> np.ndarray:
    """Encode ``docs`` to a cached ``uint16`` memmap, returning it.

    If ``cache_path`` exists it is memory-mapped and returned as-is. Otherwise the docs
    are encoded across ``n_workers`` processes (round-robin slices), concatenated, saved,
    and reopened read-only.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if logger:
            logger(f"loaded cached corpus {cache_path.name} ({len(arr)/1e6:.1f}M tokens)")
        return arr

    if logger:
        logger(f"encoding {len(docs)} docs -> {cache_path.name} with {n_workers} workers")
    slices = [docs[i::n_workers] for i in range(n_workers)]
    # forkserver (not fork) so encoding is safe even if a CUDA/NCCL context already
    # exists in this process (e.g. under DDP). Workers are torch-free regardless.
    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init, initargs=(tokenizer_path, sep)) as ex:
        parts = list(ex.map(_worker_encode, slices))   # uint16 arrays, one per worker

    arr = np.concatenate(parts)
    del parts                                          # free before np.save doubles nothing
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    if logger:
        logger(f"encoded {len(arr)/1e6:.1f}M tokens")
    return np.load(cache_path, mmap_mode="r")


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device):
    """Sample ``batch_size`` random ``(x, y)`` windows of length ``block_size``.

    ``y`` is ``x`` shifted one token (the next-token targets). ``data`` is a flat token
    array (e.g. a memmap); batches are cast to int64 and moved to ``device``.
    """
    idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = np.stack([data[i:i + block_size] for i in idx])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in idx])
    x = torch.from_numpy(x.astype(np.int64)).to(device, non_blocking=True)
    y = torch.from_numpy(y.astype(np.int64)).to(device, non_blocking=True)
    return x, y
