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

from slm.config import TrainConfig
from slm.tokenizer import SimpleTokenizer

# Per-worker state, set by the pool initializer.
_WORKER_TOK: SimpleTokenizer | None = None
_WORKER_SEP: str = ""


def _worker_init(tokenizer_path, sep: str) -> None:
    global _WORKER_TOK, _WORKER_SEP
    _WORKER_TOK = SimpleTokenizer.load(tokenizer_path)
    _WORKER_SEP = sep


def _worker_encode(docs: list[str]) -> list[int]:
    # Every document is prefixed with the separator, so concatenating slices keeps a
    # boundary token before each doc.
    return _WORKER_TOK.encode("".join(_WORKER_SEP + d for d in docs))


def load_docs(cfg: TrainConfig) -> tuple[list[str], list[str]]:
    """Return (train_docs, val_docs) as disjoint random samples of the dataset.

    The full corpus is shuffled once with ``cfg.seed`` and then sliced, so both splits
    are representative samples rather than raw file order.
    """
    from datasets import load_dataset

    os.environ["HF_HOME"] = cfg.hf_cache_dir
    ds = load_dataset(cfg.dataset_name, name=cfg.dataset_config,
                      split="train", cache_dir=cfg.hf_cache_dir)
    shuffled = ds.shuffle(seed=cfg.seed)
    train_docs = shuffled.select(range(cfg.n_train_docs))["text"]
    val_docs = shuffled.select(
        range(cfg.n_train_docs, cfg.n_train_docs + cfg.n_val_docs))["text"]
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
    ids: list[int] = []
    # forkserver (not fork) so encoding is safe even if a CUDA/NCCL context already
    # exists in this process (e.g. under DDP). Workers are torch-free regardless.
    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init, initargs=(tokenizer_path, sep)) as ex:
        for part in ex.map(_worker_encode, slices):
            ids.extend(part)

    arr = np.array(ids, dtype=np.uint16)
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
