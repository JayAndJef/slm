"""Corpus preparation and batching.

Two responsibilities, all filesystem/data — no model or training logic:

- :func:`load_docs` pulls a random sample of documents from ``cfg.dataset_mix``.
- :func:`build_corpus` encodes documents to a flat ``uint16`` token stream in parallel
  and caches it to disk as a memmap (re-runs are instant).

The worker functions live at module top level (not closures/lambdas) so they pickle by
qualified name for :class:`~concurrent.futures.ProcessPoolExecutor`. The tokenizer is
torch-free, so forked workers stay light and never inherit a CUDA context.
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from slm.config import TrainConfig
from slm.tokenizer import Tokenizer, load_tokenizer

# Per-worker state, set by the pool initializer.
_WORKER_TOK: Tokenizer | None = None
_WORKER_SEP: str = ""


def _worker_init(tokenizer_path, sep: str) -> None:
    global _WORKER_TOK, _WORKER_SEP
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOK = load_tokenizer(tokenizer_path)
    _WORKER_SEP = sep


def _worker_encode(shard, batch_size: int = 2_000) -> np.ndarray:
    """Encode one shard of the document dataset to a ``uint16`` array.

    Takes an Arrow ``Dataset``, not a list of strings. A ``Dataset`` pickles by *file
    path*, so what crosses the process boundary is a few hundred bytes of metadata and the
    worker re-memory-maps the same file — rather than tens of GB of text being serialized,
    copied through a pipe, and rebuilt as Python objects in every worker.

    Every document is prefixed with the separator, so concatenating shards keeps a
    boundary token before each doc.

    Narrows to ``uint16`` per batch instead of accumulating one Python ``list[int]``: a
    CPython int costs ~36 bytes with its list slot against 2 bytes here, which at billions
    of tokens is the difference between ~14 GiB and more memory than the machine has.
    """
    parts = [np.array(_WORKER_TOK.encode("".join(_WORKER_SEP + d for d in batch["text"])),
                      dtype=np.uint16)
             for batch in shard.iter(batch_size=batch_size)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)


def load_docs(cfg: TrainConfig) -> tuple[Dataset, Dataset]:
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
    # HF_HOME is set in slm.paths, before huggingface_hub is imported — see the note there.
    parts = []
    for name, cap in cfg.dataset_mix.items():
        ds = load_dataset(cfg.dataset_name, name=name, split="train",
                          cache_dir=cfg.hf_cache_dir)
        ds = ds.select_columns(["text"]).shuffle(seed=cfg.seed)
        if cap is not None:
            ds = ds.select(range(min(cap, len(ds))))   # random sample, not file order
        parts.append(ds)
    pool = concatenate_datasets(parts).shuffle(seed=cfg.seed)

    n_val = min(cfg.n_val_docs, len(pool))
    end = len(pool) if cfg.n_train_docs is None else min(
        len(pool), cfg.n_val_docs + cfg.n_train_docs)
    return pool.select(range(n_val, end)), pool.select(range(n_val))


def build_corpus(docs, cache_path, tokenizer_path, *, sep: str,
                 n_workers: int = 8, logger=None) -> np.ndarray:
    """Encode a document ``Dataset`` to a cached ``uint16`` memmap, returning it.

    If ``cache_path`` exists it is memory-mapped and returned as-is. Otherwise the docs
    are sharded across ``n_workers`` processes, concatenated, saved, and reopened
    read-only.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if logger:
            logger(f"loaded cached corpus {cache_path.name} ({len(arr)/1e6:.1f}M tokens)")
        return arr

    # The corpus is uint16. numpy would raise on overflow anyway, but only inside a worker
    # after the whole encode — check up front, where the message is actionable.
    n_vocab = load_tokenizer(tokenizer_path).n_vocab
    assert n_vocab <= 65536, (
        f"tokenizer has {n_vocab} ids, which does not fit the uint16 corpus "
        f"(max 65536) — widen the dtype in build_corpus and dataset.py")

    if logger:
        logger(f"encoding {len(docs)} docs -> {cache_path.name} with {n_workers} workers")
    # contiguous=False keeps the round-robin striding the old list slicing used, so the
    # corpus is a stride-permutation of the pool (contiguous=True is what would preserve
    # pool order) — irrelevant either way, hence n_workers staying out of corpus_hash.
    shards = [docs.shard(n_workers, i, contiguous=False) for i in range(n_workers)]
    # forkserver (not fork) so encoding is safe even if a CUDA/NCCL context already
    # exists in this process (e.g. under DDP). Workers are torch-free regardless.
    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init, initargs=(tokenizer_path, sep)) as ex:
        parts = list(ex.map(_worker_encode, shards))   # uint16 arrays, one per worker

    arr = np.concatenate(parts)
    del parts                                          # free before np.save doubles nothing
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    if logger:
        logger(f"encoded {len(arr)/1e6:.1f}M tokens")
    return np.load(cache_path, mmap_mode="r")
