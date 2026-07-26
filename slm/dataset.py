"""Streaming windows from a cached token corpus for DDP training.

The corpus is a flat 1-D ``uint16`` memmap. Training samples random fixed-length windows
from it (next-token targets are the same window shifted by one). Under DDP each rank —
and each DataLoader worker within a rank — must draw a *different* random stream, or the
gradient estimate is biased by replayed windows. :class:`WindowIterableDataset` folds the
rank and worker id into its RNG seed to guarantee that.

This module depends only on torch/numpy; it knows nothing about the model or Lightning.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def segment_ids(x: torch.Tensor, sep_id: int | None) -> torch.Tensor:
    """Per-token segment index for one window, used to build the attention mask.

    The cumsum is *inclusive*, so a separator token belongs to the document it introduces
    rather than the one it ends — it acts as that document's BOS. ``sep_id=None`` yields
    all zeros, i.e. one segment, which is exactly plain causal attention.

    Derived from ``x``, never ``y``: using the shifted targets would move every boundary by
    one position, which no loss curve would reveal.
    """
    if sep_id is None:
        return torch.zeros_like(x)
    return (x == sep_id).cumsum(0)


class WindowIterableDataset(IterableDataset):
    """Infinite stream of random ``(x, y)`` token windows from a memmap corpus.

    Stores the cache *path* (not the memmap) so it survives being sent to a freshly
    spawned worker process, and opens the memmap lazily inside ``__iter__``.
    """

    def __init__(self, cache_path, block_size: int, base_seed: int,
                 rank: int = 0, world_size: int = 1, sep_id: int | None = None):
        super().__init__()
        self.cache_path = str(cache_path)
        self.block_size = block_size
        self.base_seed = base_seed
        self.rank = rank
        self.world_size = world_size
        self.sep_id = sep_id

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        # Distinct RNG stream per (rank, worker); stable across persistent_workers.
        seed = (self.base_seed * 2654435761 + self.rank * 1000003 + worker_id) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        data = np.load(self.cache_path, mmap_mode="r")
        hi = len(data) - self.block_size - 1
        bs = self.block_size
        while True:
            i = int(rng.integers(0, hi))
            x = torch.from_numpy(data[i:i + bs].astype(np.int64))
            y = torch.from_numpy(data[i + 1:i + 1 + bs].astype(np.int64))
            # Emitted unconditionally (all zeros when sep_id is None) so the batch arity
            # never depends on config. A BlockMask cannot travel this path — it carries a
            # Python closure and would not survive the worker's shared-memory pickling.
            yield x, y, segment_ids(x, self.sep_id)


def build_dataloader(cache_path, *, block_size: int, batch_size: int, seed: int,
                     rank: int = 0, world_size: int = 1, num_workers: int = 0,
                     sep_id: int | None = None) -> DataLoader:
    """DataLoader over :class:`WindowIterableDataset`; default collate stacks windows."""
    ds = WindowIterableDataset(cache_path, block_size, seed, rank, world_size, sep_id)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
