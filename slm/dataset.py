"""Streaming windows from a cached token corpus for DDP training.

The corpus is a flat 1-D ``uint16`` memmap, cut into non-overlapping fixed-length windows
(next-token targets are the same window shifted by one). Each pass shuffles the window
order and hands every reader a disjoint slice of it, so within a pass no window is ever
served twice and none is skipped.

That is worth the small amount of bookkeeping. Drawing random offsets *with replacement* —
the obvious alternative — wastes part of the budget re-reading windows it has already
served: drawing N tokens' worth from a corpus of M covers only ``1 - exp(-N/M)`` of it, so
at N/M = 1/3 you touch 28% of the corpus instead of 33%. It also has no resumable position
and no real notion of an epoch.

Every reader derives the *same* permutation from ``(base_seed, epoch)`` and then takes
``order[shard::n_shards]``. Note the inversion from a with-replacement sampler: the RNG
must be **identical** across ranks and workers so they agree on the ordering, and it is the
shard index — not the seed — that keeps them from colliding.

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
        n_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        shard, n_shards = self.rank * n_workers + worker_id, self.world_size * n_workers

        data = np.load(self.cache_path, mmap_mode="r")
        bs = self.block_size
        # -1 leaves the one extra token that y's shift needs past the last window.
        n_windows = (len(data) - 1) // bs
        assert n_windows >= n_shards, (
            f"corpus holds {n_windows} windows of {bs} tokens but there are {n_shards} "
            f"readers ({self.world_size} ranks x {n_workers} workers) — some would idle")

        epoch = 0
        while True:
            order = np.random.default_rng((self.base_seed, epoch)).permutation(n_windows)
            for w in order[shard::n_shards]:
                i = int(w) * bs
                x = torch.from_numpy(data[i:i + bs].astype(np.int64))
                y = torch.from_numpy(data[i + 1:i + 1 + bs].astype(np.int64))
                # Emitted unconditionally (all zeros when sep_id is None) so the batch
                # arity never depends on config. A BlockMask cannot travel this path — it
                # carries a Python closure and would not survive shared-memory pickling.
                yield x, y, segment_ids(x, self.sep_id)
            epoch += 1


def build_dataloader(cache_path, *, block_size: int, batch_size: int, seed: int,
                     rank: int = 0, world_size: int = 1, num_workers: int = 0,
                     sep_id: int | None = None) -> DataLoader:
    """DataLoader over :class:`WindowIterableDataset`; default collate stacks windows."""
    ds = WindowIterableDataset(cache_path, block_size=block_size, base_seed=seed,
                               rank=rank, world_size=world_size, sep_id=sep_id)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
