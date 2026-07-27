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


def n_windows(n_tokens: int, block_size: int) -> int:
    """Non-overlapping windows a corpus of ``n_tokens`` yields.

    The ``-1`` leaves the one extra token that ``y``'s shift needs past the last window. A
    free function because eval sizing needs the same arithmetic, and two copies of an
    off-by-one whose whole point is subtlety is how they diverge.
    """
    return (n_tokens - 1) // block_size


class WindowIterableDataset(IterableDataset):
    """Infinite stream of shuffled ``(x, y, seg)`` token windows from a memmap corpus.

    Stores cache *paths* (not memmaps) so it survives being sent to a freshly spawned
    worker process, and opens them lazily inside ``__iter__``.

    The two optional sidecars are sliced at **different offsets**, which is why they are two
    named files rather than one array:

    - ``is_start`` aligns with ``x`` — it says which token *begins* an example.
    - ``is_target`` aligns with ``y`` — it says which token should be *predicted*, and ``y``
      is the stream shifted by one.

    Reusing one slice for both would leave the mask silently one position out, which no loss
    curve reveals.
    """

    def __init__(self, token_path, block_size: int, base_seed: int,
                 rank: int = 0, world_size: int = 1, sep_id: int | None = None,
                 target_path=None, start_path=None, mask_partial_head: bool = True):
        super().__init__()
        self.token_path = str(token_path)
        self.target_path = None if target_path is None else str(target_path)
        self.start_path = None if start_path is None else str(start_path)
        self.block_size = block_size
        self.base_seed = base_seed
        self.rank = rank
        self.world_size = world_size
        self.sep_id = sep_id
        self.mask_partial_head = mask_partial_head

    def __iter__(self):
        info = get_worker_info()
        n_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        shard, n_shards = self.rank * n_workers + worker_id, self.world_size * n_workers

        load = lambda p: None if p is None else np.load(p, mmap_mode="r")
        data = np.load(self.token_path, mmap_mode="r")
        target = load(self.target_path)
        start = load(self.start_path)
        for name, side in (("target", target), ("start", start)):
            assert side is None or len(side) == len(data), (
                f"{name} sidecar has {len(side)} entries but the corpus has {len(data)} — "
                f"they must be parallel; rebuild the corpus")

        bs = self.block_size
        n_win = n_windows(len(data), bs)
        assert n_win >= n_shards, (
            f"corpus holds {n_win} windows of {bs} tokens but there are {n_shards} "
            f"readers ({self.world_size} ranks x {n_workers} workers) — some would idle")

        epoch = 0
        while True:
            order = np.random.default_rng((self.base_seed, epoch)).permutation(n_win)
            for w in order[shard::n_shards]:
                i = int(w) * bs
                x = torch.from_numpy(data[i:i + bs].astype(np.int64))
                y = torch.from_numpy(data[i + 1:i + 1 + bs].astype(np.int64))

                if start is None:
                    seg = segment_ids(x, self.sep_id)
                else:
                    seg = torch.from_numpy(start[i:i + bs].astype(np.int64)).cumsum(0)

                if target is not None:
                    keep = torch.from_numpy(target[i + 1:i + 1 + bs].astype(bool))
                    if self.mask_partial_head and start is not None:
                        # Drop the example the window opened mid-way through: its prompt is
                        # in the previous window, so those targets train on amputated
                        # context. A window with *no* start at all sits entirely inside one
                        # long example, so every target in it is amputated and the whole
                        # window goes — the empty-head case is the worst one, not a no-op.
                        # Packed corpora always start on a bin boundary and lose nothing.
                        head = torch.nonzero(torch.from_numpy(start[i:i + bs]))
                        keep[:int(head[0]) if len(head) else bs] = False
                    y = y.masked_fill(~keep, -100)

                # Batch arity never depends on config: seg is emitted unconditionally, and
                # -100 is already written into y. A BlockMask cannot travel this path — it
                # carries a Python closure and would not survive shared-memory pickling.
                yield x, y, seg
            epoch += 1


def build_dataloader(token_path, *, block_size: int, batch_size: int, seed: int,
                     rank: int = 0, world_size: int = 1, num_workers: int = 0,
                     sep_id: int | None = None, target_path=None, start_path=None,
                     mask_partial_head: bool = True) -> DataLoader:
    """DataLoader over :class:`WindowIterableDataset`; default collate stacks windows."""
    ds = WindowIterableDataset(token_path, block_size=block_size, base_seed=seed,
                               rank=rank, world_size=world_size, sep_id=sep_id,
                               target_path=target_path, start_path=start_path,
                               mask_partial_head=mask_partial_head)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
