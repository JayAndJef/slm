"""Turning source records into token arrays, per objective.

:class:`Renderer` has one member. Two implementations:

- :class:`PretrainRenderer` — documents joined by the separator, no mask. Every token is a
  target, so both sidecars stay ``None`` and the corpus ships neither.
- :class:`ChatRenderer` — one conversation at a time through the :mod:`slm.chat` template,
  marking assistant spans as loss targets and example starts as segment boundaries.

:func:`build_renderer` dispatches on ``RenderSpec.kind``.

Three deliberate shapes, each of which the obvious alternative gets wrong:

- **Batch in, arrays out.** The pretrain renderer's entire speed is handing 2000 joined
  documents to rust in a single ``encode`` call; a record-at-a-time interface would forbid
  it and turn a 15-minute encode into hours.
- **Stateless across calls.** :func:`slm.corpus.build` shards the source and concatenates
  the results, so anything carried between calls would depend on shard assignment — i.e. on
  ``n_workers``, which is deliberately outside the corpus hash. Counts ride on the return
  value instead.
- **A spec crosses to workers, not a renderer.** A renderer holds a rust tokenizer, which
  cannot be a ``forkserver`` ``initargs`` payload; the worker builds its own.

Torch-free, so encode workers stay light and never inherit a CUDA context.
"""
from typing import NamedTuple, Protocol

import numpy as np

from slm import chat
from slm.config import RenderSpec
from slm.tokenizer import Tokenizer


class Rendered(NamedTuple):
    """One batch's tokens, plus the counts a build needs to report.

    ``is_target`` and ``is_start`` are ``uint8`` and parallel to ``ids``. Their *alignment
    differs at read time* — start goes with ``x``, target with the shifted ``y`` — which is
    why they are two named arrays rather than one packed one; see :mod:`slm.dataset`.
    """

    ids: np.ndarray                     # uint16
    is_target: np.ndarray | None        # uint8, 1 = this token is a loss target
    is_start: np.ndarray | None         # uint8, 1 = this token starts an example
    n_examples: int
    n_dropped: int
    n_dropped_tokens: int
    n_bytes: int                        # source bytes rendered, for exact tokens_per_byte
    n_pad: int = 0                      # padding added by packing; 0 until pack_bfd runs
    n_truncated: int = 0                # examples cut to max_example_tokens (kept, not dropped)

    @classmethod
    def empty(cls, *, sidecars: bool) -> "Rendered":
        z = np.empty(0, dtype=np.uint8)
        return cls(np.empty(0, dtype=np.uint16), z if sidecars else None,
                   z if sidecars else None, 0, 0, 0, 0)

    @classmethod
    def concat(cls, parts: list["Rendered"]) -> "Rendered":
        parts = [p for p in parts if len(p.ids) or p.n_dropped]
        if not parts:
            return cls.empty(sidecars=False)
        has = parts[0].is_target is not None
        cat = lambda f: np.concatenate([getattr(p, f) for p in parts]) if has else None
        return cls(np.concatenate([p.ids for p in parts]), cat("is_target"), cat("is_start"),
                   sum(p.n_examples for p in parts), sum(p.n_dropped for p in parts),
                   sum(p.n_dropped_tokens for p in parts), sum(p.n_bytes for p in parts),
                   sum(p.n_pad for p in parts), sum(p.n_truncated for p in parts))

    def n_real_tokens(self) -> int:
        """Tokens that came from source text, i.e. excluding packing padding.

        Counted rather than derived from the pad id: a reserved id is unambiguous today, but
        counting keeps ``tokens_per_byte`` honest even if padding ever reuses a real token.
        """
        return len(self.ids) - self.n_pad

    def doc_starts(self, sep_id: int | None) -> np.ndarray:
        """Index of the first token of each document, from ``is_start`` or the separator."""
        if self.is_start is not None:
            return np.flatnonzero(self.is_start)
        assert sep_id is not None, "need is_start or a sep_id to find document boundaries"
        return np.flatnonzero(self.ids == sep_id)

    def sort_by_length(self, *, sep_id: int | None = None) -> "Rendered":
        """Reorder whole documents shortest-first, so a length band is a contiguous window
        range the loader can bias toward at read time. A permutation: nothing is added or lost.
        """
        starts = self.doc_starts(sep_id)
        if len(starts) == 0:
            return self
        lengths = np.diff(starts, append=len(self.ids))
        order = np.argsort(lengths, kind="stable")

        # sep is [newline, EOT, newline], so index 0 is a bare newline: keep it as a prefix.
        take = lambda a: None if a is None else np.concatenate(
            [a[:starts[0]]] + [a[starts[i]:starts[i] + lengths[i]] for i in order])
        return self._replace(ids=take(self.ids), is_target=take(self.is_target),
                             is_start=take(self.is_start))

    def length_index(self, *, sep_id: int | None = None, n_quantiles: int = 128) -> list:
        """``[min_tokens, token_offset, doc_offset]`` triples into a length-sorted stream.

        Token offsets, not window indices, because ``block_size`` is a read-time parameter.
        The ladder pins the thresholds anyone actually types, where the equal-token-mass
        quantiles are sparsest; ``doc_offset`` makes a region's mean length recoverable.
        """
        starts = self.doc_starts(sep_id)
        if len(starts) == 0:
            return []
        lengths = np.diff(starts, append=len(self.ids))
        srt = np.sort(lengths, kind="stable")
        # + starts[0]: the sorted stream keeps the pre-first-boundary prefix at its head.
        offsets = int(starts[0]) + np.concatenate([[0], np.cumsum(srt)])

        LADDER = (256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
        marks = set(LADDER)
        marks.update(int(srt[min(len(srt) - 1, int(len(srt) * q / n_quantiles))])
                     for q in range(n_quantiles + 1))
        # Geometric marks to the max: doc-count quantiles all land in the bulk, leaving the
        # tail represented by one document that a high request would then snap onto alone.
        m = float(LADDER[-1])
        while m < srt[-1]:
            m *= 2 ** 0.5
            marks.add(int(min(m, srt[-1])))
        out = []
        for t in sorted(marks):
            i = int(np.searchsorted(srt, t, side="left"))        # first doc with length >= t
            out.append([int(t), int(offsets[i]), i])
        return out

    def pack_bfd(self, block: int, *, pad_id: int) -> "Rendered":
        """Re-order whole examples into ``block``-sized bins, first-fit-decreasing.

        Windows are cut at fixed multiples of ``block``, so on an unpacked stream roughly
        every boundary bisects an example. The tail half then trains with its prompt
        amputated, and masking it away costs ~60% of all loss targets — because ChatML puts
        the answer at the *end* of an example, so the discarded leading fragment is enriched
        in exactly the tokens worth training on.

        Packing at build time means :mod:`slm.dataset` needs no knowledge of any of this: it
        keeps slicing at multiples of ``block``, and every window now holds whole examples
        plus padding that the target sidecar already zeroes.

        One sentinel token is appended, because ``dataset.n_windows`` computes
        ``(n_tokens - 1) // block`` — on an exactly bin-aligned stream that silently drops
        the final bin, the same bin, every epoch.
        """
        assert self.is_start is not None, "packing needs example boundaries (is_start)"
        starts = np.flatnonzero(self.is_start)
        lengths = np.diff(starts, append=len(self.ids))
        assert lengths.max(initial=0) <= block, (
            f"example of {lengths.max(initial=0)} tokens exceeds the {block}-token bin — "
            f"set RenderSpec.max_example_tokens to the block size")

        order = _ffd_bins(lengths, block)
        ids = np.full(len(order) * block, pad_id, dtype=np.uint16)
        tgt = np.zeros(len(order) * block, dtype=np.uint8)
        srt = np.zeros(len(order) * block, dtype=np.uint8)
        for b, members in enumerate(order):
            at = b * block
            for e in members:
                lo, n = starts[e], lengths[e]
                ids[at:at + n] = self.ids[lo:lo + n]
                tgt[at:at + n] = self.is_target[lo:lo + n]
                srt[at] = 1
                at += n
            if at < (b + 1) * block:
                srt[at] = 1     # padding is its own segment, so pad rows attend only pad
        # Sentinel: see the docstring. is_start=1 keeps it out of the last bin's segment.
        ids = np.append(ids, pad_id).astype(np.uint16)
        tgt = np.append(tgt, 0).astype(np.uint8)
        srt = np.append(srt, 1).astype(np.uint8)
        n_pad = len(ids) - int(lengths.sum())
        return self._replace(ids=ids, is_target=tgt, is_start=srt, n_pad=n_pad)


class _OpenBins:
    """Remaining capacity per open bin, answering "leftmost bin that fits ``n``" in O(log b).

    A max-segment-tree over the capacities: ``_max[1]`` is the largest capacity anywhere,
    and descending left-first lands on the lowest-indexed bin that fits. That is *exactly*
    the bin a linear scan returns, so packing is unchanged — which matters, because
    ``corpus_hash`` cannot see a change in packing order and an existing corpus would go
    silently stale.

    Unused leaves hold capacity 0 and every example is at least one token, so they can never
    be selected and need no separate bookkeeping.
    """

    def __init__(self, leaves: int = 1024):
        self._leaves = leaves           # power of two; doubled on demand
        self._n = 0                     # bins actually open
        self._max = [0] * (2 * leaves)

    def open(self, cap: int) -> int:
        """Open a new bin with ``cap`` free, and return its index."""
        if self._n == self._leaves:
            self._grow()
        self._n += 1
        self.set(self._n - 1, cap)
        return self._n - 1

    def _grow(self) -> None:
        caps = self._max[self._leaves:self._leaves + self._n]
        self._leaves *= 2
        self._max = [0] * self._leaves + caps + [0] * (self._leaves - self._n)
        for j in range(self._leaves - 1, 0, -1):
            self._max[j] = max(self._max[2 * j], self._max[2 * j + 1])

    def cap(self, i: int) -> int:
        return self._max[self._leaves + i]

    def set(self, i: int, cap: int) -> None:
        j = self._leaves + i
        self._max[j] = cap
        j >>= 1
        while j:
            self._max[j] = max(self._max[2 * j], self._max[2 * j + 1])
            j >>= 1

    def first_fit(self, n: int) -> int | None:
        """Lowest-indexed bin with at least ``n`` free, or None if none has room."""
        if self._max[1] < n:
            return None
        j = 1
        while j < self._leaves:
            j <<= 1
            if self._max[j] < n:
                j += 1
        return j - self._leaves


def _ffd_bins(lengths: np.ndarray, block: int) -> list[list[int]]:
    """First-fit-decreasing over a length histogram; returns example indices per bin.

    FFD and best-fit are within half a percent on these distributions and share the same
    11/9-optimal bound, so the tie-break is not what matters. Fill is governed by
    ``mean_length / block``, not by the heuristic: below ~0.5 any greedy packer exceeds 99%,
    and at 0.68 even an optimal one caps near 90%.

    The histogram only *orders* the work — it cannot collapse it, because two examples of
    the same length still land in different bins. What keeps this out of O(n_examples x
    n_bins) is :class:`_OpenBins`: scanning every open bin per example measures 0.29s at
    5k examples, 4.5s at 20k and 70s at 80k — clean quadratic, i.e. ~10 hours on smoltalk2's
    1.9M chunks, single-threaded in the parent after the encode has already finished.
    Same packing through the segment tree: 9.5s at 1.28M.
    """
    by_len: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        by_len.setdefault(int(n), []).append(i)

    bins: list[list[int]] = []
    open_bins = _OpenBins()
    for n in sorted(by_len, reverse=True):
        for e in by_len[n]:
            b = open_bins.first_fit(n)
            if b is None:
                open_bins.open(block - n)
                bins.append([e])
            else:
                open_bins.set(b, open_bins.cap(b) - n)
                bins[b].append(e)
    return bins


class Renderer(Protocol):
    """One member: a batch of source rows becomes tokens plus optional sidecars."""

    def render(self, batch: dict) -> Rendered: ...


def build_renderer(spec: RenderSpec, tokenizer: Tokenizer) -> Renderer:
    """Construct the renderer for ``spec.kind``.

    Called once in the parent process before the pool spawns, and again inside each worker.
    The parent's copy exists so a bad template or a missing special token raises here rather
    than twenty minutes into an encode.
    """
    if spec.kind == "pretrain":
        return PretrainRenderer(spec, tokenizer)
    if spec.kind == "chat":
        return ChatRenderer(spec, tokenizer)
    raise ValueError(
        f"RenderSpec.kind must be 'pretrain' or 'chat', got {spec.kind!r}")


# ---------------------------------------------------------------- pretrain

class PretrainRenderer:
    """Documents concatenated behind separators; every token a target.

    Emits no sidecars. Segments come from the separator id at read time, and with no mask
    the loss covers everything — which is what pretraining means.
    """

    def __init__(self, spec: RenderSpec, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.sep = spec.sep
        self.col = spec.text_column

    def render(self, batch: dict) -> Rendered:
        docs = batch[self.col]
        text = "".join(self.sep + d for d in docs)
        # uint16 per batch rather than one growing list[int]: a CPython int costs ~36 bytes
        # against 2 here, which at billions of tokens is more memory than the machine has.
        ids = np.array(self.tok.encode(text), dtype=np.uint16)
        return Rendered(ids, None, None, len(docs), 0, 0, len(text.encode()))


# ---------------------------------------------------------------- chat

class ChatRenderer:
    """One conversation per example, through the :mod:`slm.chat` ChatML template."""

    def __init__(self, spec: RenderSpec, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.col = spec.messages_column
        self.max_len = spec.max_example_tokens
        # Resolve the markers now: a tokenizer without them fails here, in the parent, not
        # inside a forkserver worker minutes into a build.
        for marker in chat.NAMED_SPECIALS:
            tokenizer.special_id(marker)
        self._assert_spans_concat()
        # The leading separator span, encoded once. Every chunk carries it and every
        # per-pair encode repeats it, so subtracting it is what makes lengths additive —
        # see _split_turns.
        self._prefix = self._encode([])

    def _assert_spans_concat(self) -> None:
        """Encoding spans separately must equal encoding the joined string.

        Holds because every boundary falls after a special token or a newline — specials are
        hard pre-tokenizer splits, and a newline ends a ByteLevel chunk. A future template
        edit that ends a span mid-word would break it silently, teaching ids that inference,
        which encodes the whole prompt in one call, never produces.
        """
        spans = chat.render_training([{"role": "user", "content": "Hi there"},
                                      {"role": "assistant", "content": "Hello back"}])
        piecewise = [i for text, _ in spans for i in self.tok.encode(text)]
        assert piecewise == self.tok.encode("".join(t for t, _ in spans)), (
            "chat template spans do not encode the same as the joined string — end every "
            "span after a special token or a newline")

    def _encode(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        ids: list[int] = []
        tgt: list[int] = []
        for text, is_target in chat.render_training(messages):
            e = self.tok.encode(text)
            ids.extend(e)
            tgt.extend([int(is_target)] * len(e))
        return ids, tgt

    def _split_turns(self, messages: list[dict]) -> list[tuple[list[int], list[int]]]:
        """Encode a conversation, breaking it at turn boundaries instead of dropping it.

        Dropping is length-biased and expensive: the examples that exceed the block are the
        long multi-turn ones, and a fifth of examples can hold nearly half the tokens. Each
        chunk starts on a user turn, so every piece is still a well-formed conversation.

        Returns encoded ``(ids, is_target)`` chunks rather than message lists, so each turn
        pair is encoded exactly **once**. Re-encoding the accumulated prefix to measure it
        (and then re-encoding each chunk in :meth:`render`) is O(k^2) tokenizer work in the
        number of turns — ~20x the necessary encode on the 40-turn conversations that
        smol_magpie_ultra and OpenHermes are full of, against a build whose documented
        bottleneck is exactly this encode.

        The running length is exact, not an estimate: span encodings concatenate (see
        :meth:`_assert_spans_concat`), so a chunk is the sum of its pairs' encodings with
        the one shared leading separator counted once.

        ``max_example_tokens=None`` means no limit, so nothing is split.
        """
        pre_ids, pre_tgt = self._prefix
        chunks: list[tuple[list[int], list[int]]] = []
        ids, tgt, n_pairs = list(pre_ids), list(pre_tgt), 0
        for i in range(0, len(messages) - 1, 2):
            e, t = self._encode(messages[i:i + 2])
            e, t = e[len(pre_ids):], t[len(pre_ids):]       # this call's own separator
            if n_pairs and self.max_len is not None and len(ids) + len(e) > self.max_len:
                chunks.append((ids, tgt))
                ids, tgt, n_pairs = list(pre_ids), list(pre_tgt), 0
            ids += e
            tgt += t
            n_pairs += 1
        return chunks + ([(ids, tgt)] if n_pairs else [])

    @staticmethod
    def _fold_system(messages: list[dict]) -> list[dict]:
        """Prepend any system message to the first user turn.

        The template has no system role — one fewer axis for training and inference to drift
        on. Dropping the text instead would gut ``smoltalk_smollm3_systemchats_30k_no_think``,
        whose 34k conversations are *about* the system prompt: the assistant's answers
        reference a persona that would no longer be anywhere in the context.
        """
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        if system and turns and turns[0]["role"] == "user":
            turns[0] = {"role": "user", "content": f"{system}\n\n{turns[0]['content']}"}
        return turns

    def render(self, batch: dict) -> Rendered:
        ids: list[int] = []
        tgt: list[int] = []
        srt: list[int] = []
        n_ex = n_drop = n_drop_tok = n_bytes = n_trunc = 0
        for messages in batch[self.col]:
            messages = self._fold_system(messages)
            if len(messages) < 2:
                continue
            n_bytes += sum(len(m["content"].encode()) for m in messages)
            for e, t in self._split_turns(messages):
                if self.max_len is not None and len(e) > self.max_len:
                    # A single turn pair with no boundary left to split on. Truncate rather
                    # than drop: over-long examples are the long-answer ones, so discarding
                    # them is sharply length-biased — on smoltalk2 they are 8% of examples
                    # but ~20% of tokens. The tail is lost, so this example never shows the
                    # closing <|im_end|>, but the ones that fit still teach stopping.
                    e, t = e[:self.max_len], t[:self.max_len]
                    if not any(t):              # prompt alone fills the block: nothing to learn
                        n_drop += 1
                        n_drop_tok += len(e)
                        continue
                    n_trunc += 1
                srt.extend([1] + [0] * (len(e) - 1))
                ids.extend(e)
                tgt.extend(t)
                n_ex += 1
        return Rendered(np.array(ids, dtype=np.uint16), np.array(tgt, dtype=np.uint8),
                        np.array(srt, dtype=np.uint8), n_ex, n_drop, n_drop_tok, n_bytes,
                        n_truncated=n_trunc)
