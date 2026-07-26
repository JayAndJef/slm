"""Byte-level BPE tokenizers: a fast rust-backed one, and one written from scratch.

Two interchangeable backends behind the :class:`Tokenizer` protocol:

- :class:`HFTokenizer` — wraps HuggingFace ``tokenizers`` (rust). The **default**, and what
  a 32k vocab over ~15 GiB of text needs to be practical.
- :class:`SimpleTokenizer` — the from-scratch implementation this project exists to teach.
  Still fully usable: point ``--tokenizer`` at a file it saved.

:func:`load_tokenizer` picks the backend by looking at the file, since the two on-disk
formats are disjoint. Nothing else in the codebase needs to know which is in use.

Both are **torch-free** — only the standard library plus ``tokenizers`` — so
corpus-encoding workers stay lightweight and never pull a CUDA context into a child.

Design:
- The base vocabulary is the 256 byte values, so any UTF-8 text encodes without an
  out-of-vocabulary case.
- Training counts adjacent byte-pair frequencies over whitespace-delimited chunks
  (words keep their leading space), merging the most frequent pair each step. Pair
  counts are maintained incrementally (``counts`` + a ``where`` inverted index) so each
  merge only touches the chunks that changed.
- ``encode`` applies learned merges earliest-first (lowest new id), which reproduces the
  order training created them in. ``decode`` concatenates the byte expansions and
  decodes once with ``errors="replace"`` (a single token can end mid-character).
"""
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol

# Aliased: this module defines its own `Tokenizer` protocol, and the rust class would
# otherwise shadow it.
from tokenizers import Tokenizer as _RustTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

# Split into "leading whitespace + word" chunks (or a pure-whitespace run). Merges never
# cross chunk boundaries — this is what GPT-2's regex pre-split buys, and it keeps a
# leading space attached so " the" and "the" are distinct tokens.
SPLIT_PATTERN = re.compile(r"\s*\S+|\s+")


class Tokenizer(Protocol):
    """What the rest of the codebase actually needs from a tokenizer.

    Four members, derived by auditing every call site. Training is deliberately *not* here:
    the two backends train from different shapes (one giant string vs an iterator), and a
    two-branch ``if`` in one CLI command beats forcing them into a common signature.
    """

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...

    @property
    def n_vocab(self) -> int:
        """The *actual* number of ids, i.e. what ``ModelConfig.vocab_size`` must equal."""

    def sep_id(self, sep: str) -> int:
        """The single id marking a document boundary, asserted to exist.

        Lives on the backend because the two vocabularies are shaped differently and
        neither derivation ports — see each implementation.
        """


def load_tokenizer(path) -> Tokenizer:
    """Load whichever backend wrote ``path``, deciding from the file itself.

    The two schemas are disjoint: this project's own format is exactly
    ``{"vocab_size", "merges"}`` at the top level, while HuggingFace's has ``"model"``.
    Sniffing rather than a config flag means the backend can never disagree with the file,
    and ``--tokenizer PATH`` keeps selecting it with no extra CLI surface.
    """
    with open(path) as f:
        head = json.load(f)
    if "model" in head:
        return HFTokenizer.load(path)
    if "merges" in head and "vocab_size" in head:
        return SimpleTokenizer.load(path)
    raise ValueError(
        f"{path} is not a tokenizer this project recognises: expected HuggingFace's "
        f'{{"model": ...}} or SimpleTokenizer\'s {{"vocab_size", "merges"}}, '
        f"got top-level keys {sorted(head)[:8]}")


class SimpleTokenizer:
    """A trainable byte-level BPE tokenizer with save/load."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merge_rules: dict[tuple[int, int], int] = {}
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ training
    def train_tokenizer(self, text: str) -> None:
        """Learn ``vocab_size - 256`` merges from ``text`` (in place)."""
        freqs = Counter(SPLIT_PATTERN.findall(text))
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in freqs]
        weights = list(freqs.values())

        # counts[pair] -> weighted occurrences; where[pair] -> chunk indices containing it.
        counts: dict[tuple[int, int], int] = {}
        where: dict[tuple[int, int], set[int]] = {}
        for idx, ids in enumerate(chunk_ids):
            self._index_chunk(idx, ids, weights[idx], counts, where, +1)

        for _ in range(self.vocab_size - 256):
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = len(self.vocab)
            self.merge_rules[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            for idx in list(where.get(pair, ())):
                ids = chunk_ids[idx]
                self._index_chunk(idx, ids, weights[idx], counts, where, -1)
                ids = self._merge(ids, pair, new_id)
                chunk_ids[idx] = ids
                self._index_chunk(idx, ids, weights[idx], counts, where, +1)
        self._cache.clear()

    @staticmethod
    def _index_chunk(idx, ids, weight, counts, where, sign) -> None:
        """Add (sign=+1) or remove (sign=-1) one chunk's pair statistics."""
        for pair in zip(ids, ids[1:]):
            total = counts.get(pair, 0) + sign * weight
            if total <= 0:
                counts.pop(pair, None)
                where.pop(pair, None)
            else:
                counts[pair] = total
                if sign > 0:
                    where.setdefault(pair, set()).add(idx)

    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every non-overlapping occurrence of ``pair`` with ``new_id``."""
        out: list[int] = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------ encode / decode
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned applicable merge first.
            best = min(zip(ids, ids[1:]), key=lambda p: self.merge_rules.get(p, float("inf")))
            if best not in self.merge_rules:
                break
            ids = self._merge(ids, best, self.merge_rules[best])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self.vocab[t] for t in tokens).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ protocol
    @property
    def n_vocab(self) -> int:
        """Actual id count. Distinct from ``vocab_size``, which is only a training target
        and is what ``save`` writes — they coincide only if training ran to completion."""
        return len(self.vocab)

    def sep_id(self, sep: str) -> int:
        """Id of the document-boundary token.

        ``SPLIT_PATTERN`` peels the separator's trailing newline into its own chunk, so
        ``encode(sep)`` returns two ids — but ``build_corpus`` writes ``sep + doc``
        contiguously, so that newline is absorbed into the next word and ``ids[0]`` is the
        lone boundary token in the corpus.

        The assert catches the one failure that would ruin a run silently: if BPE never
        learned the separator as a merge, ``ids[0]`` is a bare ``"\\n"``, every newline
        reads as a document boundary, and training converges on a meaningless segmentation.
        """
        ids = self.encode(sep)
        marker = sep.strip().encode()
        assert marker in self.vocab[ids[0]], (
            f"separator token {ids[0]} is {self.vocab[ids[0]]!r}, which does not carry "
            f"{marker!r} — this tokenizer did not learn {sep!r} as one token")
        return ids[0]

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"vocab_size": self.vocab_size,
                 "merges": [[a, b, i] for (a, b), i in self.merge_rules.items()]},
                f,
            )

    @classmethod
    def load(cls, path) -> "SimpleTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        for a, b, new_id in sorted(data["merges"], key=lambda m: m[2]):
            tok.merge_rules[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok


class HFTokenizer:
    """Byte-level BPE backed by HuggingFace ``tokenizers`` (rust).

    Same algorithm as :class:`SimpleTokenizer`, orders of magnitude faster to train and to
    encode.

    The separator is a **declared special token**, not a merge BPE has to discover — the
    single-id property that document masking depends on is guaranteed by construction.
    """

    def __init__(self, tk):
        self._tk = tk

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, special: str) -> "HFTokenizer":
        """Train from an iterable of documents (never materialized as one string)."""
        tk = _RustTokenizer(models.BPE(unk_token=None))                      # byte-level: no UNK case
        # add_prefix_space=False: True would prepend a space to *every* encode() call, so
        # generation's per-prompt encode would disagree with the corpus's per-chunk encode.
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tk.decoder = decoders.ByteLevel()
        tk.train_from_iterator(texts, trainers.BpeTrainer(
            vocab_size=vocab_size,
            # On the *trainer*, so the special token is counted within vocab_size. Adding it
            # afterwards would append id `vocab_size` and make n_vocab one too large.
            special_tokens=[special],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        ))
        return cls(tk)

    # ------------------------------------------------------------------ encode / decode
    def encode(self, text: str) -> list[int]:
        return self._tk.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        # skip_special_tokens defaults to True, which would drop every separator from a
        # decoded corpus slice and silently bias SLMDataModule.setup's tokens_per_byte.
        return self._tk.decode(tokens, skip_special_tokens=False)

    # ------------------------------------------------------------------ protocol
    @property
    def n_vocab(self) -> int:
        return self._tk.get_vocab_size(with_added_tokens=True)

    def sep_id(self, sep: str) -> int:
        """Id of the document-boundary token — looked up directly, not inferred.

        Unlike the from-scratch backend, ``encode(sep)`` here yields *three* ids
        (``newline, separator, newline``) because ByteLevel keeps the newlines separate. So
        ``ids[0]`` would be a bare newline: taking it would make every ``\\n`` a document
        boundary. The special token is looked up by name instead.
        """
        marker = sep.strip()
        tid = self._tk.token_to_id(marker)
        assert tid is not None, (
            f"{marker!r} is not a token in this tokenizer — it must be trained as a "
            f"special token for document masking to work")
        n = self.encode(sep).count(tid)
        assert n == 1, f"{marker!r} encodes to {n} occurrences in {sep!r}, expected exactly 1"
        return tid

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tk.save(str(path))

    @classmethod
    def load(cls, path) -> "HFTokenizer":
        # Each corpus worker holds its own tokenizer; rust would otherwise spin up a rayon
        # pool per process on top of the 8 already-forked workers.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        return cls(_RustTokenizer.from_file(str(path)))
