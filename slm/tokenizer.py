"""Byte-level BPE tokenizer, implemented from scratch.

This is the single canonical tokenizer for the project (the notebook keeps its own
self-contained copy). It is deliberately **torch-free** — it imports only the standard
library — so that corpus-encoding worker processes stay lightweight and never pull a
CUDA context into a forked child.

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
import re
from collections import Counter

# Split into "leading whitespace + word" chunks (or a pure-whitespace run). Merges never
# cross chunk boundaries — this is what GPT-2's regex pre-split buys, and it keeps a
# leading space attached so " the" and "the" are distinct tokens.
SPLIT_PATTERN = re.compile(r"\s*\S+|\s+")


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

    def _merge(self, text: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every non-overlapping occurrence of ``pair`` with ``new_id``."""
        new_text: list[int] = []
        i = 0
        while i < len(text):
            if text[i] == pair[0] and i < len(text) - 1 and text[i + 1] == pair[1]:
                new_text.append(new_id)
                i += 2
            else:
                new_text.append(text[i])
                i += 1
        return new_text

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

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
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
