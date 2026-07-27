"""The on-disk corpus: documents in, a cached token stream out.

The **only** module that knows what a corpus looks like on disk, the way
:mod:`slm.checkpoint` is the only one that knows a checkpoint's shape. Everything else
addresses a corpus through :class:`Corpus`, so the file layout is described once instead of
re-derived at each call site.

A corpus is up to four files:

===========================  ============================================================
``<name>.npy``               ``uint16`` tokens; always present
``<name>.target.npy``        ``uint8``, 1 = this token is a loss target; chat corpora only
``<name>.start.npy``         ``uint8``, 1 = this token starts an example; chat corpora only
``<name>.meta``              JSON, **written last** — the commit record
===========================  ============================================================

Meta is written last on purpose: a build killed mid-``np.save`` then leaves tokens with no
meta and the next run rebuilds, rather than memory-mapping half a corpus as though it were
whole — which trains without complaint and reports nothing.

Depends on :mod:`slm.render` and ``datasets``, never on torch, Lightning or the model.
"""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from slm import chat
from slm.config import CorpusSpec, ModelConfig, SourceSpec
from slm.render import Rendered, build_renderer
from slm.tokenizer import load_tokenizer

N_WORKERS = 16          # encode processes; see the note in `build` about why it is not hashed
BATCH_DOCS = 2_000      # docs per rust encode call — throughput only, never hashed

# Per-worker state, set by the pool initializer.
_WORKER_RENDERER = None


# ---------------------------------------------------------------- identity

def corpus_hash(spec: CorpusSpec, *, tokenizer_fingerprint: str) -> str:
    """Short digest of everything that determines the token stream.

    Every input here fails *silently* when it drifts. A changed tokenizer leaves all ids in
    range, so nothing crashes, the loss curve looks normal, and the stream means nothing. A
    changed source reshuffles the pool, which moves the val split — losses stop being
    comparable with no signal at all. A changed renderer rewrites every example.

    Folding them into the filename turns each of those into a visible re-encode instead of a
    wrong run. Takes the tokenizer's *fingerprint* rather than its path or its bytes: a path
    is a location, not an identity, and the file bytes move when an unclaimed reserved slot
    is renamed — which cannot change any id (see :attr:`slm.tokenizer.Tokenizer.fingerprint`).

    ``n_workers`` and ``batch_docs`` are deliberately absent. Neither changes which documents
    are present. ``n_workers`` does permute the order shards are concatenated in, and under
    ``pack="bfd"`` it also changes which examples share a bin — so it is recorded in meta,
    where a cross-machine mismatch is visible without being a re-encode.
    """
    import hashlib
    key = repr((tokenizer_fingerprint, spec.source.hash_fields(), spec.render.hash_fields()))
    return hashlib.sha256(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- one corpus on disk

class Corpus:
    """One split of one corpus. Construct via :func:`locate` or :func:`build`, not directly.

    Holds *paths*, never open memmaps, so it survives being pickled into a DataLoader
    worker. ``split`` is an attribute rather than a public argument because the two splits
    are produced together — see :func:`build`.
    """

    def __init__(self, spec: CorpusSpec, split: str, *, data_dir: Path, digest: str):
        self.spec = spec
        self.split = split
        self.data_dir = Path(data_dir)
        self.hash = digest

    @property
    def name(self) -> str:
        """Cache name: tag, doc counts, and the hash.

        Train carries *both* counts because ``load_docs`` slices train after the val docs,
        so ``n_val_docs`` moves the train slice too.
        """
        src = self.spec.source
        if self.split == "val":
            return f"val_{self.spec.name}_{src.n_val_docs}_{self.hash}"
        n_train = "all" if src.n_train_docs is None else src.n_train_docs
        return f"train_{self.spec.name}_{src.n_val_docs}+{n_train}_{self.hash}"

    @property
    def token_path(self) -> Path:
        return self.data_dir / f"{self.name}.npy"

    @property
    def meta_path(self) -> Path:
        # `.meta`, not `.meta.json`: .gitignore is `data/*` + `!data/*.json`, so a .json
        # extension would commit every corpus's metadata.
        return self.data_dir / f"{self.name}.meta"

    @property
    def target_path(self) -> Path | None:
        """Loss-target sidecar, or None when the corpus has none.

        Derived from meta so "does this corpus have sidecars" has exactly one answer. Were
        this to return a path unconditionally, a missing file would degrade to "no loss
        mask" — and training silently covers the user's turns.
        """
        return self.data_dir / f"{self.name}.target.npy" if self.meta["has_target"] else None

    @property
    def start_path(self) -> Path | None:
        return self.data_dir / f"{self.name}.start.npy" if self.meta["has_start"] else None

    def exists(self) -> bool:
        return self.meta_path.exists()

    @property
    def meta(self) -> dict:
        if not hasattr(self, "_meta"):
            assert self.exists(), (
                f"corpus {self.name} is not built — run: "
                f"uv run main.py prepare-data --corpus {self.spec.name}")
            self._meta = json.loads(self.meta_path.read_text())
        return self._meta

    def loader_kwargs(self) -> dict:
        """The four things :func:`slm.dataset.build_dataloader` needs to open this corpus."""
        return {"token_path": self.token_path, "target_path": self.target_path,
                "start_path": self.start_path, "sep_id": self.meta["sep_id"]}

    def assert_compatible(self, model_cfg: ModelConfig, *, tokenizer_fingerprint: str,
                          doc_mask: bool) -> None:
        """Fail loudly on the mismatches that would otherwise train a plausible wrong model.

        Takes a fingerprint *string* rather than a ``Tokenizer`` so callers that already hold
        one do not construct a second, and so this module's consumers need no tokenizer.
        """
        m = self.meta
        assert model_cfg.vocab_size == m["vocab_size"], (
            f"model vocab_size {model_cfg.vocab_size} != corpus {m['vocab_size']} "
            f"({self.name}) — the embedding cannot index this stream")
        assert tokenizer_fingerprint == m["tokenizer_fingerprint"], (
            f"corpus {self.name} was built with tokenizer {m['tokenizer_fingerprint']} but "
            f"this run uses {tokenizer_fingerprint} — every id would mean something else; "
            f"pass --tokenizer for the one it was built with")
        if m["has_start"]:
            assert doc_mask, (
                f"{self.name} is packed, so --no-doc-mask lets each example attend to the "
                f"previous one — pass --doc-mask")
        assert m["has_start"] or m["sep_id"] is not None, (
            f"{self.name} has neither a start sidecar nor a sep_id, so segments would "
            f"collapse to plain causal attention — rebuild it")
        if m["pack"] == "bfd":
            assert model_cfg.block_size == m["pack_block"], (
                f"{self.name} was packed into bins of {m['pack_block']} but block_size is "
                f"{model_cfg.block_size} — bins would no longer align to windows")


def locate(spec: CorpusSpec, *, data_dir, tokenizer_path) -> tuple[Corpus, Corpus]:
    """Name a corpus pair without building it.

    Separate from :func:`build` because ``train``/``sft`` must be able to *check* that a
    corpus exists — and say which command would create it — without ever encoding one
    themselves. Building from a training entrypoint races DDP ranks on a multi-GB write.
    """
    digest = corpus_hash(spec, tokenizer_fingerprint=load_tokenizer(tokenizer_path).fingerprint)
    return (Corpus(spec, "train", data_dir=data_dir, digest=digest),
            Corpus(spec, "val", data_dir=data_dir, digest=digest))


# ---------------------------------------------------------------- documents

def load_docs(spec: SourceSpec, *, hf_cache_dir: str) -> tuple[Dataset, Dataset]:
    """Return (train_docs, val_docs) as disjoint random samples of ``spec.parts``.

    Each part is loaded, shuffled and trimmed to its cap, the trimmed parts are concatenated
    and shuffled again (so parts interleave rather than sitting in blocks), and only
    ``spec.columns`` is kept — the rest is provenance metadata. For cosmopedia in particular,
    ``prompt`` is the instruction given to Mixtral to *generate* the text, not something to
    train on.

    **Validation is taken from the front of the pool**, training from what follows. That
    ordering is deliberate: raising ``n_train_docs`` then cannot swallow the val documents,
    which is exactly how a train/val overlap gets introduced silently. Changing the *parts*
    does change the pool, and therefore the val set — so losses are only comparable across
    runs that share a source.

    Both splits come from one pool computation, so disjointness is structural rather than an
    emergent property of two calls agreeing.
    """
    # HF_HOME is set in slm.paths, before huggingface_hub is imported — see the note there.
    parts = []
    for p in spec.parts:
        ds = load_dataset(spec.dataset_name, name=p.config, split=p.split,
                          cache_dir=hf_cache_dir)
        ds = ds.select_columns(list(spec.columns)).shuffle(seed=spec.seed)
        if p.cap is not None:
            ds = ds.select(range(min(p.cap, len(ds))))   # random sample, not file order
        parts.append(ds)
    pool = concatenate_datasets(parts).shuffle(seed=spec.seed)

    n_val = min(spec.n_val_docs, len(pool))
    end = len(pool) if spec.n_train_docs is None else min(
        len(pool), spec.n_val_docs + spec.n_train_docs)
    return pool.select(range(n_val, end)), pool.select(range(n_val))


# ---------------------------------------------------------------- encode workers

def _worker_init(tokenizer_path, render_spec) -> None:
    global _WORKER_RENDERER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_RENDERER = build_renderer(render_spec, load_tokenizer(tokenizer_path))


def _worker_encode(shard) -> Rendered:
    """Render one shard of the document dataset.

    Takes an Arrow ``Dataset``, not a list of strings. A ``Dataset`` pickles by *file path*,
    so what crosses the process boundary is a few hundred bytes of metadata and the worker
    re-memory-maps the same file — rather than tens of GB of text being serialized, copied
    through a pipe, and rebuilt as Python objects in every worker.
    """
    return Rendered.concat([_WORKER_RENDERER.render(b)
                            for b in shard.iter(batch_size=BATCH_DOCS)])


# ---------------------------------------------------------------- build

def _save_atomic(path: Path, arr: np.ndarray) -> None:
    """Write via a temp file and rename, so a kill never leaves a truncated ``.npy``.

    The temp name ends in ``.npy`` because ``np.save`` appends that suffix when the path
    lacks it — ``np.save("x.npy.tmp", a)`` writes ``x.npy.tmp.npy`` and the rename then
    misses.
    """
    tmp = path.with_suffix(".tmp.npy")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _build_split(spec: CorpusSpec, docs, corpus: Corpus, *, tokenizer_path,
                 n_workers: int, logger) -> None:
    """Encode one split and write its files, meta last."""
    if logger:
        logger(f"encoding {len(docs)} docs -> {corpus.name} with {n_workers} workers")

    tok = load_tokenizer(tokenizer_path)

    # Materialise the shuffle before sharding. `load_docs` concatenates N splits and then
    # shuffles, leaving an indices mapping over N underlying tables — so every row read costs
    # a table lookup plus a scattered Arrow seek, single-threaded in the parent, before any
    # worker starts. Tolerable for cosmopedia's flat `text`; pathological for a nested
    # `messages` column, where it pegged one core for 20 minutes with no worker ever spawned.
    # flatten_indices rewrites the selection once, in parallel, into one sequential table.
    if docs._indices is not None:               # no public accessor for this
        if logger:
            logger(f"  materialising {len(docs)} rows before sharding ...")
        docs = docs.flatten_indices(num_proc=min(n_workers, 8))

    shards = [docs.shard(n_workers, i, contiguous=False) for i in range(n_workers)]
    # forkserver (not fork) so encoding is safe even if a CUDA/NCCL context already exists
    # in this process. Workers are torch-free regardless.
    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(tokenizer_path, spec.render)) as ex:
        # Report shards as they land rather than only at the end: a multi-GB encode is
        # otherwise a silent window, and "slow" and "wedged" look identical from outside.
        futures = {ex.submit(_worker_encode, s): i for i, s in enumerate(shards)}
        done, parts = 0, [None] * len(shards)
        for fut in as_completed(futures):
            parts[futures[fut]] = fut.result()
            done += 1
            if logger:
                logger(f"  shard {done}/{len(shards)} done")
        out = Rendered.concat(parts)

    if spec.render.pack == "bfd":
        # RESERVED_FMT, never the literal: slm.chat owns that format string, and renaming a
        # reserved slot is a supported workflow (see its docstring). A copy here would keep
        # matching a name that no longer exists, or pad with a token that now means something.
        out = out.pack_bfd(spec.render.pack_block,
                           pad_id=tok.special_id(chat.RESERVED_FMT.format(0)))

    sep_id = tok.sep_id(spec.render.sep)
    n_real = int(out.n_real_tokens())
    n_bytes = int(out.n_bytes)
    meta = {
        "n_tokens": len(out.ids), "n_real_tokens": n_real,
        "n_pad_tokens": len(out.ids) - n_real, "n_bytes": n_bytes,
        "tokens_per_byte": (n_real / n_bytes) if n_bytes else None,
        "vocab_size": tok.n_vocab, "tokenizer_fingerprint": tok.fingerprint,
        "sep_id": sep_id, "sep_frac": float((out.ids == sep_id).mean()),
        "n_examples": out.n_examples, "n_dropped": out.n_dropped,
        "n_dropped_tokens": out.n_dropped_tokens, "n_truncated": out.n_truncated,
        "has_target": out.is_target is not None, "has_start": out.is_start is not None,
        "pack": spec.render.pack, "pack_block": spec.render.pack_block,
        "n_workers": n_workers, "batch_docs": BATCH_DOCS,
        "target_frac": float(out.is_target.mean()) if out.is_target is not None else None,
        "spec": {"source": asdict(spec.source), "render": asdict(spec.render),
                 "name": spec.name},
    }

    corpus.data_dir.mkdir(parents=True, exist_ok=True)
    _save_atomic(corpus.token_path, out.ids)
    if out.is_target is not None:
        _save_atomic(corpus.data_dir / f"{corpus.name}.target.npy", out.is_target)
    if out.is_start is not None:
        _save_atomic(corpus.data_dir / f"{corpus.name}.start.npy", out.is_start)
    corpus.meta_path.write_text(json.dumps(meta, indent=2))     # last: the commit record

    if logger:
        logger(f"  {len(out.ids)/1e6:.1f}M tokens, {out.n_examples} examples, "
               f"sep_frac {meta['sep_frac']:.6f}, tokens_per_byte "
               f"{meta['tokens_per_byte'] and round(meta['tokens_per_byte'], 4)}")
        if out.n_dropped or out.n_truncated:
            logger(f"  truncated {out.n_truncated} examples, dropped {out.n_dropped} "
                   f"({out.n_dropped_tokens/max(1, out.n_dropped_tokens + n_real):.2%} of tokens)")


def build(spec: CorpusSpec, *, tokenizer_path, data_dir, hf_cache_dir,
          n_workers: int = N_WORKERS, logger=None) -> tuple[Corpus, Corpus]:
    """Build the (train, val) pair if missing, and return both.

    Both splits come from a single :func:`load_docs` call — building them independently
    would load and shuffle the whole pool twice, and would turn their disjointness from a
    structural fact into a coincidence of determinism.

    Keys on meta, not on the token file: an existing ``.npy`` with no meta is a corpse from
    an interrupted build and is overwritten rather than trusted.
    """
    train, val = locate(spec, data_dir=data_dir, tokenizer_path=tokenizer_path)
    if train.exists() and val.exists():
        if logger:
            logger(f"corpus {spec.name} already built ({train.name})")
        return train, val

    if logger:
        logger(f"building {spec.name}: hash {train.hash} "
               f"(tokenizer {load_tokenizer(tokenizer_path).fingerprint}, "
               f"{len(spec.source.parts)} parts)")
    train_docs, val_docs = load_docs(spec.source, hf_cache_dir=hf_cache_dir)
    # Val first: 30 seconds instead of 15 minutes to surface anything the parent missed.
    for corpus, docs in ((val, val_docs), (train, train_docs)):
        if not corpus.exists():
            _build_split(spec, docs, corpus, tokenizer_path=tokenizer_path,
                         n_workers=n_workers, logger=logger)
    return train, val
