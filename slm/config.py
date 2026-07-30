"""Typed configuration: what a corpus *is*, and how a run is tuned.

Two kinds of dataclass, and the distinction is load-bearing:

- A **Spec** describes an artifact's identity. :class:`SourceSpec` says which documents
  exist, :class:`RenderSpec` how they become tokens, :class:`CorpusSpec` pairs them. Their
  fields are digested into a corpus filename, so changing one produces a *different corpus*
  rather than a silently different run.
- A **Config** holds run knobs. :class:`ModelConfig` is the five architecture dimensions
  (exactly the keys a checkpoint stores); :class:`TrainConfig` is optimization, eval and
  runtime. Neither is ever hashed — two runs at different learning rates read the same
  corpus, and should.

:func:`default_configs` is the single source of truth for the "smoke" overrides, and
:data:`CORPUS_PRESETS` for the corpora themselves.
"""
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from slm import paths
from slm.chat import EOT

SEP = f"\n{EOT}\n"              # document separator; one token id in the corpus


# ---------------------------------------------------------------- specs (hashed)

@dataclass(frozen=True)
class SourcePart:
    """One ``(config, split)`` slice of a dataset, with an optional row cap.

    ``shards`` names the data files to fetch and is the only thing that bounds the download —
    ``cap`` bounds only what is kept, after the whole config has already been pulled.
    """

    config: str | None                  # HF config/subset name; None if unpartitioned
    split: str = "train"                # HF split — NOT Corpus's train/val split
    cap: int | None = None              # rows kept after loading; does NOT bound the download
    shards: tuple[str, ...] | None = None   # explicit data files, relative to the repo root


@dataclass
class SourceSpec:
    """Which documents exist, and how they divide into train and val.

    ``parts`` is a flat list rather than a ``{config: cap}`` mapping plus one split name,
    because datasets disagree about which axis carries the structure: cosmopedia is 7
    configs each with a ``train`` split, while smoltalk2 is one ``SFT`` config with 25
    splits. A list of ``(config, split, cap)`` expresses both without a special case.

    It is always an **explicit** list, never a pattern. A glob is stable in the file while
    the set it matches is not: an upstream split appearing would change the corpus with no
    change to its hash — the exact drift this module exists to make visible. Write the list
    with ``main.py list-splits``.
    """

    dataset_name: str
    parts: tuple[SourcePart, ...]
    columns: tuple[str, ...] = ("text",)
    n_train_docs: int | None = None     # cap on training docs after mixing; None = all
    n_val_docs: int = 5_000
    seed: int = 42                      # shuffles the document pool

    def hash_fields(self) -> tuple:
        """Everything that changes *which documents* are present, in a stable order.

        The doc counts are excluded because the corpus filename already carries them, so a
        changed cap renames the file without pretending to be a different recipe.
        """
        return (self.dataset_name, tuple(dataclasses.astuple(p) for p in self.parts),
                tuple(self.columns), self.seed)


@dataclass
class RenderSpec:
    """How a source record becomes tokens.

    Every field that can change the token stream is hashed; the ones that cannot are
    excluded and say so. ``mask_partial_head`` is the interesting exclusion — it is applied
    when a window is *read*, not when the corpus is written, so it changes which tokens
    count rather than which tokens exist.
    """

    kind: str = "pretrain"              # "pretrain" | "chat"
    sep: str = SEP
    text_column: str = "text"
    messages_column: str = "messages"
    max_example_tokens: int | None = None   # chat: drop/truncate examples longer than this
    pack: str = "flat"                  # "flat" | "sorted" | "bfd"
    pack_block: int | None = None       # bin size for bfd; must equal the model's block_size
    mask_partial_head: bool = True      # read-time only -> NOT hashed

    def hash_fields(self) -> tuple:
        """Only the fields live for this ``kind``.

        Hashing all of them would make changing ``messages_column`` on a pretrain corpus, or
        ``max_example_tokens`` on a pretrain corpus, re-encode a stream that did not change.
        """
        common = (self.kind, self.sep, self.pack, self.pack_block)
        if self.kind == "pretrain":
            return common + (self.text_column,)
        if self.kind == "chat":
            return common + (self.messages_column, self.max_example_tokens)
        raise ValueError(
            f"RenderSpec.kind must be 'pretrain' or 'chat', got {self.kind!r}")


@dataclass
class CorpusSpec:
    """A corpus's identity: where its documents come from, and how they are rendered.

    Deliberately carries no paths. ``tokenizer_path`` / ``data_dir`` / ``hf_cache_dir`` are
    *locations*, not identity — they say where to look, not what is there — and folding them
    in would bake machine-local absolute paths into the record that is supposed to describe
    the corpus. They are arguments to :func:`slm.corpus.build` instead.
    """

    source: SourceSpec
    render: RenderSpec
    name: str = ""                      # set by corpus_preset from its key; names the cache files


# ---------------------------------------------------------------- configs (never hashed)

@dataclass
class ModelConfig:
    """The five dimensions that define a JLM transformer's architecture."""

    vocab_size: int = 32_000
    hidden_dim: int = 1536
    num_heads: int = 12
    n_layer: int = 16
    block_size: int = 2048
    rope_theta: float = 10_000.0        # RoPE base; raise it when extending context

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Build from a (possibly extra-keyed) dict, ignoring unknown fields.

        Filtering rather than ``cls(**d)`` covers dicts carrying keys this class lacks,
        where ``cls(**d)`` raises; a dict merely *missing* a field already loads, since
        the dataclass default fills it in.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TrainConfig:
    """Settings for one training run: optimization, eval, runtime, and where things live."""

    # optimization
    batch_size: int = 24                # per rank, per micro-batch
    accumulate_grad_batches: int = 1    # micro-batches per optimizer step
    max_steps: int = 30000
    warmup_steps: int = 500
    lr: float = 1.2e-3
    min_lr: float = 1e-4
    decay_frac: float = 0.2             # trailing fraction of max_steps spent decaying
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    compile: bool = True                # torch.compile the model (CUDA only)
    doc_mask: bool = True               # stop attention crossing document boundaries
    sampler_seed: int | None = None     # window order only; None = follow the source seed
    window_offset: int = 0              # skip this many windows of pass 0 (see window_seed)

    # Length-band oversampling; needs a pack="sorted" corpus. All read-time, never hashed.
    long_min_tokens: int | None = None  # lower bound of the oversampled band
    long_max_tokens: int | None = None  # optional upper: docs above it keep their natural rate
    long_frac: float = 0.0              # share of served windows drawn from the band

    # memory
    grad_checkpoint: bool = False       # recompute each block in backward

    # eval / checkpoint
    eval_every: int = 1000              # -> Trainer val_check_interval (steps)
    eval_iters: int = 18                # -> Trainer limit_val_batches
    init_from: Path | None = None       # checkpoint whose *weights* start this run
    resume_from: Path | None = None     # Lightning ckpt to resume exactly (momentum, schedule)
    out_dir: Path = field(default_factory=lambda: paths.CKPT_DIR)

    # locations (not identity — see CorpusSpec)
    data_dir: Path = field(default_factory=lambda: paths.DATA_DIR)
    tokenizer_path: Path = field(default_factory=lambda: paths.TOKENIZER_PATH)
    hf_cache_dir: str = paths.HF_CACHE_DIR

    # Lightning / distributed runtime
    accelerator: str = "auto"           # "auto" | "cuda" | "cpu"
    devices: str = "1"                  # GPU count (e.g. "1", "2"), or comma-list ("0,1")
    num_nodes: int = 1
    precision: str = "bf16-mixed"
    dataloader_workers: int = 4         # per-rank DataLoader workers
    wandb: bool = False
    wandb_project: str = "jlm"

    def to_record(self) -> dict:
        """Every knob that could change the result, JSON-ready, for the checkpoint history.

        Derived from the fields rather than a whitelist, so a new knob cannot be forgotten.
        Locations are excluded: they are machine-local, not part of what produced the
        weights. ``init_from`` stays — a stale parent path still beats none.
        """
        out = {}
        for f in dataclasses.fields(self):
            if f.name in _RECORD_EXCLUDED:
                continue
            v = getattr(self, f.name)
            out[f.name] = str(v) if isinstance(v, Path) else v
        return out


# Machine-local, so not part of what produced the weights.
_RECORD_EXCLUDED = frozenset({"out_dir", "data_dir", "tokenizer_path", "hf_cache_dir",
                              "resume_from"})


def window_seed(train_cfg: TrainConfig, source: SourceSpec) -> int:
    """Seed for the *window order only*, deliberately outside the corpus hash.

    ``SourceSpec.seed`` shuffles the document pool, so changing it changes which documents
    exist and which are held out — hence its place in the hash, and hence a re-encode when
    it moves. "Serve the same corpus in a different order" is a separate question, and a
    continuation run needs exactly that: without it the second run replays the window order
    the first one already trained on.

    A function rather than a property because the two seeds now live on different objects;
    keeping the explanation in one place is worth the call.
    """
    return source.seed if train_cfg.sampler_seed is None else train_cfg.sampler_seed


# ---------------------------------------------------------------- presets

def _shards(config: str, n: int, total: int) -> tuple[str, ...]:
    """The first ``n`` of a config's ``total`` parquet shards, as explicit filenames.

    Generated rather than typed out, but still fully specified: ``total`` is baked into each
    name, so if the repo is ever re-sharded the paths stop resolving and the build fails
    loudly. That is the property a glob would lose — it would silently match the new set and
    change the corpus under an unchanged hash.
    """
    return tuple(f"{config}/train-{i:05d}-of-{total:05d}.parquet" for i in range(n))


# SmolLM-Corpus. Measured with the v2 tokenizer: cosmopedia-v2 is 376,289 docs/shard at 728
# tok/doc (0.27B/shard, 28.5B over all 104); fineweb-edu-dedup is 812,684 docs/shard at 1,168
# tok/doc (0.95B/shard, 222.0B over all 234). ~70/30 fineweb/cosmopedia by tokens, which is
# roughly SmolLM2's own recipe: real filtered web text for breadth, synthetic textbook prose
# for the clean explanatory register a small model needs to read coherently.
#
# `python-edu` is deliberately absent. Its rows carry only blob_id/repo_name/path — the source
# itself lives in Software Heritage's S3 and needs a separate fetch (measured at 380 blobs/s
# over 64 threads, so ~5.6 h for all 7.68M). See `fetch-python-edu` if it is ever added.
_SMOLLM_MIX = (
    SourcePart("fineweb-edu-dedup", shards=_shards("fineweb-edu-dedup", 22, 234)),  # ~20.9B
    SourcePart("cosmopedia-v2", shards=_shards("cosmopedia-v2", 33, 104)),          # ~8.9B
)

_COSMOPEDIA_MIX = (                     # order is hashed: keep it stable
    SourcePart("auto_math_text"),       # 1.95M rows
    SourcePart("stanford"),             # 1.02M
    SourcePart("stories", cap=1_800_000),        # of 4.99M
    SourcePart("web_samples_v2", cap=1_800_000),  # of 10.3M
    SourcePart("wikihow"),              # 179k
    SourcePart("openstax"),             # 126k
    SourcePart("khanacademy"),          # 24k
)

# smoltalk2 SFT, the non-thinking splits minus multilingual, tool-calling and 64k-context.
# See CLAUDE.md for the row counts and why each of the four is out.
_SMOLTALK2_NOTHINK = (
    "OpenThoughts3_1.2M_no_think_no_think",
    "smoltalk_smollm3_smol_magpie_ultra_no_think",
    "OpenHermes_2.5_no_think",
    "smoltalk_smollm3_smol_summarize_no_think",
    "Mixture_of_Thoughts_science_no_think",
    "smoltalk_smollm3_smol_rewrite_no_think",
    "smoltalk_smollm3_systemchats_30k_no_think",
    "smoltalk_smollm3_explore_instruct_rewriting_no_think",
    "tulu_3_sft_personas_instruction_following_no_think",
    "table_gpt_no_think",
    "smoltalk_smollm3_everyday_conversations_no_think",
)


def _smollm() -> CorpusSpec:
    return CorpusSpec(
        source=SourceSpec("HuggingFaceTB/smollm-corpus", _SMOLLM_MIX),
        render=RenderSpec(kind="pretrain"),
    )


def _smollm_sorted() -> CorpusSpec:
    """``smollm`` ordered shortest-first — same tokens, and identical under uniform sampling;
    derive it with ``sort-corpus`` rather than re-encoding."""
    return CorpusSpec(
        source=SourceSpec("HuggingFaceTB/smollm-corpus", _SMOLLM_MIX),
        render=RenderSpec(kind="pretrain", pack="sorted"),
    )


def _cosmopedia() -> CorpusSpec:
    return CorpusSpec(
        source=SourceSpec("HuggingFaceTB/cosmopedia", _COSMOPEDIA_MIX),
        render=RenderSpec(kind="pretrain"),
    )


def _smoltalk2(block_size: int = 1024) -> CorpusSpec:
    return CorpusSpec(
        source=SourceSpec(
            "HuggingFaceTB/smoltalk2",
            tuple(SourcePart("SFT", s) for s in _SMOLTALK2_NOTHINK),
            columns=("messages",), n_val_docs=2_000),
        render=RenderSpec(kind="chat", max_example_tokens=block_size,
                          pack="bfd", pack_block=block_size),
    )


def _smoke() -> CorpusSpec:
    # One shard of the real corpus, capped to a few thousand docs. Points at the same files
    # the pretrain preset uses, so smoke costs no download of its own once that is built.
    return CorpusSpec(
        source=SourceSpec(
            "HuggingFaceTB/smollm-corpus",
            (SourcePart("cosmopedia-v2", shards=_shards("cosmopedia-v2", 1, 104)),),
            n_train_docs=2_000, n_val_docs=200),
        render=RenderSpec(kind="pretrain"),
    )


def _smoke_sorted() -> CorpusSpec:
    """``smoke`` ordered shortest-first — the cheap end-to-end test of band sampling."""
    spec = _smoke()
    return CorpusSpec(source=spec.source,
                      render=RenderSpec(kind="pretrain", pack="sorted"))


CORPUS_PRESETS: dict[str, Callable[[], CorpusSpec]] = {
    "smollm": _smollm,
    "smollm-sorted": _smollm_sorted,
    "cosmopedia": _cosmopedia,
    "smoltalk2": _smoltalk2,
    "smoke": _smoke,
    "smoke-sorted": _smoke_sorted,
}


def corpus_preset(name: str) -> CorpusSpec:
    """Look up a preset by name, listing the alternatives when it misses.

    Stamps the key onto the spec, so ``--corpus X`` and the cache filename cannot disagree.
    Letting each factory name itself is two strings that are always meant to match, and the
    failure is a filename that quietly describes a different corpus than the flag that built
    it.
    """
    if name not in CORPUS_PRESETS:
        raise ValueError(
            f"unknown corpus {name!r} — expected one of {sorted(CORPUS_PRESETS)}")
    spec = CORPUS_PRESETS[name]()
    spec.name = name
    return spec


def default_configs(smoke: bool = False) -> tuple[ModelConfig, TrainConfig, CorpusSpec]:
    """Return the (model, train, corpus) triple for a real run, or a tiny smoke run.

    Smoke shrinks every dimension for a fast end-to-end sanity check, disables
    ``torch.compile`` (compile warmup would dwarf 60 steps), and routes checkpoints to
    ``checkpoints/smoke/`` so a real run's output is never touched.
    """
    if not smoke:
        return ModelConfig(), TrainConfig(), corpus_preset("smollm")

    model = ModelConfig(hidden_dim=128, num_heads=4, n_layer=4, block_size=128)
    train = TrainConfig(
        batch_size=16, max_steps=60, warmup_steps=10,
        eval_every=20, eval_iters=10,
        compile=False, out_dir=paths.CKPT_DIR / "smoke",
        dataloader_workers=0,
    )
    return model, train, corpus_preset("smoke")
