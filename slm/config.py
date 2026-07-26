"""Typed configuration for the model architecture and a training run.

Two dataclasses keep the knobs in one place and make functions depend on plain data
rather than module globals (loose coupling):

- :class:`ModelConfig` — the five architecture dimensions. Its fields are exactly the
  keys stored in a checkpoint's ``config`` dict, so a saved model round-trips through
  :meth:`ModelConfig.from_dict` / :meth:`ModelConfig.to_dict`.
- :class:`TrainConfig` — everything about a training run (data, optimization, eval,
  checkpointing, runtime).

:func:`default_configs` is the single source of truth for the tiny "smoke" overrides.
"""
import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from slm import paths


@dataclass
class ModelConfig:
    """The five dimensions that define a JLM transformer's architecture."""

    vocab_size: int = 32_000
    hidden_dim: int = 1024
    num_heads: int = 16
    n_layer: int = 12
    block_size: int = 1024

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
    """Settings for one training run: data source, optimization, eval, runtime."""

    # data
    dataset_name: str = "HuggingFaceTB/cosmopedia"
    dataset_mix: dict[str, int | None] = field(default_factory=lambda: {
        "auto_math_text": None,         # 1.95M rows
        "stanford": None,               # 1.02M
        "stories": 1_800_000,           # of 4.99M
        "web_samples_v2": 1_800_000,    # of 10.3M
        "wikihow": None,                # 179k
        "openstax": None,               # 126k
        "khanacademy": None,            # 24k
    })
    n_train_docs: int | None = None     # cap on training docs after mixing; None = all
    n_val_docs: int = 5_000
    seed: int = 42
    sampler_seed: int | None = None     # window order only; None = follow `seed`
    sep: str = "\n<|endoftext|>\n"      # literal document separator (one token id in the corpus)
    n_workers: int = 16
    tokens_per_byte: float | None = None  # measured from the corpus at setup; drives val_bpb

    # optimization
    batch_size: int = 48
    max_steps: int = 30000
    warmup_steps: int = 100
    lr: float = 1e-3
    min_lr: float = 1e-4
    decay_frac: float = 0.2             # trailing fraction of max_steps spent decaying
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    compile: bool = True                # torch.compile the model (CUDA only)
    doc_mask: bool = True               # stop attention crossing document boundaries

    # eval / checkpoint
    eval_every: int = 1000               # -> Trainer val_check_interval (steps)
    eval_iters: int = 18                # -> Trainer limit_val_batches
    tag: str = "cosmo"                  # names the cached corpus files
    init_from: Path | None = None       # compact checkpoint whose weights start this run
    out_dir: Path = field(default_factory=lambda: paths.CKPT_DIR)
    data_dir: Path = field(default_factory=lambda: paths.DATA_DIR)
    tokenizer_path: Path = field(default_factory=lambda: paths.TOKENIZER_PATH)
    hf_cache_dir: str = paths.HF_CACHE_DIR

    # Lightning / distributed runtime
    accelerator: str = "auto"           # "auto" | "cuda" | "cpu"
    devices: str = "1"                  # GPU count (e.g. "1", "2"), or comma-list of indices ("0,1")
    num_nodes: int = 1
    precision: str = "bf16-mixed"
    dataloader_workers: int = 2         # per-rank DataLoader workers (distinct from n_workers)
    wandb: bool = False
    wandb_project: str = "jlm"

    @property
    def corpus_hash(self) -> str:
        """Short digest of everything that determines the token stream.

        Every input here fails *silently* when it drifts. A changed tokenizer leaves all
        ids in range, so nothing crashes, the loss curve looks normal, and the stream means
        nothing. A changed ``dataset_mix`` or ``seed`` reshuffles the pool, which moves the
        val split — losses stop being comparable with no signal at all. A changed ``sep``
        changes the boundary token the document mask keys off.

        Folding them into the filename turns each of those into a visible re-encode instead
        of a wrong run. ``n_workers`` is deliberately absent: it permutes the order shards
        are concatenated in, but not which documents are present, and windows are shuffled
        anyway.
        """
        key = repr((sorted(self.dataset_mix.items()), self.seed, self.sep, self.dataset_name))
        return hashlib.sha256(
            Path(self.tokenizer_path).read_bytes() + key.encode()).hexdigest()[:8]

    @property
    def window_seed(self) -> int:
        """Seed for the *window order only*, deliberately outside :attr:`corpus_hash`.

        ``seed`` shuffles the document pool, so changing it changes which documents exist
        and which are held out — hence its place in the hash, and hence a re-encode when it
        moves. "Serve the same corpus in a different order" is a separate question, and a
        continuation run needs exactly that: without it the second run replays the window
        order the first one already trained on.
        """
        return self.seed if self.sampler_seed is None else self.sampler_seed

    @property
    def train_path(self) -> Path:
        """Cached train corpus: tag, both doc counts, and :attr:`corpus_hash`. Train is
        ``pool[n_val_docs:][:n_train_docs]``, so ``n_val_docs`` moves it too."""
        return self.data_dir / (
            f"train_{self.tag}_{self.n_val_docs}+{self.n_train_docs or 'all'}"
            f"_{self.corpus_hash}.npy")

    @property
    def val_path(self) -> Path:
        return self.data_dir / f"val_{self.tag}_{self.n_val_docs}_{self.corpus_hash}.npy"


def default_configs(smoke: bool = False) -> tuple[ModelConfig, TrainConfig]:
    """Return the (model, train) config pair for a real run, or a tiny smoke run.

    Smoke mode shrinks every dimension for a fast end-to-end sanity check, disables
    ``torch.compile`` (compile warmup would dwarf 60 steps), and routes checkpoints to
    ``checkpoints/smoke/`` so a real ``best.pt`` is never touched.
    """
    if not smoke:
        return ModelConfig(), TrainConfig()

    model = ModelConfig(hidden_dim=128, num_heads=4, n_layer=4, block_size=128)
    train = TrainConfig(
        # One tiny config (24k rows, 46 MB) — a capped load still downloads whole HF
        # configs, so smoke must narrow the mix, not just the doc count.
        dataset_mix={"khanacademy": None},
        n_train_docs=2_000, n_val_docs=200,
        batch_size=16, max_steps=60, warmup_steps=10,
        eval_every=20, eval_iters=10,
        compile=False, tag="smoke", out_dir=paths.CKPT_DIR / "smoke",
        dataloader_workers=0,
    )
    return model, train
