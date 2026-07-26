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
from dataclasses import dataclass, field
from pathlib import Path

from slm import paths


@dataclass
class ModelConfig:
    """The five dimensions that define a JLM transformer's architecture."""

    vocab_size: int = 4096
    hidden_dim: int = 768
    num_heads: int = 12
    n_layer: int = 12
    block_size: int = 512

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Build from a (possibly extra-keyed) dict, ignoring unknown fields.

        Filtering rather than ``cls(**d)`` keeps old checkpoints loadable if future
        configs gain new fields.
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
        "stories": 1_000_000,           # of 4.99M
        "web_samples_v2": 1_000_000,    # of 10.3M
        "wikihow": None,                # 179k
        "openstax": None,               # 126k
        "khanacademy": None,            # 24k
    })
    n_train_docs: int | None = None     # cap on training docs after mixing; None = all
    n_val_docs: int = 5_000
    seed: int = 42
    sep: str = "\n<|endoftext|>\n"      # literal document separator (BPE learned it as one token)
    n_workers: int = 8

    # optimization
    batch_size: int = 160
    max_steps: int = 6000
    warmup_steps: int = 150
    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    compile: bool = True                # torch.compile the model (CUDA only)
    doc_mask: bool = True               # stop attention crossing document boundaries

    # eval / checkpoint
    eval_every: int = 500               # -> Trainer val_check_interval (steps)
    eval_iters: int = 50                # -> Trainer limit_val_batches
    tag: str = "cosmo"                  # names the cached corpus files
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
    wandb_project: str = "slm"

    @property
    def train_path(self) -> Path:
        """Cached train corpus. Named by ``tag`` + the doc cap, so changing the cap makes
        a new file rather than silently reusing the old one."""
        return self.data_dir / f"train_{self.tag}_{self.n_train_docs or 'all'}.npy"

    @property
    def val_path(self) -> Path:
        return self.data_dir / f"val_{self.tag}_{self.n_val_docs}.npy"


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
