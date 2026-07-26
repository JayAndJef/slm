"""PyTorch Lightning surface for training the JLM transformer.

Three pieces, kept separate from the model/data/tokenizer so those stay framework-free:

- :class:`LitJLM` — wraps :class:`~slm.model.JLM`; defines the train/val steps, the
  Muon+AdamW optimizer, and the warmup+cosine LR schedule.
- :class:`SLMDataModule` — builds the token corpus once (``prepare_data``, rank-0 only)
  and serves per-rank random-window DataLoaders.
- :class:`CompactCheckpoint` — saves the project's compact ``{model, step, val, config}``
  checkpoint on improved val loss, so :func:`slm.generate.load_model` loads it unchanged.
"""
import math
import time

import lightning as L
import numpy as np
import torch

from slm import checkpoint
from slm.config import ModelConfig, TrainConfig
from slm.data import build_corpus, load_docs
from slm.dataset import build_dataloader
from slm.model import JLM, segment_block_mask
from slm.tokenizer import load_tokenizer


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Trapezoidal (warmup-stable-decay) schedule: ramp up, hold, then decay to ``min_lr``.

    Preferred over cosine because the peak is held constant rather than decaying from step
    one, so the run length is not baked into the schedule: stop anywhere in the stable
    phase, run the decay, and you have a finished model. Cosine has to know ``max_steps``
    up front, and truncating it leaves the LR stranded mid-decay.

    The decay uses ``1 - sqrt(progress)``, which holds the peak longer than a linear ramp
    and empirically edges it out; the last ``decay_frac`` of the run is spent on it.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_steps = max(1, int(cfg.max_steps * cfg.decay_frac))
    stable_end = cfg.max_steps - decay_steps
    if step < stable_end:
        return cfg.lr
    prog = min(1.0, (step - stable_end) / decay_steps)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * (1 - math.sqrt(prog))


class MuonAdamW(torch.optim.Optimizer):
    """A Muon + AdamW pair presented to Lightning as one optimizer.

    ``torch.optim.Muon`` only accepts 2D parameters, so the hidden weight matrices and
    everything else (embeddings, lm_head, LayerNorm gains, biases) need separate
    optimizers — but Lightning's *automatic* optimization drives a single one. This holds
    both and exposes their own group dicts (the same objects, not copies) as
    ``param_groups``, so an LR scheduler writing ``group["lr"]`` reaches the real
    optimizer and Lightning's gradient clipping still sees every parameter.
    """

    def __init__(self, muon: torch.optim.Optimizer, adamw: torch.optim.Optimizer):
        self.opts = (muon, adamw)
        params = [p for o in self.opts for g in o.param_groups for p in g["params"]]
        super().__init__(params, {})    # gives us Optimizer's zero_grad / hook plumbing
        self.param_groups = [g for o in self.opts for g in o.param_groups]

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for o in self.opts:
            o.step()
        return loss

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, state_dict):
        for o, sd in zip(self.opts, state_dict["opts"]):
            o.load_state_dict(sd)


class LitJLM(L.LightningModule):
    """LightningModule wrapping JLM: (logits, loss) forward → train/val steps.

    ``init_state`` seeds the weights from a previous run (see :mod:`slm.checkpoint`). It is
    passed in already loaded rather than as a path, so this class stays free of filesystem
    access and can be constructed in a test from a bare state dict.
    """

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 init_state: dict | None = None):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.save_hyperparameters(model_cfg.to_dict())
        self.model = JLM.from_config(model_cfg)
        if init_state is not None:
            self.model.load_state_dict(init_state)
        if train_cfg.compile:
            self.model = torch.compile(self.model)
        self._t_prev = None

    def on_train_start(self):
        self._t_prev = time.perf_counter()

    def _block_mask(self, seg):
        """Build the per-step attention mask, or None for plain causal attention.

        Built here rather than inside ``JLM.forward``: it must stay outside torch.compile
        (create_block_mask traces its own kernel), and building it once per step shares it
        across all blocks instead of rebuilding it per layer.
        """
        if not self.train_cfg.doc_mask:
            return None
        return segment_block_mask(seg, compiled=self.train_cfg.compile)

    def training_step(self, batch, batch_idx):
        x, y, seg = batch
        _, loss = self.model(x, y, block_mask=self._block_mask(seg))
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("segs_per_window", seg.max(-1).values.float().mean() + 1, on_step=True)
        # Throughput across all ranks; x is (batch, block) on this rank.
        now = time.perf_counter()
        if self._t_prev is not None:
            dt = now - self._t_prev
            tok_s = x.numel() * self.trainer.world_size / dt if dt > 0 else 0.0
            self.log("tok_s", tok_s, prog_bar=True, on_step=True)
        self._t_prev = now
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, seg = batch
        _, loss = self.model(x, y, block_mask=self._block_mask(seg))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        tpb = self.train_cfg.tokens_per_byte
        if tpb:
            self.log("val_bpb", loss * tpb / math.log(2), prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        cfg = self.train_cfg
        is_hidden = lambda name, p: ".blocks." in name and p.ndim == 2
        muon_params = [p for n, p in self.named_parameters() if is_hidden(n, p)]
        other_params = [p for n, p in self.named_parameters() if not is_hidden(n, p)]
        opt = MuonAdamW(
            torch.optim.Muon(muon_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                             momentum=0.95, adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(other_params, lr=cfg.lr,
                              weight_decay=cfg.weight_decay, betas=(0.9, 0.95)),
        )
        # LambdaLR multiplies base lr by this factor; normalize lr_at to a multiplier.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: lr_at(step, cfg) / cfg.lr)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class SLMDataModule(L.LightningDataModule):
    """Builds the cached corpus once, then serves per-rank random-window loaders."""

    def __init__(self, train_cfg: TrainConfig, model_cfg: ModelConfig):
        super().__init__()
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.train_path = train_cfg.train_path
        self.val_path = train_cfg.val_path

    def prepare_data(self):
        # Lightning runs this on local-rank-0 only, before setup/DDP fan-out. Files only.
        cfg = self.cfg
        if self.train_path.exists() and self.val_path.exists():
            return
        train_docs, val_docs = load_docs(cfg)
        build_corpus(train_docs, self.train_path, cfg.tokenizer_path,
                     sep=cfg.sep, n_workers=cfg.n_workers, logger=print)
        build_corpus(val_docs, self.val_path, cfg.tokenizer_path,
                     sep=cfg.sep, n_workers=cfg.n_workers, logger=print)

    def setup(self, stage=None):
        self.rank = self.trainer.global_rank
        self.world_size = self.trainer.world_size
        # The boundary id is tokenizer-specific — each backend derives it in its own
        # vocabulary and asserts it is a single token (see Tokenizer.sep_id).
        tok = load_tokenizer(self.cfg.tokenizer_path)
        self.sep_id = tok.sep_id(self.cfg.sep)
        # Tokens per byte, measured off the real corpus, so val loss can be reported as
        # bits per byte — the only figure comparable across vocab sizes.
        sample = np.load(self.val_path, mmap_mode="r")[:200_000]
        self.cfg.tokens_per_byte = len(sample) / max(1, len(tok.decode(sample.tolist()).encode()))

    def train_dataloader(self):
        return build_dataloader(
            self.train_path, block_size=self.model_cfg.block_size,
            batch_size=self.cfg.batch_size, seed=self.cfg.window_seed,
            rank=self.rank, world_size=self.world_size,
            num_workers=self.cfg.dataloader_workers, sep_id=self.sep_id)

    def val_dataloader(self):
        return build_dataloader(
            self.val_path, block_size=self.model_cfg.block_size,
            batch_size=self.cfg.batch_size, seed=self.cfg.seed + 10_000,
            rank=self.rank, world_size=self.world_size,
            num_workers=self.cfg.dataloader_workers, sep_id=self.sep_id)


class CompactCheckpoint(L.Callback):
    """Save the compact checkpoint (see :mod:`slm.checkpoint`) on improved val loss.

    ``best`` seeds the best-so-far. It defaults to infinity — meaning the first validation
    always writes — which is right for a run starting from scratch and wrong for one
    continuing from trained weights: re-warming makes val temporarily *worse*, so an
    unseeded continuation would overwrite a good checkpoint with a worse one at its first
    eval. :func:`slm.train.train` seeds it from the checkpoint being continued.
    """

    def __init__(self, out_dir, model_cfg: ModelConfig, best: float = float("inf")):
        super().__init__()
        self.out_dir = out_dir
        self.model_cfg = model_cfg
        self.best = best

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val = trainer.callback_metrics.get("val_loss")
        if val is None:
            return
        val = float(val)
        if val >= self.best:
            return
        self.best = val                       # kept in sync on all ranks (val is all-reduced)
        if not trainer.is_global_zero:
            return
        checkpoint.save(self.out_dir / "best.pt", pl_module.model,
                        step=trainer.global_step, val=val, model_cfg=self.model_cfg)
