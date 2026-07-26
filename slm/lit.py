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

from slm.config import ModelConfig, TrainConfig
from slm.data import build_corpus, load_docs
from slm.dataset import build_dataloader
from slm.model import JLM, segment_block_mask
from slm.tokenizer import SimpleTokenizer


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup for ``warmup_steps``, then cosine decay to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * prog))


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
    """LightningModule wrapping JLM: (logits, loss) forward → train/val steps."""

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.save_hyperparameters(model_cfg.to_dict())
        self.model = JLM.from_config(model_cfg)
        if train_cfg.compile:
            # Compile the inner module; Lightning wraps the LightningModule in DDP outside it.
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
        # The separator's id is tokenizer-specific — derive it, never hardcode it.
        # encode(sep) returns TWO ids: the tokenizer's split pattern peels the trailing
        # "\n" off into its own chunk. That trailing newline is not what marks a boundary
        # in the corpus — build_corpus encodes sep+doc contiguously, so it is absorbed into
        # the following word and ids[0] (b"\n<|endoftext|>") is the lone boundary token.
        tok = SimpleTokenizer.load(self.cfg.tokenizer_path)
        ids = tok.encode(self.cfg.sep)
        marker = self.cfg.sep.strip().encode()
        # Guards the one failure that would ruin a run silently: if the tokenizer never
        # learned the separator as a merge, ids[0] is a bare "\n" (3196 occurrences per 2M
        # tokens vs 1576), so every newline reads as a document boundary — training still
        # converges and every conclusion is wrong.
        assert marker in tok.vocab[ids[0]], (
            f"separator token {ids[0]} is {tok.vocab[ids[0]]!r}, which does not carry "
            f"{marker!r} — the tokenizer did not learn {self.cfg.sep!r} as one token")
        self.sep_id = ids[0]

    def train_dataloader(self):
        return build_dataloader(
            self.train_path, block_size=self.model_cfg.block_size,
            batch_size=self.cfg.batch_size, seed=self.cfg.seed,
            rank=self.rank, world_size=self.world_size,
            num_workers=self.cfg.dataloader_workers, sep_id=self.sep_id)

    def val_dataloader(self):
        return build_dataloader(
            self.val_path, block_size=self.model_cfg.block_size,
            batch_size=self.cfg.batch_size, seed=self.cfg.seed + 10_000,
            rank=self.rank, world_size=self.world_size,
            num_workers=self.cfg.dataloader_workers, sep_id=self.sep_id)


class CompactCheckpoint(L.Callback):
    """Save the compact ``{model, step, val, config}`` checkpoint on improved val loss.

    Writes the plain JLM state_dict (stripping any ``torch.compile`` ``_orig_mod.``
    prefix) so :func:`slm.generate.load_model` loads it with ``strict=True``.
    """

    def __init__(self, out_dir, model_cfg: ModelConfig):
        super().__init__()
        self.out_dir = out_dir
        self.model_cfg = model_cfg
        self.best = float("inf")

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
        core = getattr(pl_module.model, "_orig_mod", pl_module.model)
        state = {k: v.detach().cpu() for k, v in core.state_dict().items()}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": state, "step": trainer.global_step, "val": val,
                    "config": self.model_cfg.to_dict()}, self.out_dir / "best.pt")
