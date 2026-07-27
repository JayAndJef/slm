"""PyTorch Lightning surface for training the JLM transformer.

Two pieces, kept separate from the model/corpus/tokenizer so those stay framework-free:

- :class:`LitJLM` — wraps :class:`~slm.model.JLM`; defines the train/val steps, the
  Muon+AdamW optimizer, and the warmup-stable-decay LR schedule.
- :class:`SLMDataModule` — serves per-rank window DataLoaders over corpora that already
  exist. It never builds one: see its docstring.

Checkpointing is Lightning's ``ModelCheckpoint`` (configured in :mod:`slm.train`), not a
callback here. It already tracks best-by-val, and additionally carries optimizer and
scheduler state — which is what makes an interrupted run resumable, and what the
save-on-improvement-only callback this replaces could not do.
"""
import math
import time

import lightning as L
import numpy as np
import torch

from slm.config import CorpusSpec, ModelConfig, TrainConfig, window_seed
from slm.dataset import build_dataloader, n_windows
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

    ``provenance`` is written into the checkpoint alongside the architecture, because
    ``hyper_parameters`` is the only part of a Lightning checkpoint whose contents this
    project chooses, and :func:`slm.checkpoint.load` turns it into ``meta``. Without it a
    fine-tuned model is indistinguishable from a base one at generation time and ``--chat``
    has to be remembered by hand. ``ModelConfig.from_dict`` ignores the extra keys, so the
    architecture read back is unaffected.

    ``save_hyperparameters`` is given an explicit dict, never called bare: a bare call
    captures ``__init__``'s arguments, which would write ``init_state``'s ~800 MB into
    every checkpoint.
    """

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 init_state: dict | None = None, tokens_per_byte: float | None = None,
                 provenance: dict | None = None):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.tokens_per_byte = tokens_per_byte
        self.save_hyperparameters({**model_cfg.to_dict(), **(provenance or {})},
                                  ignore=["init_state"])
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

    @staticmethod
    def _safe_targets(y):
        """``(targets, scale)`` for a batch that may be entirely ``-100``.

        Cross-entropy over an all-ignored batch is 0/0 = NaN, and under DDP a NaN poisons
        the all-reduce. Returning ``None`` from the step instead would *hang*: that rank
        skips backward while the others block in NCCL until the watchdog fires.

        ``torch.where`` on the loss is also not enough — it backpropagates the unselected
        branch, and ``0 * NaN`` is NaN. So the substitution happens on the targets, before
        the forward, leaving the graph finite end to end.
        """
        n = (y != -100).sum()
        return (y if n > 0 else torch.zeros_like(y)), (n > 0).float()

    def training_step(self, batch, batch_idx):
        x, y, seg = batch
        targets, scale = self._safe_targets(y)
        _, loss = self.model(x, targets, block_mask=self._block_mask(seg))
        loss = loss * scale
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("frac_loss_tokens", (y != -100).float().mean(), on_step=True)
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
        targets, scale = self._safe_targets(y)
        _, loss = self.model(x, targets, block_mask=self._block_mask(seg))
        loss = loss * scale
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        tpb = self.tokens_per_byte
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
    """Serves per-rank window loaders over corpora that already exist.

    Takes built :class:`~slm.corpus.Corpus` objects rather than building them. Lightning's
    DDP strategy re-executes ``sys.argv`` per rank, so encoding here would race several
    processes on a multi-GB write; ``prepare-data`` owns that, and the CLI checks the corpus
    exists before a Trainer is ever constructed.
    """

    def __init__(self, train_cfg: TrainConfig, model_cfg: ModelConfig,
                 spec: CorpusSpec, train_corpus, val_corpus):
        super().__init__()
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.spec = spec
        self.train_corpus = train_corpus
        self.val_corpus = val_corpus
        self.tokens_per_byte = val_corpus.meta["tokens_per_byte"]

    def setup(self, stage=None):
        self.rank = self.trainer.global_rank
        self.world_size = self.trainer.world_size
        fingerprint = load_tokenizer(self.cfg.tokenizer_path).fingerprint
        for c in (self.train_corpus, self.val_corpus):
            c.assert_compatible(self.model_cfg, tokenizer_fingerprint=fingerprint,
                                doc_mask=self.cfg.doc_mask)

        clean = max(1, n_windows(self.val_corpus.meta["n_tokens"], self.model_cfg.block_size)
                    // max(1, self.cfg.batch_size * self.world_size))
        if self.cfg.eval_iters != clean and self.trainer.is_global_zero:
            print(f"eval_iters={self.cfg.eval_iters} but one clean pass over "
                  f"{self.val_corpus.name} is {clean} at batch {self.cfg.batch_size} x "
                  f"{self.world_size} ranks — larger resamples the same windows")

    def _loader(self, corpus, seed: int):
        return build_dataloader(
            block_size=self.model_cfg.block_size, batch_size=self.cfg.batch_size, seed=seed,
            rank=self.rank, world_size=self.world_size,
            num_workers=self.cfg.dataloader_workers,
            mask_partial_head=self.spec.render.mask_partial_head,
            **corpus.loader_kwargs())

    def train_dataloader(self):
        return self._loader(self.train_corpus, window_seed(self.cfg, self.spec.source))

    def val_dataloader(self):
        return self._loader(self.val_corpus, self.spec.source.seed + 10_000)
