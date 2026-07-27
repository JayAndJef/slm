"""Training entrypoint: build a Lightning Trainer and fit.

Thin driver — the model/step/optimizer logic lives in :mod:`slm.lit`. This assembles the
logger and callbacks, parses the device spec, and calls ``trainer.fit``. Multi-GPU is
data-parallel (DDP): pass ``devices`` as a count or a comma-list of PyTorch device indices.
"""
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from slm import checkpoint
from slm.config import ModelConfig, TrainConfig
from slm.lit import LitJLM, SLMDataModule
from slm.tokenizer import load_tokenizer


def parse_devices(spec: str):
    """'3' -> 3 (a count); '0,1' -> [0, 1] (explicit PyTorch indices).

    Public because the CLI needs the same answer to size an epoch: a second copy of this
    got ``--devices -1`` (Lightning's "every GPU" sentinel) wrong. See :func:`n_devices`.
    """
    spec = str(spec).strip()
    if "," in spec:
        return [int(s) for s in spec.split(",") if s.strip() != ""]
    return int(spec)


def n_devices(spec: str) -> int:
    """How many ranks ``spec`` actually resolves to, resolving the ``-1`` sentinel."""
    devices = parse_devices(spec)
    if isinstance(devices, list):
        return len(devices)
    return torch.cuda.device_count() if devices < 0 else max(1, devices)


def _make_logger(cfg: TrainConfig):
    if cfg.wandb:
        return WandbLogger(project=cfg.wandb_project)
    return CSVLogger(save_dir=str(cfg.out_dir), name="logs")


def train(model_cfg: ModelConfig | None, train_cfg: TrainConfig,
          spec, train_corpus, val_corpus):
    """Fit a model with Lightning; returns the path to the compact best checkpoint.

    With ``train_cfg.init_from`` set, the run starts from that checkpoint's **weights** —
    not its optimizer moments, scheduler position or dataloader offset, none of which the
    compact format stores. The LR schedule therefore starts from scratch, so a continuation
    must be given one that makes sense from a trained model (a bare anneal, or a re-warm)
    rather than the schedule that produced it.

    The architecture then comes from the checkpoint and ``model_cfg`` is ignored — weights
    can only be rebuilt at the dimensions they were trained at — so callers that continue a
    run may pass ``None``.

    Every checkpoint written here records which corpus and tokenizer produced it. That is
    what lets ``generate`` pick ChatML for a fine-tuned model without being told, and what
    lets any consumer refuse a tokenizer the weights were not trained against; both are
    silent failures otherwise, and neither is recoverable from the weights.
    """
    init_state = None
    if train_cfg.init_from is not None:
        init_state, model_cfg, meta = checkpoint.load(train_cfg.init_from)
        val_str = "n/a" if meta["val"] is None else f"{meta['val']:.4f}"
        print(f"continuing from {train_cfg.init_from} "
              f"(step {meta['step']}, val {val_str}) — weights only")
    assert model_cfg is not None, "pass a ModelConfig, or set train_cfg.init_from"

    # The vocab/tokenizer asserts live on the corpus now (SLMDataModule.setup), where they
    # check the stream the model will actually read rather than a tokenizer file beside it.
    torch.set_float32_matmul_precision("high")   # use Tensor Cores for fp32 matmuls
    devices = parse_devices(train_cfg.devices)
    n_ranks = len(devices) if isinstance(devices, list) else devices
    strategy = "ddp" if n_ranks and n_ranks > 1 else "auto"

    trainer = L.Trainer(
        accelerator=train_cfg.accelerator,
        devices=devices,
        num_nodes=train_cfg.num_nodes,
        strategy=strategy,
        precision=train_cfg.precision,
        max_steps=train_cfg.max_steps,
        max_epochs=-1,
        gradient_clip_val=train_cfg.grad_clip,
        val_check_interval=train_cfg.eval_every,
        limit_val_batches=train_cfg.eval_iters,
        use_distributed_sampler=False,          # IterableDataset shards via per-rank seed
        log_every_n_steps=max(1, train_cfg.eval_every // 10),
        callbacks=[
            # Lightning owns checkpointing. save_last is the *latest*, which is what a
            # resume needs; save_top_k=1 on val_loss is the model a continuation or SFT
            # should start from. save_on_exception covers a crash mid-run, which the
            # save-on-improvement-only callback this replaces did not.
            ModelCheckpoint(dirpath=train_cfg.out_dir, monitor="val_loss", mode="min",
                            save_top_k=1, save_last=True, save_on_exception=True,
                            filename="step{step}-val{val_loss:.4f}",
                            auto_insert_metric_name=False),
            RichProgressBar(),
        ],
        logger=_make_logger(train_cfg),
    )
    dm = SLMDataModule(train_cfg, model_cfg, spec, train_corpus, val_corpus)
    provenance = {
        "corpus": {"name": spec.name, "hash": train_corpus.hash,
                   "render_kind": spec.render.kind},
        "tokenizer_fingerprint": load_tokenizer(train_cfg.tokenizer_path).fingerprint,
        "tokens_per_byte": dm.tokens_per_byte,      # val_bpb = val_loss * this / ln 2
    }
    lit = LitJLM(model_cfg, train_cfg, init_state=init_state,
                 tokens_per_byte=dm.tokens_per_byte, provenance=provenance)
    del init_state
    trainer.fit(lit, datamodule=dm, ckpt_path=train_cfg.resume_from)
    return getattr(trainer.checkpoint_callback, "best_model_path", None)
