"""Training entrypoint: build a Lightning Trainer and fit.

Thin driver — the model/step/optimizer logic lives in :mod:`slm.lit`. This assembles the
logger and callbacks, parses the device spec, and calls ``trainer.fit``. Multi-GPU is
data-parallel (DDP): pass ``devices`` as a count or a comma-list of PyTorch device indices.
"""
import lightning as L
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from slm import checkpoint
from slm.config import ModelConfig, TrainConfig
from slm.lit import CompactCheckpoint, LitJLM, SLMDataModule
from slm.tokenizer import load_tokenizer


def _parse_devices(spec: str):
    """'3' -> 3 (a count); '0,1' -> [0, 1] (explicit PyTorch indices)."""
    spec = str(spec).strip()
    if "," in spec:
        return [int(s) for s in spec.split(",") if s.strip() != ""]
    return int(spec)


def _make_logger(cfg: TrainConfig):
    if cfg.wandb:
        return WandbLogger(project=cfg.wandb_project)
    return CSVLogger(save_dir=str(cfg.out_dir), name="logs")


def train(model_cfg: ModelConfig | None, train_cfg: TrainConfig):
    """Fit a model with Lightning; returns the path to the compact best checkpoint.

    With ``train_cfg.init_from`` set, the run starts from that checkpoint's **weights** —
    not its optimizer moments, scheduler position or dataloader offset, none of which the
    compact format stores. The LR schedule therefore starts from scratch, so a continuation
    must be given one that makes sense from a trained model (a bare anneal, or a re-warm)
    rather than the schedule that produced it.

    The architecture then comes from the checkpoint and ``model_cfg`` is ignored — weights
    can only be rebuilt at the dimensions they were trained at — so callers that continue a
    run may pass ``None``.
    """
    init_state, best = None, float("inf")
    if train_cfg.init_from is not None:
        init_state, model_cfg, meta = checkpoint.load(train_cfg.init_from)
        if meta["val"] is not None:
            best = meta["val"]          # only improvements on the source model get saved
        val_str = "n/a" if meta["val"] is None else f"{meta['val']:.4f}"
        print(f"continuing from {train_cfg.init_from} "
              f"(step {meta['step']}, val {val_str}) — weights only")
    assert model_cfg is not None, "pass a ModelConfig, or set train_cfg.init_from"

    n_vocab = load_tokenizer(train_cfg.tokenizer_path).n_vocab
    assert model_cfg.vocab_size == n_vocab, (
        f"model vocab_size {model_cfg.vocab_size} != tokenizer {n_vocab} "
        f"({train_cfg.tokenizer_path})")

    torch.set_float32_matmul_precision("high")   # use Tensor Cores for fp32 matmuls
    devices = _parse_devices(train_cfg.devices)
    n_devices = len(devices) if isinstance(devices, list) else devices
    strategy = "ddp" if n_devices and n_devices > 1 else "auto"

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
            CompactCheckpoint(train_cfg.out_dir, model_cfg, best=best),
            RichProgressBar(),
        ],
        logger=_make_logger(train_cfg),
    )
    lit = LitJLM(model_cfg, train_cfg, init_state=init_state)
    del init_state
    dm = SLMDataModule(train_cfg, model_cfg)
    trainer.fit(lit, datamodule=dm)
    return train_cfg.out_dir / "best.pt"
