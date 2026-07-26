"""Training entrypoint: build a Lightning Trainer and fit.

Thin driver — the model/step/optimizer logic lives in :mod:`slm.lit`. This assembles the
logger and callbacks, parses the device spec, and calls ``trainer.fit``. Multi-GPU is
data-parallel (DDP): pass ``devices`` as a count or a comma-list of PyTorch device indices.
"""
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar

from slm.config import ModelConfig, TrainConfig
from slm.lit import CompactCheckpoint, LitJLM, SLMDataModule


def _parse_devices(spec: str):
    """'3' -> 3 (a count); '0,1' -> [0, 1] (explicit PyTorch indices)."""
    spec = str(spec).strip()
    if "," in spec:
        return [int(s) for s in spec.split(",") if s.strip() != ""]
    return int(spec)


def _make_logger(cfg: TrainConfig):
    if cfg.wandb:
        from lightning.pytorch.loggers import WandbLogger  # lazy: only when --wandb
        return WandbLogger(project=cfg.wandb_project)
    from lightning.pytorch.loggers import CSVLogger
    return CSVLogger(save_dir=str(cfg.out_dir), name="logs")


def train(model_cfg: ModelConfig, train_cfg: TrainConfig):
    """Fit a model with Lightning; returns the path to the compact best checkpoint."""
    import torch
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
            CompactCheckpoint(train_cfg.out_dir, model_cfg),
            RichProgressBar(),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=_make_logger(train_cfg),
    )
    lit = LitJLM(model_cfg, train_cfg)
    dm = SLMDataModule(train_cfg, model_cfg)
    trainer.fit(lit, datamodule=dm)
    return train_cfg.out_dir / "best.pt"
