"""Training entrypoint: build a Lightning Trainer and fit.

Thin driver — the model/step/optimizer logic lives in :mod:`slm.lit`. This assembles the
logger and callbacks, parses the device spec, and calls ``trainer.fit``. Multi-GPU is
data-parallel (DDP): pass ``devices`` as a count or a comma-list of PyTorch device indices.
"""
import hashlib
import subprocess
import uuid
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from slm import checkpoint
from slm.config import ModelConfig, TrainConfig
from slm.lit import LitJLM, SLMDataModule
from slm.tokenizer import load_tokenizer

_SAFE_OVERRIDES = ("block_size", "rope_theta")


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


def tokens_per_update(train_cfg: TrainConfig, block_size: int) -> int:
    """Tokens per *optimizer* step: four multipliers, of which ``batch_size`` is only the
    first (it is per rank per micro-batch; DDP and accumulation multiply it again).

    Beside :func:`n_devices` because that is where the world-size subtlety already lives.
    """
    return (train_cfg.batch_size * block_size
            * n_devices(train_cfg.devices) * train_cfg.num_nodes
            * train_cfg.accumulate_grad_batches)


def _make_logger(cfg: TrainConfig, run_id: str):
    """Pass the run id through, so a restart appends to that run instead of forking a new one."""
    if cfg.wandb:
        return WandbLogger(project=cfg.wandb_project, name=f"{cfg.out_dir.name}-{run_id}",
                           id=run_id, resume="allow")
    return CSVLogger(save_dir=str(cfg.out_dir), name="logs")


def _git_stamp() -> str:
    """``git describe``, plus a digest of the uncommitted diff — a bare sha claims this code
    produced these weights, which is false whenever anything is uncommitted."""
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                             text=True, timeout=5, check=True).stdout.strip()
        if out.endswith("-dirty"):
            diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True,
                                  timeout=5, check=True).stdout
            out += "+" + hashlib.sha256(diff).hexdigest()[:8]
        return out
    except Exception:
        return "unknown"


def _run_record(train_cfg, model_cfg, spec, corpus_hash, tokens_per_byte, fingerprint,
                world_size, run_id, command) -> dict:
    """This run's entry in the checkpoint history: config, provenance, empty results."""
    # batch_size is per rank per micro-batch; the optimization dynamics follow the product.
    effective_batch = (train_cfg.batch_size * world_size
                       * train_cfg.accumulate_grad_batches)
    return {
        "run_id": run_id,
        "command": command,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **train_cfg.to_record(),
        **model_cfg.to_dict(),
        "corpus": {"name": spec.name, "hash": corpus_hash, "render_kind": spec.render.kind},
        "tokenizer_fingerprint": fingerprint,
        "tokens_per_byte": tokens_per_byte,
        "world_size": world_size,
        "effective_batch": effective_batch,
        "tokens_per_update": effective_batch * model_cfg.block_size,
        "git": _git_stamp(),
        "updates": None, "val_loss": None, "val_bpb": None,
        "tokens_seen": 0, "tokens_seen_total": None, "wall_s": None,
    }


def _parent_history(meta: dict) -> list:
    """A parent checkpoint's run chain, synthesizing one entry for files written before
    history existed so the chain is never silently empty."""
    if meta.get("history"):
        return list(meta["history"])
    if not (meta.get("corpus") or meta.get("tokenizer_fingerprint")):
        return []
    return [{"command": "unknown", "backfilled": True, "corpus": meta.get("corpus"),
             "tokenizer_fingerprint": meta.get("tokenizer_fingerprint"),
             "tokens_per_byte": meta.get("tokens_per_byte"),
             "updates": meta.get("step"), "val_loss": meta.get("val")}]


def train(model_cfg: ModelConfig | None, train_cfg: TrainConfig,
          spec, train_corpus, val_corpus, model_overrides: dict | None = None,
          command: str | None = None, target_tokens: float | None = None):
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
    init_state, history = None, []
    if train_cfg.init_from is not None:
        init_state, model_cfg, meta = checkpoint.load(train_cfg.init_from)
        history = _parent_history(meta)
        val_str = "n/a" if meta["val"] is None else f"{meta['val']:.4f}"
        print(f"continuing from {train_cfg.init_from} "
              f"(step {meta['step']}, val {val_str}) — weights only")
    assert model_cfg is not None, "pass a ModelConfig, or set train_cfg.init_from"

    for field, value in (model_overrides or {}).items():
        assert field in _SAFE_OVERRIDES, (
            f"cannot override {field!r} on a continuation — only {_SAFE_OVERRIDES} leave "
            f"the parameter shapes alone, anything else fails load_state_dict(strict=True)")
        print(f"{field}: {getattr(model_cfg, field)} -> {value}")
        setattr(model_cfg, field, value)

    # The vocab/tokenizer asserts live on the corpus now (SLMDataModule.setup), where they
    # check the stream the model will actually read rather than a tokenizer file beside it.
    torch.set_float32_matmul_precision("high")   # use Tensor Cores for fp32 matmuls
    devices = parse_devices(train_cfg.devices)
    n_ranks = n_devices(train_cfg.devices)
    strategy = "ddp" if n_ranks > 1 else "auto"

    if target_tokens is not None:
        per_update = tokens_per_update(train_cfg, model_cfg.block_size)
        train_cfg.max_steps = max(1, round(target_tokens / per_update))
        print(f"{target_tokens/1e9:.3f}B tokens = {train_cfg.max_steps:,} steps at "
              f"{per_update:,} tokens/step ({train_cfg.batch_size} per rank x "
              f"{n_ranks * train_cfg.num_nodes} ranks x "
              f"{train_cfg.accumulate_grad_batches} accum x {model_cfg.block_size})")

    # Before the logger: a resume must reuse the run id or W&B forks a new run.
    resumed_history = []
    if train_cfg.resume_from is not None:
        resumed_history = _parent_history(checkpoint.load(train_cfg.resume_from)[2])
    prior = resumed_history[-1] if resumed_history else {}
    run_id = prior.get("run_id") or uuid.uuid4().hex[:8]

    trainer = L.Trainer(
        accelerator=train_cfg.accelerator,
        devices=devices,
        num_nodes=train_cfg.num_nodes,
        strategy=strategy,
        precision=train_cfg.precision,
        max_steps=train_cfg.max_steps,
        max_epochs=-1,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
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
        logger=_make_logger(train_cfg, run_id),
    )
    dm = SLMDataModule(train_cfg, model_cfg, spec, train_corpus, val_corpus)
    fingerprint = load_tokenizer(train_cfg.tokenizer_path).fingerprint
    record = _run_record(train_cfg, model_cfg, spec, train_corpus.hash, dm.tokens_per_byte,
                         fingerprint, n_ranks * train_cfg.num_nodes, run_id,
                         command or ("continue-train" if train_cfg.init_from else "train"))

    if train_cfg.resume_from is not None:
        # Replaces the last record. The new one wins outright; only identity carries over.
        record.update(command=prior.get("command") or record["command"],
                      init_from=record["init_from"] or prior.get("init_from"),
                      resumed=prior.get("resumed", 0) + 1)
        history = resumed_history[:-1] + [record]
    else:
        history = history + [record]

    provenance = {
        "corpus": record["corpus"],
        "tokenizer_fingerprint": fingerprint,
        "tokens_per_byte": dm.tokens_per_byte,      # val_bpb = val_loss * this / ln 2
        "history": history,
    }
    lit = LitJLM(model_cfg, train_cfg, init_state=init_state,
                 tokens_per_byte=dm.tokens_per_byte, provenance=provenance)
    del init_state
    trainer.fit(lit, datamodule=dm, ckpt_path=train_cfg.resume_from)
    return getattr(trainer.checkpoint_callback, "best_model_path", None)
