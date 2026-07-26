"""The compact checkpoint format: ``{"model", "step", "val", "config"}``.

Every consumer goes through here — :class:`slm.lit.CompactCheckpoint` writes,
:func:`slm.generate.load_model` reads for inference, and :func:`slm.train.train` reads to
continue a run — so the on-disk shape is described in exactly one place rather than
re-derived at each call site.

Deliberately *not* a Lightning checkpoint. It carries the weights plus the architecture
needed to rebuild them, and nothing else: no optimizer moments, no scheduler state, no
dataloader position. A continuation is therefore a fresh run that happens to start from
trained weights, never a resumption of the old one — see :func:`slm.train.train`.
"""
from pathlib import Path

import torch

from slm.config import ModelConfig


def save(path, model, *, step: int, val: float, model_cfg: ModelConfig) -> None:
    """Write ``model``'s weights plus the metadata needed to rebuild it.

    Stores the *uncompiled* module (unwrapping ``torch.compile``'s ``_orig_mod``) so a
    compiled run stays loadable with ``strict=True``.

    Parameters that share storage are mapped to a single shared CPU tensor. The obvious
    ``{k: v.detach().cpu() for ...}`` silently breaks that sharing — and with it
    ``torch.save``'s storage dedup — which for a tied embedding/lm_head means writing the
    same ``vocab_size x hidden_dim`` matrix to disk twice.
    """
    core = getattr(model, "_orig_mod", model)
    shared: dict[int, torch.Tensor] = {}
    state: dict[str, torch.Tensor] = {}
    for name, tensor in core.state_dict().items():
        key = tensor.data_ptr()
        if key not in shared:
            shared[key] = tensor.detach().cpu()
        state[name] = shared[key]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "step": step, "val": val,
                "config": model_cfg.to_dict()}, path)


def load(path, map_location="cpu") -> tuple[dict, ModelConfig, dict]:
    """Return ``(state_dict, model_cfg, meta)`` from a compact checkpoint.

    The architecture comes back as a :class:`~slm.config.ModelConfig` because the weights
    dictate it: a checkpoint can only be rebuilt at the dimensions it was trained at.
    ``meta`` carries ``step`` and ``val`` for display and for seeding a continuation's
    best-so-far.

    ``weights_only=False`` is safe here — these are our own checkpoints, holding tensors
    plus a small plain-dict config.
    """
    ck = torch.load(path, map_location=map_location, weights_only=False)
    meta = {"step": ck.get("step"), "val": ck.get("val")}
    return ck["model"], ModelConfig.from_dict(ck["config"]), meta
