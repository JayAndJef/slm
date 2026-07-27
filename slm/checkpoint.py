"""Reading and writing model weights, in every shape this project produces.

Two formats exist, and :func:`load` reads both:

- **Lightning** (``last.ckpt``, ``…val_loss=….ckpt``) — what training writes. Carries
  optimizer moments, scheduler position and loop state, so ``trainer.fit(ckpt_path=…)``
  resumes a run exactly. Keys are prefixed by the LightningModule's attribute layout.
- **Compact** (``format: 2``) — what :func:`slm.export` writes for handing a model onward.
  Weights, provenance and the tokenizer itself, and nothing else.

:func:`load` sniffs which it was given from the keys present, the way
:func:`slm.tokenizer.load_tokenizer` sniffs its backend, and normalizes both to one
``(state_dict, ModelConfig, meta)`` triple. Consumers — :func:`slm.generate.load_model`,
``continue-train``, ``sft`` — therefore never branch on format, and the archived v1
checkpoint keeps loading.

Everything non-tensor travels as a **JSON string**, not a nested dict. ``Path`` and
``datetime`` both raise under ``torch.load(weights_only=True)``, so storing them directly
would force ``weights_only=False`` forever — precisely when the format becomes something you
hand to other people, which is the threat model that flag exists for.
"""
import json
import pickle
import warnings
from pathlib import Path

import torch

from slm.config import ModelConfig

FORMAT = 2
_LIGHTNING_PREFIXES = ("model._orig_mod.", "model.")     # longest first

# Meta keys :func:`load` derives from the file's own fields rather than from ``meta_json``.
# Exported here so a writer strips exactly these instead of maintaining its own list: a key
# that drifts off that list is stale provenance copied forward under a name that reads
# current — a v2 file claiming ``format: "lightning"``, say.
DERIVED_META = ("format", "step", "val", "tokenizer_json", "tokenizer_fingerprint")


def _strip(key: str) -> str:
    for p in _LIGHTNING_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def _lightning_val(ck: dict) -> float | None:
    """The monitored value at the step a Lightning checkpoint was written, if recorded.

    ``ModelCheckpoint`` keeps it in its callback state and nothing else in the file carries
    it, so without this every training checkpoint reports ``val n/a`` — including through
    ``export``, whose compact ``val`` field then never holds anything. Read defensively: a
    run with no such callback, or a future rename, just yields None.
    """
    for key, state in (ck.get("callbacks") or {}).items():
        if str(key).startswith("ModelCheckpoint") and state.get("current_score") is not None:
            return float(state["current_score"])
    return None


def _shared_cpu_state(model) -> dict[str, torch.Tensor]:
    """``state_dict`` on CPU, preserving storage sharing.

    The obvious ``{k: v.detach().cpu() for ...}`` silently breaks that sharing — and with it
    ``torch.save``'s storage dedup — which for a tied embedding/lm_head means writing the
    same ``vocab_size x hidden_dim`` matrix to disk twice (930 MB against 799 MB).
    """
    core = getattr(model, "_orig_mod", model)
    shared: dict[int, torch.Tensor] = {}
    state: dict[str, torch.Tensor] = {}
    for name, tensor in core.state_dict().items():
        key = tensor.data_ptr()
        if key not in shared:
            shared[key] = tensor.detach().cpu()
        state[name] = shared[key]
    return state


def save(path, model, *, model_cfg: ModelConfig, step: int, val: float | None = None,
         meta: dict | None = None, tokenizer_json: bytes | None = None,
         tokenizer_fingerprint: str | None = None) -> None:
    """Write the compact format: weights, architecture, provenance, and the tokenizer.

    Stores the *uncompiled* module (unwrapping ``torch.compile``'s ``_orig_mod``) so a
    compiled run stays loadable with ``strict=True``.

    Embedding the tokenizer costs 2.3 MB against ~800 MB of weights and removes the entire
    class of "trained on tokenizer A, generated with B" — a mismatch that passes every
    vocab-size check while making each id mean something else.
    """
    payload = {
        "format": FORMAT,
        "model": _shared_cpu_state(model),
        "config": model_cfg.to_dict(),
        "step": step,
        "val": val,
        # allow_nan=False so an inf/nan sneaking in raises here rather than producing a file
        # no non-Python JSON reader accepts.
        "meta_json": json.dumps(meta or {}, allow_nan=False, default=str),
    }
    if tokenizer_json is not None:
        payload["tokenizer_json"] = tokenizer_json
        payload["tokenizer_fingerprint"] = tokenizer_fingerprint

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load(path, map_location="cpu") -> tuple[dict, ModelConfig, dict]:
    """Return ``(state_dict, model_cfg, meta)`` from a checkpoint of any format.

    The architecture comes back as a :class:`~slm.config.ModelConfig` because the weights
    dictate it: a checkpoint can only be rebuilt at the dimensions it was trained at. It is
    kept out of ``meta`` because it is the one thing a caller must have *before* it can
    build a model, and the one typed thing here.

    ``weights_only=True``: the compact format is tensors plus JSON strings and bytes by
    construction. Lightning files carry pickled callback and loop state, so they need the
    escape hatch — the fallback is deliberate and narrow.

    Narrow means ``UnpicklingError`` only, which is what ``weights_only`` raises for a
    *disallowed global*. A truncated download or a corrupt archive raises ``RuntimeError``
    and now propagates, rather than being retried under an unpickler that would execute
    whatever the file contains — the exact threat model the compact format exists for.
    Anything that is not a ``.ckpt`` also warns before taking that path, since for the
    compact format needing it at all means the file is not what it claims to be.
    """
    try:
        ck = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        if Path(path).suffix != ".ckpt":
            warnings.warn(
                f"{path} does not load under weights_only=True "
                f"({str(e).splitlines()[0]}); falling back to the pickle unpickler, which "
                f"executes code from the file. Expected only for a Lightning .ckpt.",
                RuntimeWarning, stacklevel=2)
        ck = torch.load(path, map_location=map_location, weights_only=False)

    if "state_dict" in ck:                                  # Lightning
        state = {_strip(k): v for k, v in ck["state_dict"].items()}
        hp = dict(ck.get("hyper_parameters") or {})
        # hparams first: they carry the run's provenance (corpus, tokenizer), but must not
        # be able to overwrite what this file itself says its step and format are.
        meta = {**hp, "step": ck.get("global_step"), "val": _lightning_val(ck),
                "format": "lightning"}
        return state, ModelConfig.from_dict(hp), meta

    fmt = ck.get("format", 1)                               # v1 has no `format` key
    assert fmt <= FORMAT, (
        f"{path} is format {fmt} but this code understands up to {FORMAT} — ignoring a "
        f"newer writer's fields would load the weights and silently drop their provenance; "
        f"update slm/checkpoint.py")

    # Embedded provenance first, then the file's own fields on top: a `format` or `step`
    # carried inside meta_json from an earlier checkpoint describes *that* file, and letting
    # it win would have this one report a format it is not.
    derived = {"format": fmt, "step": ck.get("step"), "val": ck.get("val"),
               "tokenizer_json": ck.get("tokenizer_json"),
               "tokenizer_fingerprint": ck.get("tokenizer_fingerprint")}
    assert tuple(derived) == DERIVED_META, "DERIVED_META must name exactly these keys"
    meta = json.loads(ck.get("meta_json") or "{}")
    meta.update(derived)
    return ck["model"], ModelConfig.from_dict(ck["config"]), meta
