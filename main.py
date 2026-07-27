"""Command-line entrypoint for the slm package.

    uv run main.py train [--smoke] [--hidden-dim N ...]
    uv run main.py continue-train --init-from PATH --out-dir PATH
    uv run main.py prepare-data [--smoke]
    uv run main.py generate [--checkpoint PATH] [--prompt STR]
    uv run main.py train-tokenizer --out PATH

Each command builds config objects and calls exactly one package function.
"""
import json
import math
from pathlib import Path

import click
import numpy as np
import torch

from slm import chat, checkpoint, corpus, paths
from slm.config import (CORPUS_PRESETS, SEP, TrainConfig, corpus_preset,
                        default_configs)
from slm.generate import EOT, is_chat_checkpoint, load_model, tokenizer_for
from slm.model import JLM
from slm.generate import stream as run_stream
from slm.tokenizer import HFTokenizer, SimpleTokenizer, load_tokenizer
from slm.train import n_devices
from slm.train import train as run_train

# Flags shared by every command that runs a training loop. Declared once and applied by
# `training_options` so `train` and `continue-train` cannot drift apart — the two differ in
# what they *default* to, not in what they accept.
#
# Every default is None, meaning "not passed". That matters: a flag defaulting to its own
# value would silently clobber --smoke's sizing, or a continuation's anneal schedule.
#
# Every flag here is a TrainConfig field, so each command names its target explicitly:
# `_apply(train_cfg, **opts)`. Misrouting is impossible by construction. A flag that ever
# targets SourceSpec or RenderSpec has to declare its owner next to the flag rather than be
# searched for across config objects, or "this reached the wrong object" stops being
# impossible and becomes a runtime assert.
_TRAINING_OPTIONS = [
    click.option("--devices", default=None,
                 help="GPU count, or comma-list of PyTorch indices (e.g. 2,3,4,5)."),
    click.option("--accelerator", default=None, help='"auto" | "cuda" | "cpu".'),
    click.option("--num-nodes", type=int, default=None),
    click.option("--precision", default=None, help='e.g. "bf16-mixed", "32-true".'),
    click.option("--dataloader-workers", type=int, default=None),
    click.option("--wandb", is_flag=True, default=None, help="Log to Weights & Biases."),
    click.option("--wandb-project", default=None),
    click.option("--batch-size", type=int, default=None),
    click.option("--max-steps", type=int, default=None),
    click.option("--lr", type=float, default=None, help="Peak learning rate."),
    click.option("--min-lr", type=float, default=None, help="Floor the decay anneals to."),
    click.option("--warmup-steps", type=int, default=None),
    click.option("--decay-frac", type=float, default=None,
                 help="Trailing fraction of max_steps spent decaying (1.0 = pure anneal)."),
    click.option("--sampler-seed", type=int, default=None,
                 help="Reshuffles the window order without renaming the corpus cache."),
    click.option("--eval-every", type=int, default=None, help="Steps between validations."),
    click.option("--eval-iters", type=int, default=None,
                 help="Val batches per rank (18 is one clean pass at batch 48 x 4 ranks)."),
    click.option("--compile/--no-compile", "compile_", default=None,
                 help="torch.compile the model (default on for real runs)."),
    click.option("--doc-mask/--no-doc-mask", "doc_mask", default=None,
                 help="Stop attention crossing document boundaries (default on)."),
    click.option("--tokenizer", "tokenizer_path", type=click.Path(exists=True), default=None),
    click.option("--resume", "resume_from", type=click.Path(exists=True), default=None,
                 help="Lightning .ckpt to continue: optimizer momentum, LR schedule "
                      "position and loop counters restored. The window stream is NOT — it "
                      "restarts at the head of its permutation and re-serves windows the "
                      "killed run already trained on; pass --sampler-seed to draw fresh "
                      "ones. Distinct from --init-from, which takes weights only."),
]

# click param name -> TrainConfig field, where they cannot match (`compile` is a builtin).
_FIELD_ALIASES = {"compile_": "compile"}
_PATH_FIELDS = {"out_dir", "tokenizer_path", "init_from", "data_dir", "resume_from"}


def training_options(f):
    """Attach the shared training flags. Reversed so --help lists them as declared."""
    for option in reversed(_TRAINING_OPTIONS):
        f = option(f)
    return f


def _apply(cfg, **opts) -> None:
    """Set every explicitly-passed option on ``cfg``; ``None`` means 'not passed'."""
    for name, value in opts.items():
        if value is None:
            continue
        name = _FIELD_ALIASES.get(name, name)
        assert hasattr(cfg, name), f"{type(cfg).__name__} has no field {name!r}"
        setattr(cfg, name, Path(value) if name in _PATH_FIELDS else value)


@click.group()
def cli():
    """From-scratch small language model: train, generate, or train a tokenizer."""


def _require_corpus(spec, train_cfg, *, smoke: bool):
    """Locate an already-built corpus pair, or explain how to build it.

    Training entrypoints never build. Lightning's DDP strategy re-executes ``sys.argv`` for
    every rank, so building here would race four processes on a multi-GB write; and a stray
    flag that moved the hash would otherwise trigger a silent 15-minute re-encode at the head
    of a multi-hour run instead of an error.
    """
    train, val = corpus.locate(spec, data_dir=train_cfg.data_dir,
                               tokenizer_path=train_cfg.tokenizer_path)
    missing = [c.name for c in (train, val) if not c.exists()]
    if missing:
        raise click.UsageError(
            f"corpus not built: {', '.join(missing)} — run: uv run main.py prepare-data "
            f"--corpus {spec.name}{' --smoke' if smoke else ''}")
    return train, val


@cli.command()
@click.option("--smoke", is_flag=True, help="Tiny fast end-to-end sanity run.")
@click.option("--corpus", "corpus_name", default=None,
              help=f"Corpus preset: {', '.join(sorted(CORPUS_PRESETS))}.")
@click.option("--out-dir", type=click.Path(), default=None,
              help="Checkpoint output dir. Required unless --smoke.")
@training_options
# ModelConfig overrides, likewise None-defaulted.
@click.option("--vocab-size", type=int, default=None)
@click.option("--hidden-dim", type=int, default=None)
@click.option("--num-heads", type=int, default=None)
@click.option("--n-layer", type=int, default=None)
@click.option("--block-size", type=int, default=None)
def train(smoke, corpus_name, out_dir, vocab_size, hidden_dim, num_heads, n_layer,
          block_size, **opts):
    """Train a model from scratch with Lightning (or a tiny --smoke run)."""
    if out_dir is None and not smoke:
        raise click.UsageError(
            "--out-dir is required (it should be a fresh directory, not checkpoints/) — "
            "a from-scratch run starts best-so-far at infinity and overwrites the "
            "destination's last.ckpt at its first validation")
    model_cfg, train_cfg, spec = default_configs(smoke=smoke)
    if corpus_name:
        spec = corpus_preset(corpus_name)
    _apply(model_cfg, vocab_size=vocab_size, hidden_dim=hidden_dim,
           num_heads=num_heads, n_layer=n_layer, block_size=block_size)
    _apply(train_cfg, out_dir=out_dir, **opts)
    run_train(model_cfg, train_cfg, spec, *_require_corpus(spec, train_cfg, smoke=smoke))


@cli.command("continue-train")
@click.option("--init-from", type=click.Path(exists=True), required=True,
              help="Compact checkpoint whose weights start this run.")
@click.option("--out-dir", type=click.Path(), required=True,
              help="Checkpoint output dir. Required, and should differ from --init-from's: "
                   "the source model is the thing this run is trying to beat.")
@click.option("--corpus", "corpus_name", default=None,
              help=f"Corpus preset: {', '.join(sorted(CORPUS_PRESETS))}.")
@training_options
def continue_train(init_from, out_dir, corpus_name, **opts):
    """Continue training from a checkpoint's weights under a fresh LR schedule.

    Weights only — the compact format carries no optimizer moments, scheduler position or
    dataloader offset, so momentum rebuilds over the first few hundred steps. The
    architecture is read from the checkpoint; model-dimension flags would be meaningless
    here and are deliberately absent.

    Defaults describe a **pure anneal**: no warmup, no stable phase, decay straight from
    ``--lr`` to ``--min-lr`` across ``--max-steps``. That is what a fully-decayed model
    wants. To instead extend training, re-warm by passing a higher --lr with --warmup-steps
    and a --decay-frac below 1.0, and expect val to regress before it improves.
    """
    _, train_cfg, spec = default_configs()
    if corpus_name:
        spec = corpus_preset(corpus_name)
    train_cfg.init_from, train_cfg.out_dir = Path(init_from), Path(out_dir)
    train_cfg.max_steps, train_cfg.warmup_steps = 4_000, 0
    train_cfg.lr, train_cfg.min_lr, train_cfg.decay_frac = 1e-4, 0.0, 1.0
    _apply(train_cfg, **opts)
    # architecture comes from the checkpoint
    run_train(None, train_cfg, spec, *_require_corpus(spec, train_cfg, smoke=False))


@cli.command()
@click.option("--init-from", type=click.Path(exists=True), required=True,
              help="Checkpoint whose weights start this run (any format).")
@click.option("--out-dir", type=click.Path(), required=True)
@click.option("--corpus", "corpus_name", default="smoltalk2", show_default=True)
@click.option("--epochs", type=float, default=3.0, show_default=True,
              help="Passes over the SFT corpus; sets max_steps from its token count.")
@training_options
def sft(init_from, out_dir, corpus_name, epochs, **opts):
    """Supervised fine-tune a pretrained model on an instruction corpus.

    A separate command rather than ``continue-train --corpus``, on the same reasoning that
    separates ``train`` from ``continue-train``: the *defaults* differ, while
    ``training_options`` keeps what they accept from drifting.

    The LR is ~30x below pretraining's peak. That heuristic transfers here only because
    ``configure_optimizers`` gives Muon ``adjust_lr_fn="match_rms_adamw"``, which rescales
    its update to the size AdamW would take; raw Muon's natural scale is ~10x higher and the
    same number would land far too hot. ``warmup_steps`` is 50 for a concrete reason too:
    no checkpoint format carries optimizer moments, so Muon's momentum buffer rebuilds from
    zero, and at ``momentum=0.95`` its time constant is 1/(1-0.95) = 20 steps.
    """
    _, train_cfg, _ = default_configs()
    spec = corpus_preset(corpus_name)
    train_cfg.init_from, train_cfg.out_dir = Path(init_from), Path(out_dir)
    train_cfg.lr, train_cfg.min_lr = 3e-5, 0.0
    train_cfg.warmup_steps, train_cfg.decay_frac = 50, 1.0
    train_cfg.eval_every = 200
    train_cfg.weight_decay = 0.01       # 0.1 over ~9k steps is pointless shrinkage here
    _apply(train_cfg, **opts)

    train_corpus, val_corpus = _require_corpus(spec, train_cfg, smoke=False)
    if opts.get("max_steps") is None:
        # There is otherwise no way to say "3 epochs" — only --max-steps, which is a number
        # you compute by hand and get wrong. Tokens/step needs the window length, and a
        # packed corpus is the only one that records it (the model's block_size, asserted
        # equal at setup, is not available here: sft reads its architecture from the
        # checkpoint inside train()).
        block = train_corpus.meta["pack_block"]
        if not isinstance(block, int):
            raise click.UsageError(
                f"{train_corpus.name} is not packed, so it records no window length and "
                f"tokens/step is unknown — pass --max-steps explicitly")
        per_step = train_cfg.batch_size * block * n_devices(train_cfg.devices)
        train_cfg.max_steps = max(1, round(train_corpus.meta["n_tokens"] * epochs / per_step))
        click.echo(f"{epochs} epochs over {train_corpus.meta['n_tokens']/1e6:.0f}M tokens "
                   f"= {train_cfg.max_steps} steps at {per_step:,} tokens/step")
    run_train(None, train_cfg, spec, train_corpus, val_corpus)


@cli.command("sft-eval")
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--tokenizer", "tokenizer_path", type=click.Path(exists=True), default=None)
@click.option("--chat/--no-chat", "chat_mode", default=True,
              help="--no-chat scores the base model on the same prompts, for a delta.")
@click.option("--prompts", "prompts_path", type=click.Path(exists=True),
              default=str(paths.DATA_DIR / "eval-prompts.json"), show_default=True)
@click.option("--max-new-tokens", type=int, default=256, show_default=True)
@click.option("--arc/--no-arc", "run_arc", default=True,
              help="Score ARC-Easy for capability regression (the Phase 7 gate).")
@click.option("--arc-items", type=int, default=200, show_default=True)
@click.option("--device", default="cuda:6", show_default=True)
def sft_eval(checkpoint, tokenizer_path, chat_mode, prompts_path, max_new_tokens,
             run_arc, arc_items, device):
    """Measure what SFT changed: does it answer, does it stop, and did it forget.

    Run it on the *base* model first with --no-chat. That is the reference the SFT numbers
    are read against, and it must exist before the fine-tune, not after.
    """
    from slm import evaluate

    dev = torch.device(device)
    model, cfg, meta = load_model(checkpoint, dev)
    tok = tokenizer_for(meta, cfg, tokenizer_path)
    prompts = json.loads(Path(prompts_path).read_text())["prompts"]

    fmt = evaluate.format_compliance(model, tok, prompts, block_size=cfg.block_size,
                                     device=dev, max_new_tokens=max_new_tokens,
                                     chat_mode=chat_mode)
    click.echo(f"\n=== format ({'chat' if chat_mode else 'base'}, n={fmt['n']}) ===")
    for k in ("stopped", "hit_cap", "stray_role", "relapsed_to_eot", "repetition"):
        click.echo(f"  {k:<18} {fmt[k]:.1%}")
    click.echo(f"  {'mean_len':<18} {fmt['mean_len']:.0f} tokens")
    for s in fmt["samples"]:
        click.echo(f"  | {s[:160]!r}")

    if run_arc:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation",
                          cache_dir=paths.HF_CACHE_DIR).select(range(arc_items))
        items = [{"question": r["question"], "options": r["choices"]["text"],
                  "answer": r["choices"]["label"].index(r["answerKey"])}
                 for r in ds if r["answerKey"] in r["choices"]["label"]]
        arc = evaluate.multiple_choice(model, tok, items, block_size=cfg.block_size,
                                       device=dev)
        click.echo(f"\n=== ARC-Easy (n={arc['n']}) ===")
        click.echo(f"  accuracy           {arc['accuracy']:.1%}   "
                   f"(chance {1/4:.1%}; compare against the base model, not chance)")


@cli.command("list-splits")
@click.argument("dataset")
@click.option("--config", "config_name", default=None, help="Restrict to one config/subset.")
def list_splits(dataset, config_name):
    """Print a dataset's configs, splits and row counts, for pasting into a corpus preset.

    Corpus specs name their (config, split) parts explicitly rather than matching a pattern,
    because a pattern is stable in the file while the set it matches is not — an upstream
    split appearing would change the corpus with no change to its hash. This command is how
    that explicit list gets written; it is never consulted at build time.
    """
    from datasets import get_dataset_config_names, load_dataset_builder

    names = [config_name] if config_name else get_dataset_config_names(dataset)
    for name in names:
        info = load_dataset_builder(dataset, name).info
        click.echo(f"\n=== {dataset} :: {name} "
                   f"({', '.join(info.features) if info.features else 'no features'}) ===")
        for split, v in sorted((info.splits or {}).items(), key=lambda kv: -kv[1].num_examples):
            click.echo(f"  {v.num_examples:>10,}  {split}")


def _prune(cfg, *, delete: bool) -> None:
    """Report cached corpora that no preset resolves to, and optionally remove them.

    The hash in a corpus filename is what makes drift visible — change the tokenizer, the
    source or the renderer and you get a *different* file rather than a silently different
    run. The cost is that nothing ever collects the superseded one, so `data/` accumulates
    every intermediate state a project passes through. This is the collector.

    Reachability is computed from the presets and the *current* tokenizer, so a corpus can
    become unreachable without being wrong — a file with no `.meta` simply predates the
    commit-record format and cannot be read at all.
    """
    live = set()
    for name in CORPUS_PRESETS:
        for c in corpus.locate(corpus_preset(name), data_dir=cfg.data_dir,
                               tokenizer_path=cfg.tokenizer_path):
            live.add(c.name)

    groups: dict[str, list[Path]] = {}
    for p in sorted(Path(cfg.data_dir).glob("*")):
        if p.suffix not in (".npy", ".meta"):
            continue
        stem = p.name.split(".npy")[0].split(".meta")[0]
        stem = stem.replace(".target", "").replace(".start", "")
        if stem not in live:
            groups.setdefault(stem, []).append(p)

    if not groups:
        click.echo(f"no orphaned corpora in {cfg.data_dir}")
        return
    total = 0
    for stem, paths in sorted(groups.items()):
        size = sum(p.stat().st_size for p in paths)
        total += size
        note = "" if any(p.suffix == ".meta" for p in paths) else "  (no .meta — unreadable)"
        click.echo(f"  {size/1e9:7.2f} GB  {stem}{note}")
    click.echo(f"  {total/1e9:7.2f} GB total across {len(groups)} corpora")

    if not delete:
        click.echo(f"\nreachable from the current presets and tokenizer "
                   f"({cfg.tokenizer_path.name}):")
        for n in sorted(live):
            click.echo(f"    {n}")
        click.echo("\nnothing deleted — re-run with --delete")
        return
    for paths in groups.values():
        for p in paths:
            p.unlink()
    click.echo(f"deleted {total/1e9:.2f} GB")


@cli.command("prepare-data")
@click.option("--smoke", is_flag=True, help="Prepare the tiny smoke corpus.")
@click.option("--corpus", "corpus_name", default=None,
              help=f"Corpus preset: {', '.join(sorted(CORPUS_PRESETS))}.")
@click.option("--tokenizer", "tokenizer_path", type=click.Path(exists=True), default=None,
              help="Must match train's, or the hash differs and nothing finds this corpus.")
@click.option("--data-dir", type=click.Path(), default=None)
@click.option("--workers", type=int, default=corpus.N_WORKERS, show_default=True,
              help="Encode processes. Outside the hash, but recorded in meta: it permutes "
                   "shard order, and under bfd packing it changes which examples share a bin.")
@click.option("--dry-run", is_flag=True,
              help="Print what would be built (paths, hash, existing meta) and exit.")
@click.option("--prune", is_flag=True,
              help="List cached corpora no preset resolves to. Add --delete to remove them.")
@click.option("--delete", is_flag=True, help="With --prune, actually delete.")
def prepare_data(smoke, corpus_name, tokenizer_path, data_dir, workers, dry_run,
                 prune, delete):
    """Encode + cache the train/val corpora once (no GPU), so training runs just mmap them."""
    _, cfg, spec = default_configs(smoke=smoke)
    if corpus_name:
        spec = corpus_preset(corpus_name)
    _apply(cfg, tokenizer_path=tokenizer_path, data_dir=data_dir)

    if prune:
        _prune(cfg, delete=delete)
        return

    if dry_run:
        tok = load_tokenizer(cfg.tokenizer_path)
        click.echo(f"corpus     {spec.name}  ({spec.render.kind}, pack={spec.render.pack})")
        click.echo(f"tokenizer  {cfg.tokenizer_path}  fingerprint {tok.fingerprint}  "
                   f"n_vocab {tok.n_vocab}")
        click.echo(f"parts      {len(spec.source.parts)}  seed {spec.source.seed}  "
                   f"n_val_docs {spec.source.n_val_docs}")
        for c in corpus.locate(spec, data_dir=cfg.data_dir,
                               tokenizer_path=cfg.tokenizer_path):
            mark = "EXISTS" if c.exists() else "missing"
            click.echo(f"  [{mark}] {c.token_path}")
            if c.exists():
                m = c.meta
                click.echo(f"           {m['n_tokens']/1e6:.1f}M tokens, "
                           f"sep_frac {m['sep_frac']:.6f}, "
                           f"tokens_per_byte {m['tokens_per_byte']}")
                sample = np.load(c.token_path, mmap_mode="r")[:512].tolist()
                click.echo(f"           {load_tokenizer(cfg.tokenizer_path).decode(sample)[:300]!r}")
        return

    corpus.build(spec, tokenizer_path=cfg.tokenizer_path, data_dir=cfg.data_dir,
                 hf_cache_dir=cfg.hf_cache_dir, n_workers=workers, logger=click.echo)


@cli.command()
@click.option("--checkpoint", type=click.Path(exists=True),
              default=str(paths.CKPT_DIR / "best.pt"), show_default=True)
@click.option("--prompt", default=EOT, help="Text to continue.")
@click.option("--max-new-tokens", type=int, default=250)
@click.option("--temperature", type=float, default=0.8)
@click.option("--top-p", type=float, default=0.9, show_default=True,
              help="Nucleus sampling: keep the top tokens summing to this mass. "
                   "1.0 disables it.")
@click.option("--num-samples", type=int, default=2)
@click.option("--device", default="cuda:3")
@click.option("--tokenizer", "tokenizer_path", type=click.Path(exists=True), default=None,
              help="Defaults to the tokenizer embedded in the checkpoint, else "
                   f"{paths.TOKENIZER_PATH}.")
@click.option("--chat/--no-chat", "chat_mode", default=None,
              help="Wrap --prompt as a user turn and stop on <|im_end|>. Inferred from the "
                   "checkpoint's recorded render kind when not passed.")
def generate(checkpoint, prompt, max_new_tokens, temperature, top_p, num_samples, device,
             tokenizer_path, chat_mode):
    """Generate text from a trained checkpoint."""
    dev = torch.device(device)
    model, cfg, meta = load_model(checkpoint, dev)

    tok = tokenizer_for(meta, cfg, tokenizer_path)
    if chat_mode is None:
        chat_mode = is_chat_checkpoint(meta)
    stop_id = tok.special_id(chat.IM_END) if chat_mode else None
    if chat_mode:
        prompt = chat.render_prompt([{"role": "user", "content": prompt}])

    n_params = sum(p.numel() for p in model.parameters())
    val_str = "n/a" if meta["val"] is None else f"{meta['val']:.3f}"
    click.echo(f"loaded {checkpoint}: step {meta['step']}, val {val_str}, "
               f"{n_params/1e6:.1f}M params{', chat' if chat_mode else ''}\n")
    for i in range(num_samples):
        click.echo(f"--- sample {i + 1} (temp {temperature}, top_p {top_p}) ---")
        for chunk in run_stream(model, tok, prompt=prompt, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_p=top_p, stop_id=stop_id,
                                block_size=cfg.block_size, device=dev):
            click.echo(chunk, nl=False)
            click.get_text_stream("stdout").flush()
        click.echo("\n")


@cli.command()
@click.argument("ckpt", type=click.Path(exists=True))
@click.option("--out", "out_path", type=click.Path(), required=True)
@click.option("--tokenizer", "tokenizer_path", type=click.Path(exists=True),
              default=str(paths.TOKENIZER_PATH), show_default=True)
def export(ckpt, out_path, tokenizer_path):
    """Convert any checkpoint into the compact, self-contained format.

    Training writes Lightning checkpoints, whose ``state_dict`` keys carry a prefix that
    depends on run configuration (``model.`` vs ``model._orig_mod.`` under compile) and
    which need Lightning conventions to interpret. This flattens that and embeds the
    tokenizer, producing one file that loads with ``weights_only=True`` and cannot be
    paired with the wrong vocabulary.
    """
    state, model_cfg, meta = checkpoint.load(ckpt)
    tok = load_tokenizer(tokenizer_path)
    assert model_cfg.vocab_size == tok.n_vocab, (
        f"checkpoint vocab {model_cfg.vocab_size} != {tokenizer_path} ({tok.n_vocab}) — "
        f"pass --tokenizer for the one this model was trained with")
    recorded = meta.get("tokenizer_fingerprint")
    if recorded and recorded != tok.fingerprint:
        raise click.UsageError(
            f"{ckpt} was trained with tokenizer {recorded} but --tokenizer is "
            f"{tok.fingerprint} — embedding it would make the file permanently wrong")

    model = JLM.from_config(model_cfg)
    model.load_state_dict(state)
    # checkpoint.DERIVED_META, not a copy: these are the keys `load` fills in from the file's
    # own fields, so carrying them over would describe the source file, not this one.
    keep = {k: v for k, v in meta.items() if k not in checkpoint.DERIVED_META}
    checkpoint.save(out_path, model, model_cfg=model_cfg, step=meta.get("step") or 0,
                    val=meta.get("val"), meta=keep,
                    tokenizer_json=Path(tokenizer_path).read_bytes(),
                    tokenizer_fingerprint=tok.fingerprint)
    click.echo(f"exported -> {out_path} "
               f"({Path(out_path).stat().st_size/1e6:.1f} MB, tokenizer {tok.fingerprint})")


@cli.command()
@click.argument("ckpt", type=click.Path(exists=True))
def inspect(ckpt):
    """Print what a checkpoint is: architecture, provenance, and where it came from."""
    _, model_cfg, meta = checkpoint.load(ckpt)
    val, tpb = meta.get("val"), meta.get("tokens_per_byte")
    # val_bpb is derived, never stored: it is val_loss rescaled by the corpus's tokens/byte,
    # which is the only form comparable across vocab sizes.
    bpb = "unknown" if val is None or not tpb else f"{val * tpb / math.log(2):.5f}"
    click.echo(f"{ckpt}")
    click.echo(f"  format      {meta.get('format')}")
    click.echo(f"  arch        {model_cfg}")
    click.echo(f"  step        {meta.get('step')}")
    click.echo(f"  val         {'n/a' if val is None else f'{val:.4f}'}   val_bpb {bpb}")
    click.echo(f"  tokenizer   {meta.get('tokenizer_fingerprint') or 'not recorded'}"
               f"  {'(embedded)' if meta.get('tokenizer_json') else ''}")
    corpus_meta = meta.get("corpus") or {}
    click.echo(f"  corpus      {corpus_meta.get('name', 'unknown')} "
               f"{corpus_meta.get('hash', '')}")
    for node in meta.get("lineage") or []:
        click.echo(f"    <- {node.get('run_id', '?')} step {node.get('step')} "
                   f"val {node.get('val')}")


@cli.command("train-tokenizer")
@click.option("--out", "out_path", type=click.Path(), required=True,
              help="Where to save the tokenizer JSON (required — never defaults to the kept one).")
@click.option("--vocab-size", type=int, default=32_000, show_default=True)
@click.option("--n-docs", type=int, default=10000, show_default=True,
              help="Random sample of documents to train on.")
@click.option("--backend", type=click.Choice(["hf", "simple"]), default="hf",
              show_default=True,
              help="hf = rust-backed tokenizers; simple = the from-scratch implementation.")
@click.option("--reserved", type=int, default=None,
              help=f"Size of the declared special block (default {chat.N_RESERVED}). hf only.")
def train_tokenizer(out_path, vocab_size, n_docs, backend, reserved):
    """Train a new BPE tokenizer on a random sample of the corpus, excluding the val split."""
    if reserved is not None and backend == "simple":
        raise click.UsageError(
            "--reserved applies to --backend hf only — SimpleTokenizer declares no special "
            "tokens, it can only learn markers frequent enough to become merges")
    n_reserved = chat.N_RESERVED if reserved is None else reserved

    cfg = TrainConfig()
    src = corpus_preset("cosmopedia").source
    src.n_train_docs = n_docs
    train_docs, _ = corpus.load_docs(src, hf_cache_dir=cfg.hf_cache_dir)
    click.echo(f"training {backend} tokenizer (vocab {vocab_size}, reserved {n_reserved}) on "
               f"{len(train_docs)} docs from {src.dataset_name} "
               f"({len(src.parts)} parts, seed {src.seed}, n_val_docs {src.n_val_docs}) ...")
    if backend == "hf":
        tok = HFTokenizer.train((d["text"] for d in train_docs),
                                vocab_size=vocab_size, specials=chat.specials(n_reserved))
    else:
        tok = SimpleTokenizer(vocab_size=vocab_size)
        tok.train_tokenizer(SEP.join(train_docs["text"]))
    tok.save(out_path)

    click.echo(f"saved -> {out_path}")
    click.echo(f"  n_vocab     {tok.n_vocab}")
    click.echo(f"  fingerprint {tok.fingerprint}")
    click.echo(f"  sep_id      {tok.sep_id(SEP)}")
    if backend == "hf":
        ids = [tok.special_id(s) for s in chat.specials(n_reserved)]
        n_merges = len(json.loads(Path(out_path).read_text())["model"]["merges"])
        click.echo(f"  specials    ids {ids[0]}..{ids[-1]} "
                   f"({'contiguous from 0' if ids == list(range(len(ids))) else 'NOT CONTIGUOUS'})")
        click.echo(f"  merges      {n_merges} (expect {vocab_size - 256 - n_reserved})")


if __name__ == "__main__":
    cli()
