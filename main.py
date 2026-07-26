"""Command-line entrypoint for the slm package.

    uv run main.py train [--smoke] [--hidden-dim N ...]
    uv run main.py continue-train --init-from PATH --out-dir PATH
    uv run main.py prepare-data [--smoke]
    uv run main.py generate [--checkpoint PATH] [--prompt STR]
    uv run main.py train-tokenizer --out PATH

Each command builds config objects and calls exactly one package function.
"""
from pathlib import Path

import click
import torch

from slm import paths
from slm.config import ModelConfig, TrainConfig, default_configs
from slm.data import build_corpus, load_docs
from slm.generate import EOT, load_model
from slm.generate import stream as run_stream
from slm.tokenizer import HFTokenizer, SimpleTokenizer, load_tokenizer
from slm.train import train as run_train

# Flags shared by every command that runs a training loop. Declared once and applied by
# `training_options` so `train` and `continue-train` cannot drift apart — the two differ in
# what they *default* to, not in what they accept.
#
# Every default is None, meaning "not passed". That matters: a flag defaulting to its own
# value would silently clobber --smoke's sizing, or a continuation's anneal schedule.
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
]

# click param name -> TrainConfig field, where they cannot match (`compile` is a builtin).
_FIELD_ALIASES = {"compile_": "compile"}
_PATH_FIELDS = {"out_dir", "tokenizer_path", "init_from"}


def training_options(f):
    """Attach the shared training flags. Reversed so --help lists them as declared."""
    for option in reversed(_TRAINING_OPTIONS):
        f = option(f)
    return f


def _apply(cfg: ModelConfig | TrainConfig, **opts) -> None:
    """Set every explicitly-passed option on ``cfg``; ``None`` means 'not passed'."""
    for name, value in opts.items():
        if value is None:
            continue
        name = _FIELD_ALIASES.get(name, name)
        setattr(cfg, name, Path(value) if name in _PATH_FIELDS else value)


@click.group()
def cli():
    """From-scratch small language model: train, generate, or train a tokenizer."""


@cli.command()
@click.option("--smoke", is_flag=True, help="Tiny fast end-to-end sanity run.")
@click.option("--out-dir", type=click.Path(), default=None, help="Checkpoint output dir.")
@training_options
# ModelConfig overrides, likewise None-defaulted.
@click.option("--vocab-size", type=int, default=None)
@click.option("--hidden-dim", type=int, default=None)
@click.option("--num-heads", type=int, default=None)
@click.option("--n-layer", type=int, default=None)
@click.option("--block-size", type=int, default=None)
def train(smoke, out_dir, vocab_size, hidden_dim, num_heads, n_layer, block_size, **opts):
    """Train a model from scratch with Lightning (or a tiny --smoke run)."""
    model_cfg, train_cfg = default_configs(smoke=smoke)
    _apply(model_cfg, vocab_size=vocab_size, hidden_dim=hidden_dim,
           num_heads=num_heads, n_layer=n_layer, block_size=block_size)
    _apply(train_cfg, out_dir=out_dir, **opts)
    run_train(model_cfg, train_cfg)


@cli.command("continue-train")
@click.option("--init-from", type=click.Path(exists=True), required=True,
              help="Compact checkpoint whose weights start this run.")
@click.option("--out-dir", type=click.Path(), required=True,
              help="Checkpoint output dir. Required, and should differ from --init-from's: "
                   "the source model is the thing this run is trying to beat.")
@training_options
def continue_train(init_from, out_dir, **opts):
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
    _, train_cfg = default_configs()
    train_cfg.init_from, train_cfg.out_dir = Path(init_from), Path(out_dir)
    train_cfg.max_steps, train_cfg.warmup_steps = 4_000, 0
    train_cfg.lr, train_cfg.min_lr, train_cfg.decay_frac = 1e-4, 0.0, 1.0
    _apply(train_cfg, **opts)
    run_train(None, train_cfg)          # architecture comes from the checkpoint


@cli.command("prepare-data")
@click.option("--smoke", is_flag=True, help="Prepare the tiny smoke corpus.")
def prepare_data(smoke):
    """Encode + cache the train/val corpora once (no GPU), so DDP runs just mmap them."""
    _, cfg = default_configs(smoke=smoke)
    train_docs, val_docs = load_docs(cfg)
    build_corpus(train_docs, cfg.train_path, cfg.tokenizer_path,
                 sep=cfg.sep, n_workers=cfg.n_workers, logger=click.echo)
    build_corpus(val_docs, cfg.val_path, cfg.tokenizer_path,
                 sep=cfg.sep, n_workers=cfg.n_workers, logger=click.echo)


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
@click.option("--tokenizer", "tokenizer_path", type=click.Path(exists=True),
              default=str(paths.TOKENIZER_PATH), show_default=True)
def generate(checkpoint, prompt, max_new_tokens, temperature, top_p, num_samples, device,
             tokenizer_path):
    """Generate text from a trained checkpoint."""
    dev = torch.device(device)
    model, cfg, meta = load_model(checkpoint, dev)
    tok = load_tokenizer(tokenizer_path)
    assert cfg.vocab_size == tok.n_vocab, (
        f"checkpoint has vocab {cfg.vocab_size} but {tokenizer_path} has {tok.n_vocab} — "
        f"pass --tokenizer for the one this model was trained with")
    n_params = sum(p.numel() for p in model.parameters())
    val_str = "n/a" if meta["val"] is None else f"{meta['val']:.3f}"
    click.echo(f"loaded {checkpoint}: step {meta['step']}, val {val_str}, "
               f"{n_params/1e6:.1f}M params\n")
    for i in range(num_samples):
        click.echo(f"--- sample {i + 1} (temp {temperature}, top_p {top_p}) ---")
        for chunk in run_stream(model, tok, prompt=prompt, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_p=top_p,
                                block_size=cfg.block_size, device=dev):
            click.echo(chunk, nl=False)
            click.get_text_stream("stdout").flush()
        click.echo("\n")


@cli.command("train-tokenizer")
@click.option("--out", "out_path", type=click.Path(), required=True,
              help="Where to save the tokenizer JSON (required — never defaults to the kept one).")
@click.option("--vocab-size", type=int, default=32_000, show_default=True)
@click.option("--n-docs", type=int, default=10000, show_default=True,
              help="Random sample of documents to train on.")
@click.option("--backend", type=click.Choice(["hf", "simple"]), default="hf",
              show_default=True,
              help="hf = rust-backed tokenizers; simple = the from-scratch implementation.")
def train_tokenizer(out_path, vocab_size, n_docs, backend):
    """Train a new BPE tokenizer on a random sample of the corpus."""
    cfg = TrainConfig(n_train_docs=n_docs)
    train_docs, _ = load_docs(cfg)          # a Dataset; rows are dicts with a "text" key
    click.echo(f"training {backend} tokenizer (vocab {vocab_size}) on "
               f"{len(train_docs)} docs ...")
    if backend == "hf":
        tok = HFTokenizer.train((d["text"] for d in train_docs),
                                vocab_size=vocab_size, special=cfg.sep.strip())
    else:
        tok = SimpleTokenizer(vocab_size=vocab_size)
        tok.train_tokenizer(cfg.sep.join(train_docs["text"]))
    tok.save(out_path)
    click.echo(f"saved -> {out_path} ({tok.n_vocab} tokens, "
               f"sep_id {tok.sep_id(cfg.sep)})")


if __name__ == "__main__":
    cli()
