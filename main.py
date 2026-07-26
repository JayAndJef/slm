"""Command-line entrypoint for the slm package.

    uv run main.py train [--smoke] [--hidden-dim N ...]
    uv run main.py generate [--checkpoint PATH] [--prompt STR]
    uv run main.py train-tokenizer --out PATH

Each command builds config objects and calls exactly one package function.
"""
from pathlib import Path

import click
import torch

from slm import paths
from slm.config import TrainConfig, default_configs
from slm.data import build_corpus, load_docs
from slm.generate import generate as run_generate
from slm.generate import load_model
from slm.tokenizer import HFTokenizer, SimpleTokenizer, load_tokenizer
from slm.train import train as run_train


@click.group()
def cli():
    """From-scratch small language model: train, generate, or train a tokenizer."""


@cli.command()
@click.option("--smoke", is_flag=True, help="Tiny fast end-to-end sanity run.")
@click.option("--devices", default=None,
              help="GPU count or comma-list of PyTorch indices. NOTE: PyTorch orders "
                   "devices FASTEST_FIRST, so 0,1 are the Blackwells and 2-7 the A6000s "
                   "(not nvidia-smi order).")
@click.option("--accelerator", default=None, help='"auto" | "cuda" | "cpu".')
@click.option("--num-nodes", type=int, default=None)
@click.option("--precision", default=None, help='e.g. "bf16-mixed", "32-true".')
@click.option("--dataloader-workers", type=int, default=None)
@click.option("--wandb", is_flag=True, help="Log to Weights & Biases.")
@click.option("--wandb-project", default=None)
@click.option("--max-steps", type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--lr", type=float, default=None)
@click.option("--compile/--no-compile", "compile_", default=None,
              help="torch.compile the model (default on for real runs).")
@click.option("--doc-mask/--no-doc-mask", "doc_mask", default=None,
              help="Stop attention crossing document boundaries (default on).")
@click.option("--out-dir", type=click.Path(), default=None, help="Checkpoint output dir.")
@click.option("--tokenizer", "tokenizer_path", type=click.Path(exists=True), default=None)
# ModelConfig overrides (default None -> only applied when explicitly passed, so
# --smoke sizing is never silently clobbered by a flag's own default).
@click.option("--vocab-size", type=int, default=None)
@click.option("--hidden-dim", type=int, default=None)
@click.option("--num-heads", type=int, default=None)
@click.option("--n-layer", type=int, default=None)
@click.option("--block-size", type=int, default=None)
def train(smoke, devices, accelerator, num_nodes, precision, dataloader_workers, wandb,
          wandb_project, max_steps, batch_size, lr, compile_, doc_mask, out_dir,
          tokenizer_path, vocab_size, hidden_dim, num_heads, n_layer, block_size):
    """Train a model with Lightning (real run, or a tiny --smoke run)."""
    model_cfg, train_cfg = default_configs(smoke=smoke)

    for name, val in [("vocab_size", vocab_size), ("hidden_dim", hidden_dim),
                      ("num_heads", num_heads), ("n_layer", n_layer),
                      ("block_size", block_size)]:
        if val is not None:
            setattr(model_cfg, name, val)
    for name, val in [("devices", devices), ("accelerator", accelerator),
                      ("num_nodes", num_nodes), ("precision", precision),
                      ("dataloader_workers", dataloader_workers),
                      ("wandb_project", wandb_project),
                      ("max_steps", max_steps), ("batch_size", batch_size),
                      ("lr", lr), ("compile", compile_), ("doc_mask", doc_mask)]:
        if val is not None:
            setattr(train_cfg, name, val)
    if wandb:                       # is_flag: only turn on when passed
        train_cfg.wandb = True
    if out_dir is not None:
        train_cfg.out_dir = Path(out_dir)
    if tokenizer_path is not None:
        train_cfg.tokenizer_path = Path(tokenizer_path)

    run_train(model_cfg, train_cfg)


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
@click.option("--prompt", default="\n<|endoftext|>\n", help="Text to continue.")
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
    click.echo(f"loaded {checkpoint}: step {meta['step']}, val {meta['val']:.3f}, "
               f"{n_params/1e6:.1f}M params\n")
    for i in range(num_samples):
        text = run_generate(model, tok, prompt=prompt, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p,
                            block_size=cfg.block_size, device=dev)
        click.echo(f"--- sample {i + 1} (temp {temperature}, top_p {top_p}) ---")
        click.echo(text.replace("\n<|endoftext|>\n", "\n").strip() + "\n")


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
    cfg = TrainConfig(n_train_docs=n_docs, n_val_docs=0)
    train_docs, _ = load_docs(cfg)
    mb = sum(len(d.encode()) for d in train_docs) / 1e6
    click.echo(f"training {backend} tokenizer (vocab {vocab_size}) on {len(train_docs)} "
               f"docs ({mb:.1f} MB) ...")
    if backend == "hf":
        tok = HFTokenizer.train(train_docs, vocab_size=vocab_size, special=cfg.sep.strip())
    else:
        tok = SimpleTokenizer(vocab_size=vocab_size)
        tok.train_tokenizer(cfg.sep.join(train_docs))
    tok.save(out_path)
    click.echo(f"saved -> {out_path} ({tok.n_vocab} tokens, "
               f"sep_id {tok.sep_id(cfg.sep)})")


if __name__ == "__main__":
    cli()
