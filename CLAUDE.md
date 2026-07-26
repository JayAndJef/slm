# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style — Important

This is a **learning project**. The user wants to write the code themselves. Offer help, general direction, and implementation guidance (concepts, API pointers, small illustrative snippets, debugging hints) — do **not** generate large chunks of code or complete implementations unless explicitly asked.

## Project Overview

A from-scratch small language model built for learning. The full stack is implemented by hand: a byte-level BPE tokenizer, a GPT-style decoder-only transformer, and the training/generation pipeline — all trained on a slice of HuggingFaceFW/fineweb-edu (sample-10BT, cached at `/data/zejiaqi/huggingface-cache`).

- `slm/` — the canonical package (single source of truth). Loosely-coupled modules, each documented: `paths` (repo-root paths), `config` (`ModelConfig`/`TrainConfig` dataclasses + `default_configs`), `tokenizer` (`SimpleTokenizer` — chunked, incremental-count BPE; torch-free), `model` (`MultiheadSelfAttention` with **RoPE** → `TransformerBlock` → `JLM`; no absolute pos embedding), `data` (`load_docs`, parallel `build_corpus` via forkserver, `get_batch`), `dataset` (`WindowIterableDataset` — per-rank-seeded random windows), `lit` (**PyTorch Lightning** surface: `LitJLM` LightningModule, `SLMDataModule`, `CompactCheckpoint` callback), `train` (thin Lightning `Trainer` driver), `generate` (`load_model`, `generate`). Intra-package imports are absolute (`from slm.x import ...`).
- `main.py` — the `click` CLI entrypoint: `uv run main.py {train [--smoke] | generate | train-tokenizer | prepare-data}`. `train` uses Lightning: DDP multi-GPU via `--devices` (count like `2`, or comma-list of PyTorch indices like `0,1`), bf16-mixed, warmup/cosine LR, grad clip, `torch.compile`, RichProgressBar, and optional `--wandb`. `prepare-data` encodes+caches corpora once (no GPU) so DDP runs just mmap them. Model dims overridable via `--hidden-dim/--num-heads/--n-layer/--block-size/--vocab-size`. **DDP effective batch = `batch_size × num_devices`.**
- `notebooks/slm.ipynb` — the original cell-by-cell workspace; keeps its **own** self-contained copies of the classes (it does NOT import `slm/`). Left as the learning scratchpad.
- `notebooks/tokenizer.json` — the trained 4096-vocab tokenizer (merge rules), referenced by `slm.paths.TOKENIZER_PATH`. The model's token ids are baked to this; keep it with any trained model.

### Checkpoint format
`checkpoints/best.pt` = `{"model": state_dict, "step", "val", "config": {vocab_size, hidden_dim, num_heads, n_layer, block_size}}`. `slm.generate.load_model` rebuilds the model from the stored `config`. `save_checkpoint` always stores the **uncompiled** weights (`getattr(model, "_orig_mod", model)`) so `torch.compile`'d runs stay loadable with `strict=True`.

### Key facts / gotchas
- **GPU indexing:** PyTorch orders devices FASTEST_FIRST, so `cuda:0/1` are the Blackwells and `cuda:2–7` the A6000s — this does NOT match `nvidia-smi`'s PCI order. Training currently targets `cuda:3` (an A6000).
- The separator `"\n<|endoftext|>\n"` is a literal string that BPE happened to learn as a single token (id is run-specific — derive it, don't hardcode).
- Corpus is a flat 1D token stream; `get_batch` samples random fixed-length windows (windows may cross document boundaries — intended).
- A trained ~92M model (768 hidden, 12 heads, 12 layers, block 512) reached val loss ~2.97 on ~270M tokens; `checkpoints/best.pt` holds it.
- It's a **base** model (next-token completion), not instruction-tuned — it continues prompts, it doesn't answer them.

## Environment & Package Management

This project uses **uv** (not pip/poetry) with Python 3.10 (pinned in `.python-version`). The virtualenv lives at `.venv/`.

- Add a dependency: `uv add <package>` (updates `pyproject.toml` and `uv.lock`)
- Sync the environment from the lockfile: `uv sync`
- Run the entry point: `uv run main.py`
- Run any script/command in the env: `uv run <command>`

Note: Jupyter/ipykernel packages currently exist in `.venv` but are not tracked in `pyproject.toml`/`uv.lock` (they were installed outside uv, likely by the VS Code Jupyter extension). If notebook tooling needs to be reproducible, add it via `uv add --dev`.

## Current State

- No test suite, linter, or formatter is configured. Correctness is checked with inline asserts (tokenizer round-trips, `get_batch` off-by-one, attention causality).
- Notebooks are the primary workspace; they use the `.venv` Python 3.10 kernel.
- `data/` (encoded corpora) and `checkpoints/` (model weights) are large and local-only.
