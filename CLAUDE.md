# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style — Important

This is a **learning project**. The user wants to write the code themselves. Offer help, general direction, and implementation guidance (concepts, API pointers, small illustrative snippets, debugging hints) — do **not** generate large chunks of code or complete implementations unless explicitly asked.

## Project Overview

A from-scratch small language model built for learning. The full stack is implemented by hand: a byte-level BPE tokenizer, a GPT-style decoder-only transformer, and the training/generation pipeline — all trained on a slice of HuggingFaceFW/fineweb-edu (sample-10BT, cached at `/data/zejiaqi/huggingface-cache`).

- `slm/` — the canonical package (single source of truth). Loosely-coupled modules, each documented: `paths` (repo-root paths), `config` (`ModelConfig`/`TrainConfig` dataclasses + `default_configs`), `tokenizer` (`SimpleTokenizer` — chunked, incremental-count BPE; torch-free), `model` (`MultiheadSelfAttention` with **RoPE** → `TransformerBlock` with **SwiGLU** FFN → `JLM`; no absolute pos embedding; `segment_block_mask` builds the FlexAttention mask), `data` (`load_docs`, parallel `build_corpus` via forkserver, `get_batch`), `dataset` (`WindowIterableDataset` — per-rank-seeded random windows; `segment_ids`), `lit` (**PyTorch Lightning** surface: `LitJLM` LightningModule, `SLMDataModule`, `CompactCheckpoint` callback, `MuonAdamW` optimizer pair), `train` (thin Lightning `Trainer` driver), `generate` (`load_model`, `generate`). Intra-package imports are absolute (`from slm.x import ...`).
- `main.py` — the `click` CLI entrypoint: `uv run main.py {train [--smoke] | generate | train-tokenizer | prepare-data}`. `train` uses Lightning: DDP multi-GPU via `--devices` (count like `2`, or comma-list of PyTorch indices like `0,1`), bf16-mixed, warmup/cosine LR, grad clip, `torch.compile`, RichProgressBar, and optional `--wandb`. `prepare-data` encodes+caches corpora once (no GPU) so DDP runs just mmap them. Model dims overridable via `--hidden-dim/--num-heads/--n-layer/--block-size/--vocab-size`. **DDP effective batch = `batch_size × num_devices`.**
- `notebooks/slm.ipynb` — the original cell-by-cell workspace; keeps its **own** self-contained copies of the classes (it does NOT import `slm/`). Left as the learning scratchpad.
- `notebooks/tokenizer.json` — the trained 4096-vocab tokenizer (merge rules), referenced by `slm.paths.TOKENIZER_PATH`. The model's token ids are baked to this; keep it with any trained model.

### Checkpoint format
`checkpoints/best.pt` = `{"model": state_dict, "step", "val", "config": {vocab_size, hidden_dim, num_heads, n_layer, block_size}}`. `slm.generate.load_model` rebuilds the model from the stored `config`. `save_checkpoint` always stores the **uncompiled** weights (`getattr(model, "_orig_mod", model)`) so `torch.compile`'d runs stay loadable with `strict=True`.

### Key facts / gotchas
- **GPU indexing:** PyTorch orders devices FASTEST_FIRST, so `cuda:0/1` are the Blackwells and `cuda:2–7` the A6000s — this does NOT match `nvidia-smi`'s PCI order. Training currently targets `cuda:3` (an A6000).
- The separator `"\n<|endoftext|>\n"` is a literal string that BPE happened to learn as a single token (id is run-specific — derive it, don't hardcode).
- Corpus is a flat 1D token stream; windows are random fixed-length slices that **may span two or more documents** (~38% do, at block 512 over ~1341-token docs). `TrainConfig.doc_mask` (default **on**, `--no-doc-mask` to disable) stops attention crossing those boundaries via **FlexAttention**.
- **Batch contract: `(x, y, seg)`**, all `(B, block_size)` int64. `seg` is a per-token segment index (inclusive cumsum of the separator token) emitted *unconditionally*, so batch arity never depends on config. `LitJLM` turns it into a `BlockMask` in `training_step` — outside `torch.compile`, once per step rather than per layer — and passes it as `JLM.forward(..., block_mask=)`. The model never sees segment ids, so a future prompt-answer dataset can define segments structurally without touching `model.py`.
- The separator's token id is **derived at runtime**, never hardcoded: `tok.encode(cfg.sep)` returns *two* ids (the split pattern peels off the trailing `\n`); `ids[0]` (`b"\n<|endoftext|>"`) is the real boundary token, since `build_corpus` encodes `sep+doc` contiguously and the trailing newline is absorbed into the next word. `SLMDataModule.setup` asserts that token actually carries the `<|endoftext|>` marker — without it, a tokenizer mismatch would silently make every newline a boundary.
- `segs_per_window` is logged every step as a tripwire: ~1.09 at block 128, higher at 512. A flat 1.0 means the mask has silently degenerated to plain causal.
- Attention has three paths, chosen in `MultiheadSelfAttention.forward`: `_attend_flex` (when a `block_mask` is passed — wins over `use_sdpa`), `_attend_sdpa` (flash, causal-only), `_attend_manual` (longhand, causal-only, kept for learning; **TODO** to hand-write its masked version). The latter two are causal-only, so never "verify" flex against them.
- `JLM.blocks` is an `nn.ModuleList` (not `Sequential`) because blocks take a mask argument; `blocks.N.*` state_dict keys are unchanged. Loss uses `ignore_index=-100` — inert during pretraining, the hook for SFT answer-only loss.
- Best result so far: **val 2.670** (perplexity 14.4) at 768 hidden / 12 heads / 12 layers / block 512, 6000 steps × batch 160 × 3 GPUs = 1.475B tokens, ~1h15m. An earlier ~92M model reached val ~2.97 on ~270M tokens. Both predate SwiGLU, so their weights no longer load (`blocks.N.ffn.0/2` → `ffn.w1/w2/w3`) — the loss numbers are the surviving record. Val split is deterministic (`seed`, `n_train_docs`, `n_val_docs`, tokenizer, `block_size` unchanged), so these are comparable.
- **`CompactCheckpoint` starts each run at `self.best = inf`**, so the first validation overwrites `out_dir/best.pt` no matter how good the stored model is. Pass `--out-dir` per run, or seed `self.best` from the existing file.
- Do **not** raise `n_train_docs` without moving the val split: val is `docs[n_train_docs : n_train_docs + n_val_docs]`, but the cache filename is `val_{tag}_{n_val_docs}.npy` — a bigger train set absorbs the old val docs while the stale val cache is silently reused.
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
