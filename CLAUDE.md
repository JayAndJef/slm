# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style — Important

This is a **learning project**. The user wants to write the code themselves. Offer help, general direction, and implementation guidance (concepts, API pointers, small illustrative snippets, debugging hints) — do **not** generate large chunks of code or complete implementations unless explicitly asked.

## Project Overview

A from-scratch small language model built for learning. The full stack is implemented by hand: a BPE tokenizer, a GPT-style decoder-only transformer, and the training/generation pipeline — trained on **HuggingFaceTB/cosmopedia** (synthetic textbook-style text, cached at `/data/zejiaqi/huggingface-cache`; `HF_HOME` is set in `slm/paths.py` **before** any HF import, since `huggingface_hub` resolves it at import time).

- `slm/` — the canonical package (single source of truth). Loosely-coupled modules, each documented: `paths` (repo-root paths + `HF_HOME`), `config` (`ModelConfig`/`TrainConfig` dataclasses + `default_configs`), `tokenizer` (`Tokenizer` Protocol; `HFTokenizer` rust-backed **default**, `SimpleTokenizer` from-scratch fallback, `load_tokenizer` sniffs which by JSON schema), `model` (`MultiheadSelfAttention` with **QK norm + RoPE** → `TransformerBlock` with **SwiGLU** FFN → `JLM`; no absolute pos embedding; `segment_block_mask` builds the FlexAttention mask), `data` (`load_docs`, parallel `build_corpus` via forkserver, `get_batch`), `dataset` (`WindowIterableDataset` — shuffled non-overlapping windows; `segment_ids`), `checkpoint` (the compact on-disk format — `save`/`load`), `lit` (**PyTorch Lightning** surface: `LitJLM`, `SLMDataModule`, `CompactCheckpoint`, `MuonAdamW`, `lr_at`), `train` (thin Lightning `Trainer` driver), `generate` (`load_model`, `generate`). Intra-package imports are absolute (`from slm.x import ...`).
- `main.py` — the `click` CLI: `uv run main.py {train [--smoke] | continue-train | generate | train-tokenizer | prepare-data}`. Training flags shared by `train` and `continue-train` live in one `_TRAINING_OPTIONS` list applied by the `training_options` decorator, so the two cannot drift; they differ in **defaults**, not in what they accept. Every flag defaults to `None` meaning "not passed" — a flag defaulting to its own value would silently clobber `--smoke` sizing or a continuation's schedule. `prepare-data` encodes+caches corpora once (no GPU) so DDP runs just mmap them. **DDP effective batch = `batch_size × num_devices`.**
- `notebooks/slm.ipynb` — the original cell-by-cell workspace; keeps its **own** self-contained copies of the classes (it does NOT import `slm/`). Left as the learning scratchpad.
- `data/tokenizer-32k.json` — the real tokenizer (HF format, vocab 32000), at `slm.paths.TOKENIZER_PATH`. `notebooks/tokenizer.json` is the old 4096-vocab `SimpleTokenizer` artifact kept as the fallback (`paths.SIMPLE_TOKENIZER_PATH`) and still loaded by the notebook. `.gitignore` uses `data/*` + `!data/*.json` so the tokenizer ships but corpora don't.

### Checkpoint format
`slm/checkpoint.py` is the **only** module that knows the on-disk shape: `{"model": state_dict, "step", "val", "config": {vocab_size, hidden_dim, num_heads, n_layer, block_size}}`. `save` stores the **uncompiled** weights (`getattr(model, "_orig_mod", model)`) so `torch.compile`'d runs load with `strict=True`, and maps tensors by `data_ptr` before `.cpu()` so shared storage survives — without that, `torch.save`'s dedup is defeated and the tied embedding/lm_head matrix is written twice (930 MB → 799 MB when fixed). `load` returns `(state_dict, ModelConfig, meta)`; the architecture comes back from the file because weights can only be rebuilt at the dimensions they were trained at.

### Continuing a run
`continue-train --init-from CK --out-dir DIR` starts from a checkpoint's **weights only** — the compact format carries no optimizer moments, scheduler position or dataloader offset, so Muon/AdamW momentum rebuilds over the first few hundred steps. Consequences:

- The LR schedule restarts from scratch, so a continuation needs one that makes sense *from a trained model*. Defaults are a **pure anneal** (`warmup_steps=0`, `decay_frac=1.0`, `lr=1e-4`, `min_lr=0.0`, `max_steps=4000`) — no warmup, no stable phase, straight decay. To extend training instead, re-warm: higher `--lr`, nonzero `--warmup-steps`, `--decay-frac` below 1.0, and expect val to regress before it improves.
- Architecture is read from the checkpoint; `train()` accepts `model_cfg=None` for this path and model-dimension flags are deliberately absent from the command.
- `--out-dir` is **required** and should differ from the source's — the source model is the thing the run is trying to beat.
- `train()` seeds `CompactCheckpoint(best=...)` from the source checkpoint's `val`, so only genuine improvements are saved.

### Key facts / gotchas
- **GPU indexing:** PyTorch orders devices FASTEST_FIRST, so `cuda:0/1` are the Blackwells (95 GiB) and `cuda:2–7` the A6000s (44.4 GiB) — this does NOT match `nvidia-smi`'s PCI order, where the same cards are 6/7 and 0–5. The Blackwells are frequently occupied by other users; check `nvidia-smi --query-compute-apps` before assuming they're free. Recent runs use `--devices 2,3,4,5`.
- **`TrainConfig.corpus_hash`** digests the tokenizer bytes **plus** `(dataset_mix, seed, sep, dataset_name)` and goes in the corpus filenames (`train_cosmo_all_3c061ac0.npy`). Every one of those fails *silently* when it drifts — a swapped tokenizer leaves all ids in range and the loss curve looks normal; a changed mix or seed moves the val split with no signal. Changing any of them forces a visible re-encode instead of a wrong run. `n_workers` is deliberately excluded (it permutes shard concatenation order, not corpus contents).
- **`sampler_seed` / `window_seed` vs `seed`.** `seed` shuffles the document pool and is *in* `corpus_hash`, so bumping it triggers a 17-minute re-encode. `sampler_seed` (via the `window_seed` property) reshuffles only the window order and is **outside** the hash — that's what a continuation run needs, or it replays the exact window order the previous run already trained on. Val is pinned to `seed`, never `window_seed`, so it measures the same thing across runs.
- **`sep_id` is derived per backend, never hardcoded.** `HFTokenizer.sep_id` looks the marker up by name (`token_to_id("<|endoftext|>")`) — it **cannot** use `ids[0]`, because with ByteLevel `encode("\n<|endoftext|>\n")` yields `[Ċ, EOT, Ċ]` and `ids[0]` is a bare newline, which would make every `\n` a document boundary. `SimpleTokenizer.sep_id` uses `ids[0]` and asserts the marker is in that token's bytes. Corpus `a[0]` is therefore a newline token and `a[1]` is the separator — that is expected, not a bug.
- **Batch contract: `(x, y, seg)`**, all `(B, block_size)` int64. `seg` is a per-token segment index (inclusive cumsum of the separator token) emitted *unconditionally*, so batch arity never depends on config. `LitJLM._block_mask` turns it into a `BlockMask` in `training_step` — outside `torch.compile`, once per step rather than per layer — and passes it as `JLM.forward(..., block_mask=)`. The model never sees segment ids, so a future prompt-answer dataset can define segments structurally without touching `model.py`.
- `segs_per_window` is logged every step as a tripwire: ~1.7 at block 512, ~2.5 at block 1024 (docs average 694 tokens). A flat 1.0 means the mask has silently degenerated to plain causal.
- **Document lengths (measured on the real corpus):** mean 694 tokens, p50 641, p90 1040, p99 1612, max 4612. With `doc_mask` on, block size only helps up to the document length — mean attendable context is 302 tokens at block 512, 344 at 1024 (**+14%**), 348.6 at 2048 (**+1.2%**). So 1024 is the sensible ceiling; beyond it is wasted.
- Attention has three paths, chosen in `MultiheadSelfAttention.forward`: `_attend_flex` (when a `block_mask` is passed — wins over `use_sdpa`), `_attend_sdpa` (flash, causal-only), `_attend_manual` (longhand, causal-only, kept for learning; **TODO** to hand-write its masked version). The latter two are causal-only, so never "verify" flex against them.
- `JLM.blocks` is an `nn.ModuleList` (not `Sequential`) because blocks take a mask argument. Loss uses `ignore_index=-100` — inert during pretraining, the hook for SFT answer-only loss.
- **Tied embedding needs an explicit init.** `lm_head.weight = embedding.weight` makes the head inherit `nn.Embedding`'s N(0,1), giving logit std 11–28 and an initial loss of ~117. `nn.init.normal_(embedding.weight, std=0.02)` after tying is the fix. **Sanity check: initial loss must be ≈ `ln(vocab_size)` = 10.37.**
- **Muon group selection is name-based:** `is_hidden = ".blocks." in name and p.ndim == 2`, evaluated on `LightningModule.named_parameters()` (so keys read `model.blocks.…`, and `model._orig_mod.blocks.…` under compile). On a bare `JLM` the leading dot doesn't match and the Muon group comes out empty — `torch.optim.Muon` then raises "empty parameter list". QK-norm/LayerNorm gains are 1D and correctly route to AdamW.
- **`torch.compile` roughly halves activation memory** — do not size batches from an eager measurement. On a 44.4 GiB A6000 at hidden 1024 / block 1024, compiled peaks are B=32 → 22.4 GiB, B=40 → 27.5, B=48 → 32.6, B=56 → 37.8, B=64 → OOM; eager OOMs already at B=40. The trigger is the `B×T×vocab` logits tensor (4.0 GiB in bf16 at B=64), which cross-entropy then upcasts to fp32 and holds a gradient for. **Chunked cross-entropy is the untaken lever** that would free ~13 GiB.
- **Val loss is not comparable across vocab size, block size, or dataset.** `val_bpb = val_loss × tokens_per_byte / ln 2` (measured at `SLMDataModule.setup` from 200k val tokens) fixes the vocab axis only. Record `(corpus_hash, block_size, vocab_size)` with every number.

### Results
| model | corpus | tokens | val_loss | val_bpb | notes |
|---|---|---|---|---|---|
| **187.1M** (1024/16/12, block 1024, vocab 32k) | cosmopedia `3c061ac0` | 5.90B (1.23 ep) | **1.733** | **0.499** | 30k steps × batch 48 × 4 A6000s, 7h44m. lr 1e-3 → 1e-4, warmup 100, `decay_frac` 0.2 |
| ~150M (768/12/12, block 512, vocab 4096) | fineweb-edu | 1.475B | 2.670 | — | pre-SwiGLU; weights no longer load |
| ~92M (vocab 4096) | fineweb-edu | ~270M | ~2.97 | — | pre-SwiGLU |

Only the first row is a live baseline — the others measured a different model on different data with a different tokenizer, and no arithmetic recovers the comparison. For the current corpus, `tokens_per_byte = 0.200`, so **`val_bpb = val_loss × 0.289`**.

Observed at 30k steps: the 6k-step cooldown (24k→30k) gained **0.18** while the preceding 16k stable steps gained only **0.125** — normal WSD behaviour, and the tail was still descending, i.e. not converged. Note the run annealed only to `min_lr = lr/10`, not to 0.

### Corpus / throughput reference
- `dataset_mix`: `auto_math_text` (1.95M rows), `stanford` (1.02M), `stories` (capped 1.8M of 4.99M), `web_samples_v2` (capped 1.8M of 10.3M), `wikihow` (179k), `openstax` (126k), `khanacademy` (24k). `web_samples_v1` is excluded. **`load_docs` takes val from the FRONT of the pool**, train from what follows, so raising `n_train_docs` cannot swallow val docs.
- Encoded: **4.784B train tokens** (6.895M docs), 3.45M val tokens (5000 docs), **4.99 bytes/token**. One epoch at batch 48 × block 1024 × 4 GPUs (196,608 tokens/step) = 24,333 steps.
- Encoding rate ~1.66 MB/s/core; 16 workers do the 23.87 GB corpus in ~15 min, then a few minutes to gather 8.7 GB through the pool pipes and `np.save`. `build_corpus` logs only at start and end, so that whole window looks silent.
- Training throughput at batch 48 / block 1024: **60.8k tok/s/GPU** standalone, **215.6k tok/s across 4 DDP ranks** (~11% all-reduce cost), 0.92 s/step.
- `eval_iters=18` is exactly one clean pass over the val shard at batch 48 × 4 ranks (3,373 windows / 4 / 48 = 17.6). Larger values just resample the same windows.

## Environment & Package Management

This project uses **uv** (not pip/poetry) with Python 3.10 (pinned in `.python-version`). The virtualenv lives at `.venv/`.

- Add a dependency: `uv add <package>` (updates `pyproject.toml` and `uv.lock`)
- Sync the environment from the lockfile: `uv sync`
- Run the entry point: `uv run main.py`
- Run any script/command in the env: `uv run <command>`

Scripts run from outside the repo root need `PYTHONPATH=/data/zejiaqi/model_tests/slm` — Python puts the *script's* directory on `sys.path`, not the cwd, so `uv run python /tmp/foo.py` cannot import `slm`.

Note: Jupyter/ipykernel packages exist in `.venv` but are not tracked in `pyproject.toml`/`uv.lock` (installed outside uv, likely by the VS Code Jupyter extension). If notebook tooling needs to be reproducible, add it via `uv add --dev`.

## Current State

- No test suite, linter, or formatter is configured. Correctness is checked with inline asserts (tokenizer round-trips, `get_batch` off-by-one, attention causality) and ad-hoc verification scripts.
- `data/` (encoded corpora) and `checkpoints/` (model weights) are large and local-only.
- It's a **base** model (next-token completion), not instruction-tuned — it continues prompts, it doesn't answer them. Cosmopedia is prompt/response data but only the `text` column is trained on; `prompt` is the instruction given to Mixtral to *generate* the text, not something to train on.
