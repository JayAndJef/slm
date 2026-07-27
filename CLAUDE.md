# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style — Important

A learning project, with a deliberate split of work:

- **The user writes** core model code and features — anything in `slm/model.py`, training-step internals, architecture and sizing decisions.
- **Claude writes** the surrounding plumbing: corpus/data layer, config, checkpointing, CLI, tokenizer wiring, eval harnesses. Implement these directly rather than offering snippets.
- **The user runs** every long GPU job. Claude stages the command and the gate to read at t+30min, and runs the cheap verification itself (smoke runs, corpus builds, structural asserts).

Before touching `model.py` or a training step, hand it back.

## Project Overview

A from-scratch small language model built for learning. The full stack is implemented by hand: a BPE tokenizer, a GPT-style decoder-only transformer, and the training/generation pipeline — trained on **HuggingFaceTB/cosmopedia** (synthetic textbook-style text, cached at `/data/zejiaqi/huggingface-cache`; `HF_HOME` is set in `slm/paths.py` **before** any HF import, since `huggingface_hub` resolves it at import time).

- `slm/` — the canonical package (single source of truth). Loosely-coupled modules, each documented: `paths` (repo-root paths + `HF_HOME`), `chat` (**leaf, imports nothing**: the ChatML template, `specials(n)`, `is_reserved`), `config` (Spec vs Config — see below), `tokenizer` (`Tokenizer` Protocol; `HFTokenizer` rust-backed **default**, `SimpleTokenizer` from-scratch fallback, `load_tokenizer` sniffs which by JSON schema), `model` (`MultiheadSelfAttention` with **QK norm + RoPE** → `TransformerBlock` with **SwiGLU** FFN → `JLM`; no absolute pos embedding; `segment_block_mask` builds the FlexAttention mask), `render` (`Renderer` Protocol, `PretrainRenderer`/`ChatRenderer`, `Rendered`, BFD packing), `corpus` (**the only module that knows the on-disk corpus shape**: `load_docs`, `corpus_hash`, `Corpus`, `locate`, `build`, forkserver pool), `dataset` (`WindowIterableDataset` — shuffled non-overlapping windows; `segment_ids`, `n_windows`), `checkpoint` (`save` writes the compact format; `load` **sniffs** Lightning / compact-v2 / compact-v1), `lit` (**PyTorch Lightning** surface: `LitJLM`, `SLMDataModule`, `MuonAdamW`, `lr_at`), `train` (thin Lightning `Trainer` driver), `generate` (`load_model`, `stream`, `generate`). Intra-package imports are absolute (`from slm.x import ...`). There is no `slm/data.py` — it folded into `corpus`.
- `main.py` — the `click` CLI: `uv run main.py {train [--smoke] | continue-train | generate | export | inspect | prepare-data | list-splits | train-tokenizer}`. Training flags shared by the training commands live in one `_TRAINING_OPTIONS` list applied by the `training_options` decorator, so they cannot drift; they differ in **defaults**, not in what they accept. Every flag defaults to `None` meaning "not passed" — a flag defaulting to its own value would silently clobber `--smoke` sizing or a continuation's schedule. **`--out-dir` is required on `train`** (a from-scratch run seeds `best=inf`, so its first validation would overwrite whatever sits in the default dir). **DDP effective batch = `batch_size × num_devices`.**
- **Training commands never build a corpus.** They `corpus.locate(...)` and error with the `prepare-data` command to run. Lightning's DDP re-executes `sys.argv` per rank, so building there would race N processes on a multi-GB write; and it turns a stray hash-moving flag into an error rather than a silent 15-minute re-encode at the head of a multi-hour run.

### Specs vs Configs
A **Spec** is artifact identity and is hashed into the corpus filename: `SourceSpec` (`dataset_name` + a flat tuple of `SourcePart(config, split, cap)` + `columns` + `seed`), `RenderSpec` (`kind`, `sep`, `pack`, `pack_block`, …), `CorpusSpec` (the pair + a `name` stamped from its `CORPUS_PRESETS` key, so flag and filename cannot disagree). A **Config** is run knobs and is never hashed: `ModelConfig`, `TrainConfig`. `CorpusSpec` deliberately carries **no paths** — `tokenizer_path`/`data_dir`/`hf_cache_dir` are locations, not identity, and live on `TrainConfig`.

`parts` is a flat list rather than `{config: cap}` + one split because datasets disagree about which axis carries the structure: cosmopedia is 7 configs × `train`, smoltalk2 is one `SFT` config × 25 splits. It is always **explicit, never a pattern** — a glob is stable in the file while the set it matches is not, so an upstream split appearing would change the corpus with no change to its hash. Write the list with `list-splits`.
- `notebooks/slm.ipynb` — the original cell-by-cell workspace; keeps its **own** self-contained copies of the classes (it does NOT import `slm/`). Left as the learning scratchpad.
- `data/tokenizer-32k.json` — the live tokenizer (HF format, vocab 32000), at `slm.paths.TOKENIZER_PATH`, and the **only** one. v1 and every v1-era checkpoint were deleted once v2 superseded them; nothing on disk predates the reserved-special block. `notebooks/tokenizer.json` is the old 4096-vocab `SimpleTokenizer` artifact (`paths.SIMPLE_TOKENIZER_PATH`), still loaded by the notebook. `.gitignore` uses `data/*` + `!data/*.json` so tokenizers ship but corpora don't — which is also why corpus metadata is `<name>.meta`, not `.meta.json`.
- **Housekeeping:** `prepare-data --prune` lists cached corpora that no preset resolves to (`--delete` removes them). The hash-in-filename scheme has no garbage collection by design — a changed tokenizer/source/renderer mints a new file so drift is visible — so orphans accumulate at every such change and this is what collects them.

### The reserved special-token block
v2 declares **32 specials on the `BpeTrainer`** (`chat.specials()`), so they are counted *within* `vocab_size`: `n_vocab` stays exactly 32000, `ModelConfig.vocab_size` never moves, and there is no embedding to resize. Ids 0..31 are `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, then 29 `<|reserved_k|>` slots. Adding specials to an already-trained tokenizer instead appends ids *past* `vocab_size`, which then disagrees with the number the embedding is sized from.

**Measured cost of the block: 0.004%.** Three tokenizers on held-out `pool[20000:22000]` — v1 (31743 merges, sampled `pool[0:10k]`) 5.0479 bytes/token; a control (31741 merges, `pool[5k:15k]`) 5.0322; v2 (31712 merges, same sample) 5.0320. So control→v2 isolates the 29 extra reserved slots at 0.004%, and v1→control is 0.31% of *training-sample* variation, not block cost. Do not read the v1↔v2 difference as a regression.

Reserved slots exist so a future marker is a **rename, not a re-encode**: `Tokenizer.fingerprint` excludes reserved ids **by id range**, never by matching the name. Verified — renaming `<|reserved_2|>` to `<|tool_call|>` leaves the fingerprint unchanged and the id fixed, while different training text or a different `n_reserved` both move it. (Filtering by name is self-defeating: a renamed slot stops matching, enters the digest, and moves the hash anyway.)

**v1 vs v2 provenance:** v1 was trained on `pool[0:10000]`, which contains the whole `pool[0:5000]` val split, so its merges saw val text and its `val_bpb` is very slightly optimistic. `train-tokenizer` samples `pool[n_val_docs:]` and prints the source it drew from, so a changed mix or seed is visible.

### Checkpoints: Lightning writes, compact publishes
**Training writes Lightning checkpoints only.** `ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, save_on_exception=True)` produces `last.ckpt` (the *latest* — what `--resume` needs) and `step{N}-val{X}.ckpt` (best-by-val — what `continue-train`/`sft` should start from). They carry optimizer moments, scheduler position and loop state, so `--resume` continues a run *exactly*. There is no `best.pt` written during training.

**`slm/checkpoint.py` `load` sniffs three shapes** and normalizes all of them to one `(state_dict, ModelConfig, meta)` triple, so `generate`/`continue-train`/`sft` never branch on format: Lightning (`state_dict` + `hyper_parameters`, keys stripped of `model.`/`model._orig_mod.`), compact v2 (`format: 2`), compact v1 (no `format` key — pre-format-2 files; none remain on disk, kept for anything restored from elsewhere). It imports nothing from Lightning; reading one is just `torch.load`.

**`export CKPT --out X.pt`** writes the compact format for handing a model onward: `{format, model, config, step, val, meta_json, tokenizer_json, tokenizer_fingerprint}`. Everything non-tensor is a **JSON string**, because `Path` and `datetime` raise under `torch.load(weights_only=True)` — storing them directly would force `weights_only=False` exactly when the file becomes shareable. Verified: the export loads with `weights_only=True`, and keeps **255 unique `data_ptr`s across 256 tensors**, i.e. `save` still maps by `data_ptr` before `.cpu()` so the tied embedding/lm_head matrix isn't written twice (930 MB → 799 MB). The embedded tokenizer (2.3 MB) means `generate` needs no `--tokenizer` and cannot be paired with the wrong vocabulary.

Every checkpoint written by `train()` records its provenance in `hyper_parameters` — `corpus` (`name`, `hash`, `render_kind`), `tokenizer_fingerprint`, `tokens_per_byte`. `checkpoint.load` surfaces that as `meta`, which is what lets `generate` choose ChatML for an SFT model with no `--chat`, `generate`/`sft-eval` refuse a tokenizer the weights were not trained against, and `inspect` derive `val_bpb`. `ModelConfig.from_dict` filters unknown keys, so the extra hparams cannot disturb the architecture read. `meta["val"]` for a Lightning file is read out of `ModelCheckpoint`'s callback state (`current_score`); nothing else in the file carries it.

**Three restart modes, deliberately distinct** — say which you mean: `--resume` (Lightning: momentum, schedule position, loop counters), `continue-train --init-from` (weights only, fresh schedule — the anneal/re-warm workflow), `sft --init-from` (weights only, different corpus).

**`--resume` does *not* restore the dataloader position.** Lightning only restores loader state for loaders satisfying `_Stateful` (`combined_loader.py:380`), and `WindowIterableDataset` behind a plain `DataLoader` is not one — so a resumed run restarts at the head of its window permutation and re-trains on windows the killed run already saw. Lightning does not even warn: `training_epoch_loop.py:218` gates that warning behind `num_training_batches != inf`, and this dataset is infinite. Pass `--sampler-seed` on a resume to draw a different permutation.

### Continuing a run
`continue-train --init-from CK --out-dir DIR` starts from a checkpoint's **weights only** — no optimizer moments, scheduler position or dataloader offset, so Muon/AdamW momentum rebuilds over the first few hundred steps. (Use `--resume` instead if you meant to continue an interrupted run.) Consequences:

- The LR schedule restarts from scratch, so a continuation needs one that makes sense *from a trained model*. Defaults are a **pure anneal** (`warmup_steps=0`, `decay_frac=1.0`, `lr=1e-4`, `min_lr=0.0`, `max_steps=4000`) — no warmup, no stable phase, straight decay. To extend training instead, re-warm: higher `--lr`, nonzero `--warmup-steps`, `--decay-frac` below 1.0, and expect val to regress before it improves.
- Architecture is read from the checkpoint; `train()` accepts `model_cfg=None` for this path and model-dimension flags are deliberately absent from the command.
- `--out-dir` is **required** and should differ from the source's — the source model is the thing the run is trying to beat.
- **Best-so-far starts at infinity, not at the source's val.** `ModelCheckpoint` knows nothing about the checkpoint the run was seeded from, so a continuation that never beats its input still writes a `step{N}-val{X}.ckpt`. That file is not a result — read the `val` in its name against the source's before treating it as one. Seeding it deliberately isn't done: `sft` runs on a different corpus, where the source's val is not on the same scale and any seed would suppress every checkpoint.

### Key facts / gotchas
- **Import `slm` (or `slm.paths`) before anything HuggingFace.** `paths.py` sets `HF_HOME`, `HF_HUB_CACHE` and `HF_XET_CACHE` to `/data/zejiaqi/huggingface-cache`, but `huggingface_hub` resolves those into module constants *at import time*, so setting them afterwards has no effect. A script that does `from datasets import load_dataset` above `from slm import ...` silently downloads to `~/.cache` on `/home` instead — which happened here and filled an already-97%-full `/home` with 29 GB of smoltalk2, killing the download with `No space left on device`. `paths.py` now emits a `RuntimeWarning` when it finds `huggingface_hub` already in `sys.modules`, so the failure is visible rather than a mystery. Note `/home` runs near capacity independently of this project; `/data` has ~2.7 TB.
- **GPU indexing:** PyTorch orders devices FASTEST_FIRST, so `cuda:0/1` are the Blackwells (95 GiB) and `cuda:2–7` the A6000s (44.4 GiB) — this does NOT match `nvidia-smi`'s PCI order, where the same cards are 6/7 and 0–5. The Blackwells are frequently occupied by other users; check `nvidia-smi --query-compute-apps` before assuming they're free. Recent runs use `--devices 2,3,4,5`.
- **`corpus.corpus_hash(spec, tokenizer_fingerprint=)`** digests the tokenizer **fingerprint** plus `source.hash_fields()` and `render.hash_fields()`, and goes in the corpus filenames (`train_cosmopedia_5000+all_c19dd0bf.npy` — the train name carries **both** doc counts, since `load_docs` slices train *after* the val docs and `n_val_docs` therefore moves the train slice too). Every input fails *silently* when it drifts — a swapped tokenizer leaves all ids in range and the loss curve looks normal; a changed source moves the val split with no signal. Folding them into the filename forces a visible re-encode instead of a wrong run. It takes the **fingerprint, not the file bytes**, so renaming an unclaimed reserved slot doesn't invalidate the corpus. `hash_fields()` excludes what provably cannot change the stream: doc counts (the filename carries them) and, on `RenderSpec`, the fields dead for that `kind` plus `mask_partial_head` (applied at read time, not write time).
- **The hash covers *spec* drift, not *code* drift.** Editing `render.py` changes the token stream while `corpus_hash` stays put, so an existing corpus is silently stale — no hash can cover this. After changing renderer logic, delete the affected `.npy`/`.meta` by hand. (Hit during development: switching over-long chat examples from drop to truncate changed the stream under an unchanged `3413f960`.)
- **`n_workers` and `batch_docs` are outside the hash**, recorded in `meta` instead. Neither changes which documents are present. But `n_workers` does change *order* (`shard(contiguous=False)` interleaves differently for 8 vs 16) and — under `pack="bfd"` — **which examples share a bin**, hence the attention mask. So two machines with different core counts write differently-ordered streams to the same filename; meta makes that visible without making it a re-encode.
- **`sampler_seed` vs `SourceSpec.seed`.** `seed` shuffles the document pool and is *in* the hash, so bumping it triggers a full re-encode. `sampler_seed` (via `config.window_seed(train_cfg, source)`) reshuffles only the window order and is **outside** the hash — that's what a continuation run needs, or it replays the exact window order the previous run already trained on. Val is pinned to `source.seed + 10_000`, never the window seed, so it measures the same thing across runs. The two now live on different objects; `window_seed` is a named function so the explanation stays in one place.
- **`sep_id` is derived per backend, never hardcoded.** `HFTokenizer.sep_id` looks the marker up by name (`token_to_id("<|endoftext|>")`) — it **cannot** use `ids[0]`, because with ByteLevel `encode("\n<|endoftext|>\n")` yields `[Ċ, EOT, Ċ]` and `ids[0]` is a bare newline, which would make every `\n` a document boundary. `SimpleTokenizer.sep_id` uses `ids[0]` and asserts the marker is in that token's bytes. Corpus `a[0]` is therefore a newline token and `a[1]` is the separator — that is expected, not a bug.
- **Batch contract: `(x, y, seg)`**, all `(B, block_size)` int64, with `-100` already written into `y`. `seg` is emitted *unconditionally*, so batch arity never depends on config, and **`model.py` needed zero changes for SFT**. `LitJLM._block_mask` turns `seg` into a `BlockMask` in `training_step` — outside `torch.compile`, once per step rather than per layer.
- **Two sidecars, sliced at different offsets — this is the off-by-one that matters.** A chat corpus ships `<name>.start.npy` and `<name>.target.npy` (`uint8`, parallel to the tokens). `is_start[i : i+bs]` aligns with **`x`** (`seg = cumsum(...)`, cast to int64 first — `np.cumsum` on uint8 returns *uint64*). `is_target[i+1 : i+1+bs]` aligns with **`y`**, the shifted stream, and writes `-100` where 0 — **after** `.astype(np.int64)`, since writing `-100` into a uint16 view gives 65436, in range for a 32k vocab and a wrong-but-plausible loss. They are two separately named files precisely so the offset difference is visible at the call site.
- **`mask_partial_head` masks the *whole* window when it contains no example start.** Such a window sits entirely inside one long example, so every target in it is amputated — the empty-`head` case is the worst one, not a no-op. Only relevant to an unpacked (`pack="flat"`) chat corpus; a bfd corpus always opens a window on a bin boundary, so `head[0] == 0` and nothing is masked either way.
- **A packed corpus derives segments from `is_start`, not from `sep_id`** — with an `<|im_end|>` per turn, `cumsum(x == sep_id)` would split one conversation into N segments and turn 2 could not attend to turn 1. Nothing errors; the model just never learns context. `assert_compatible` requires `has_start or sep_id is not None`, so segments can never silently collapse to plain causal.
- **`segs_per_window` has two different expected values.** Flat: `block_size × sep_frac + 1` (2.48 at block 1024, `sep_frac` 0.001442). Packed: `n_examples/n_bins + 1`, where the `+1` is the padding segment. Reading a packed run against the flat identity gives a wrong expectation.
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
| **187.1M** anneal — v1 reference | cosmopedia `3c061ac0` | +0.79B | **1.7113** | **0.49334** | `continue-train` from the row below, 4000 steps, pure anneal to 0. wandb `qymtkyg1`. **Weights deleted** |
| **187.1M** (1024/16/12, block 1024, vocab 32k) | cosmopedia `3c061ac0` | 5.90B (1.23 ep) | 1.7326 | 0.49950 | 30k steps × batch 48 × 4 A6000s, 7h44m. lr 1e-3 → 1e-4, warmup 100, `decay_frac` 0.2. wandb `7ncni0dr`. **Weights deleted** |
| ~150M (768/12/12, block 512, vocab 4096) | fineweb-edu | 1.475B | 2.670 | — | pre-SwiGLU; weights no longer load |
| ~92M (vocab 4096) | fineweb-edu | ~270M | ~2.97 | — | pre-SwiGLU |

Only the first two rows are comparable — the others measured a different model on different data with a different tokenizer, and no arithmetic recovers the comparison. For the current corpus, `tokens_per_byte = 0.200`, so **`val_bpb = val_loss × 0.289`**.

**No v1 weights survive** — both rows are numbers only, kept as the reference the v2 re-pretrain is gated against (`val_bpb` must land in [0.490, 0.500]). They are *not* comparable to anything trained on a different tokenizer except through bpb, which is exactly why bpb is the gate. The `1.733` figure in older notes refers to the second row before its anneal.

Observed at 30k steps: the 6k-step cooldown (24k→30k) gained **0.18** while the preceding 16k stable steps gained only **0.125** — normal WSD behaviour, and the tail was still descending, i.e. not converged. That run annealed only to `min_lr = lr/10`; the separate 4000-step anneal to 0 then bought a further **0.021**.

### SFT corpus: `HuggingFaceTB/smoltalk2`
Subset **`SFT`** (3,383,242 rows over 25 splits; the other subsets are `Mid` and `Preference`). Columns: `messages`, `chat_template_kwargs`, `source`. Regenerate this table with `uv run main.py list-splits HuggingFaceTB/smoltalk2 --config SFT`.

Splits divide 10 `_think` (1,481,833 rows) / 15 `_no_think` (1,901,409). **A `_think$` regex matches both** — `_no_think` also ends in `_think`. This is one reason corpus specs name their `(config, split)` parts explicitly instead of matching a pattern; the other is that a pattern is stable in the file while the set it matches is not, so an upstream split appearing would change the corpus with no change to its `corpus_hash`.

**Selected: 11 splits, 1,572,190 rows** — the `_no_think` set minus four:

| excluded | rows | why |
|---|---|---|
| `smoltalk_multilingual_8languages_lang_5_no_think` | 254,047 | base model is English-only cosmopedia and the 32k tokenizer was trained on it; non-English tokenizes far worse per byte |
| `xlam_traces_no_think` | 59,962 | structured JSON function-call output, a format we're not targeting |
| `hermes_function_calling_v1_no_think` | 8,961 | same |
| `LongAlign_64k_context_lang_annotated_lang_6_no_think` | 6,249 | 64k-context examples; every one overflows block 1024 |

Kept, largest first: `OpenThoughts3_1.2M_no_think_no_think` (435,193 — the doubled suffix is upstream's), `smoltalk_smollm3_smol_magpie_ultra_no_think` (406,843), `OpenHermes_2.5_no_think` (384,900), `smoltalk_smollm3_smol_summarize_no_think` (96,061), `Mixture_of_Thoughts_science_no_think` (86,110), `smoltalk_smollm3_smol_rewrite_no_think` (53,262), `smoltalk_smollm3_systemchats_30k_no_think` (33,997), `smoltalk_smollm3_explore_instruct_rewriting_no_think` (30,391), `tulu_3_sft_personas_instruction_following_no_think` (29,970), `table_gpt_no_think` (13,203), `smoltalk_smollm3_everyday_conversations_no_think` (2,260).

Thinking splits are excluded wholesale: a 187M model cannot carry a chain of thought, and traces are long enough to be exactly the examples that overflow `block_size`.

**Encoded (`3413f960`, val split):** 1.516M tokens over 1480 bins, 2559 examples, `target_frac` **0.701**, fill **95.4%**, 151 truncated / 45 dropped = **3.09% of tokens**. Train draws from 1,570,190 conversations.

Three renderer decisions the numbers forced:
- **Over-long single turn-pairs are truncated, not dropped.** Dropping is sharply length-biased — it discarded **19.7%** of val tokens from ~8% of examples, since the overflowing ones are exactly the long-answer ones. Truncating brings that to 3.09%. A truncated example loses its closing `<|im_end|>` and so never teaches stopping; the ones that fit still do. The 45 remaining drops are prompts that alone fill the block, where no target survives.
- **System messages fold into the first user turn.** The template has no system role (one fewer train/inference drift axis), but dropping the text would gut `smoltalk_smollm3_systemchats_30k_no_think`, whose conversations are *about* the system prompt.
- **`target_frac` 0.70, not the 0.3–0.5 a rule of thumb suggests** — smoltalk2's assistant turns are long. Gate `frac_loss_tokens` against the recorded value, averaged over a few hundred windows: per-batch spread is 0.66–0.76, so one batch of 4 reads 0.80 and looks broken.

**BFD packing is single-threaded in the parent, *after* the parallel encode, and logs nothing** — that window looks identical to a hang. `_ffd_bins` scanning every open bin per example measured 0.29 s at 5k examples, 4.5 s at 20k, 70 s at 80k (clean quadratic), i.e. ~10 hours on smoltalk2's ~1.9M chunks. It now finds the same bin — leftmost that fits, so packing is byte-identical — through a max-segment-tree over bin capacities (`_OpenBins`): **9.5 s at 1.28M examples**. Verified against the old form on 40 random inputs and by rebuilding `val_smoltalk2_2000_3413f960` byte-for-byte.

### Corpus / throughput reference
- `dataset_mix`: `auto_math_text` (1.95M rows), `stanford` (1.02M), `stories` (capped 1.8M of 4.99M), `web_samples_v2` (capped 1.8M of 10.3M), `wikihow` (179k), `openstax` (126k), `khanacademy` (24k). `web_samples_v1` is excluded. **`load_docs` takes val from the FRONT of the pool**, train from what follows, so raising `n_train_docs` cannot swallow val docs.
- Encoded under **v2** (`c19dd0bf`): **4.8002B train tokens** (6,894,565 docs), 3.46M val tokens (5000 docs), **5.008 bytes/token** (`tokens_per_byte` 0.1997 train / 0.19996 val), `sep_frac` 0.001436 train / 0.001442 val. One epoch at batch 48 × block 1024 × 4 GPUs (196,608 tokens/step) = **24,415 steps**. (v1 `3c061ac0` was 4.784B tokens at 4.99 bytes/token; the +0.34% is v2's compression, matching the measured 0.315% — not a change in what documents are present, which is identical.)
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

- No test suite, linter, or formatter is configured. Correctness is checked with inline asserts (tokenizer round-trips, attention causality) and ad-hoc verification scripts.
- `data/` (encoded corpora) and `checkpoints/` (model weights) are large and local-only.
- It's a **base** model (next-token completion), not instruction-tuned — it continues prompts, it doesn't answer them. Cosmopedia is prompt/response data but only the `text` column is trained on; `prompt` is the instruction given to Mixtral to *generate* the text, not something to train on.
