"""Measuring what SFT actually changed.

Val loss says the model fits the instruction distribution. It says nothing about the three
things SFT is for, so this measures those directly:

- :func:`format_compliance` — does it answer and then *stop*? Greedy decode, count how often
  ``<|im_end|>`` arrives before the cap, how often a stray role marker appears, and how often
  the model relapses into base-model behaviour by emitting the document separator.
- :func:`multiple_choice` — did it forget? Length-normalised log-likelihood over answer
  options, scored through the ordinary forward pass. Small models forget fast and it does not
  show up in samples; it shows up as several points of accuracy.
- :func:`repetition_rate` — does it loop? A masked val loss is blind to degeneration.

All string matching and log-likelihood: no judge model, no API. Everything is greedy and
seeded, so two runs of the same checkpoint give the same numbers.
"""
import torch
import torch.nn.functional as F

from slm import chat

MAX_LOGIT_ELEMS = 64_000_000    # ~256 MB of fp32 logits per decode step; see greedy_batch


def _vocab_size(model) -> int:
    """``lm_head``'s width, unwrapping ``torch.compile``; 0 if it cannot be read."""
    core = getattr(model, "_orig_mod", model)
    return int(getattr(getattr(core, "lm_head", None), "out_features", 0))


@torch.no_grad()
def greedy_batch(model, tokenizer, prompts: list[str], *, max_new_tokens: int,
                 block_size: int, device, stop_id: int,
                 max_logit_elems: int = MAX_LOGIT_ELEMS) -> list[list[int]]:
    """Greedy-decode a batch of prompts, returning each continuation's token ids.

    Batched because :func:`slm.generate.stream` re-runs the full forward per token for one
    sequence: 100 prompts x 256 tokens is ~30 minutes there against well under a minute here.
    Still no KV cache — the win is the batch dimension, which is enough for an eval you run
    after every experiment.

    Left-padding would need an attention mask the model does not take, so prompts are grouped
    by length and each group is decoded as a rectangle.

    A wide group is split further so the ``(B, T, vocab_size)`` logits stay under
    ``max_logit_elems``. The forward returns logits for every position while this reads only
    the last row, so the rest is pure allocation — 40 prompts x 1024 positions x 32000 ids is
    5.2 GiB in fp32, built and freed once per generated token, and it is the same tensor that
    triggers the training-side OOM. Sub-batching changes nothing about the output.
    """
    encoded = [tokenizer.encode(p) for p in prompts]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    out: list[list[int]] = [[] for _ in prompts]
    vocab = _vocab_size(model)

    i = 0
    while i < len(order):
        # One rectangle per run of equal-length prompts.
        j = i
        while j < len(order) and len(encoded[order[j]]) == len(encoded[order[i]]):
            j += 1
        n_ctx = len(encoded[order[i]])
        t_max = min(block_size, n_ctx + max_new_tokens)      # the forward crops to block_size
        rows = max(1, max_logit_elems // (t_max * vocab)) if vocab else j - i

        for k in range(i, j, rows):
            idx = order[k:min(k + rows, j)]
            ids = torch.tensor([encoded[m] for m in idx], device=device)
            done = torch.zeros(len(idx), dtype=torch.bool, device=device)
            for _ in range(max_new_tokens):
                logits, _ = model(ids[:, -block_size:])
                nxt = logits[:, -1, :].argmax(-1, keepdim=True)
                del logits          # the next step allocates a fresh one; never hold two
                done |= nxt.squeeze(1) == stop_id
                ids = torch.cat([ids, nxt], dim=1)
                if bool(done.all()):
                    break
            for row, m in enumerate(idx):
                gen = ids[row, n_ctx:].tolist()
                out[m] = gen[:gen.index(stop_id) + 1] if stop_id in gen else gen
        i = j
    return out


def format_compliance(model, tokenizer, questions: list[str], *, block_size: int, device,
                      max_new_tokens: int = 256, chat_mode: bool = True) -> dict:
    """Fraction that answer and stop cleanly, plus the ways they fail to.

    ``chat_mode=False`` runs the same prompts through the base model for a delta. The base
    model should score ~0 on ``stopped``: if it does not, the prompts are not testing
    instruction-following.
    """
    stop_id = tokenizer.special_id(chat.IM_END if chat_mode else chat.EOT)
    prompts = [chat.render_prompt([{"role": "user", "content": q}], tokenizer.declared_specials) if chat_mode else q
               for q in questions]
    gens = greedy_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens,
                        block_size=block_size, device=device, stop_id=stop_id)

    im_start = tokenizer.special_id(chat.IM_START)
    eot = tokenizer.special_id(chat.EOT)
    stopped = [g and g[-1] == stop_id for g in gens]
    texts = [tokenizer.decode(g) for g in gens]
    return {
        "n": len(gens),
        "stopped": sum(map(bool, stopped)) / len(gens),
        "mean_len": sum(len(g) for g in gens) / len(gens),
        "hit_cap": sum(len(g) >= max_new_tokens for g in gens) / len(gens),
        "stray_role": sum(im_start in g for g in gens) / len(gens),
        "relapsed_to_eot": sum(eot in g[:-1] for g in gens) / len(gens),
        "repetition": repetition_rate(texts),
        "samples": texts[:3],
    }


def repetition_rate(texts: list[str], *, n: int = 20, times: int = 3) -> float:
    """Fraction of generations containing an ``n``-word span repeated ``times`` or more.

    Catches the degenerate looping that a masked val loss cannot see, because a loop is
    locally high-probability at every step.
    """
    hits = 0
    for t in texts:
        words = t.split()
        grams: dict[tuple, int] = {}
        for i in range(len(words) - n + 1):
            g = tuple(words[i:i + n])
            grams[g] = grams.get(g, 0) + 1
        hits += any(c >= times for c in grams.values())
    return hits / max(1, len(texts))


@torch.no_grad()
def multiple_choice(model, tokenizer, items: list[dict], *, block_size: int, device) -> dict:
    """Accuracy over ``{"question", "options", "answer"}`` items by length-normalised loglik.

    Normalising by token count is what stops the model simply preferring short options. Runs
    through the ordinary ``model(x, targets)`` path — no generation, so it is seconds rather
    than minutes, which is what makes it usable as a gate.

    An over-long question+option pair loses tokens off the **front**, so the answer span
    slides back by exactly that many. Measuring it against the untruncated question instead
    starts the score slice past the answer — usually past the end, giving -inf — and the item
    is then decided by truncation rather than by the model.
    """
    correct = 0
    for item in items:
        scores = []
        n_question = len(tokenizer.encode(item["question"]))     # same for every option
        for opt in item["options"]:
            ids = tokenizer.encode(f"{item['question']} {opt}")
            n_ctx = max(0, n_question - max(0, len(ids) - block_size))
            ids = ids[-block_size:]
            x = torch.tensor([ids], device=device)
            logits, _ = model(x)
            logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
            tgt = x[0, 1:]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[max(0, n_ctx - 1):]
            scores.append(float(tok_lp.mean()) if len(tok_lp) else float("-inf"))
        correct += int(max(range(len(scores)), key=scores.__getitem__) == item["answer"])
    return {"n": len(items), "accuracy": correct / max(1, len(items))}
